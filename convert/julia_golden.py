"""Golden generation for Julia 1 (K8).

Reads the pinned Hugging Face checkout under models/julia-1/repo (checkpoint +
tokenizer + julia package) and the published parity cases under
models/julia-1/onnx/parity-cases.json (100 requests with the original fp32
logits from Supersonic Labs).

Outputs:

1. tests/golden/julia-1/parity100.json
   All 100 requests, strict encoding (max_length 1024, head_length 256):
   input_ids, attention_mask, markers, qtype, the published logits (gate
   reference) and our fp32 recomputation plus the max deviation between them.
2. models/julia-1/golden/layers.{bin,index.json}
   The 8 longest requests (all > 128 tokens, so local window-128 attention is
   exercised against the global layers): hidden states after the embedding
   LayerNorm, after every encoder layer, after final_norm, after the type
   embedding add, after both head layers, at the marker positions, plus logits.
3. tests/golden/julia-1/tokenizer_ids.json
   Raw tokenizer ids (add_special_tokens=False) for every head/option/state
   text of the 100 requests, the 1000 public corpus texts and a small
   multilingual probe set (Portuguese, German, Korean, Emoji).

Usage: .venv-julia/bin/python convert/julia_golden.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "models/julia-1/repo"
CASES = ROOT / "models/julia-1/onnx/parity-cases.json"
CORPUS = ROOT / "tests/corpus/texts1000.json"
sys.path.insert(0, str(REPO))
from julia.data import Collator, sequence  # noqa: E402
from julia.model import JuliaDecisionModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MAX_LENGTH = 1024
HEAD_LENGTH = 256

MULTILINGUAL = [
    "O cliente não conseguiu abrir a porta porque a chave estava presa.",
    "Der Student beantwortete die Frage trotz der schwierigen Formulierung.",
    "그녀는 시험에서 가장 높은 점수를 받았기 때문에 상을 받았다.",
    "Das Ergebnis ✓ wurde mit 🎯 und 🚀 bestätigt — alles 👍.",
    "Escolha a opção que melhor descreve a situação descrita acima.",
    "한국어 텍스트도 토크나이저가 올바르게 처리해야 한다 🇰🇷.",
]


def corpus_texts():
    doc = json.loads(CORPUS.read_text())
    return doc["texts"] if isinstance(doc, dict) else doc


def main():
    from transformers import AutoTokenizer  # noqa: E402 re-import for clarity
    tokenizer = AutoTokenizer.from_pretrained(REPO / "tokenizer", trust_remote_code=False)
    cases = json.loads(CASES.read_text())
    model = JuliaDecisionModel.from_pretrained(REPO, memory_map=True).eval()
    # Eager attention for golden capture: it is the textbook math the numpy
    # spec and the WGSL kernels implement. The default sdpa fused kernel
    # deviates up to ~6e-2 on individual hidden positions (verified: sdpa vs
    # eager on case0 shows the same spikes as numpy vs sdpa).
    model.encoder.config._attn_implementation = "eager"
    collate = Collator(tokenizer, MAX_LENGTH, HEAD_LENGTH)

    # 1. The 100 parity requests: strict encoding, fp32 logits, verify the
    #    published reference numbers against this runtime.
    encoded = []
    items = []
    recomputed = []
    for case in cases:
        request = case["request"]
        e = sequence(tokenizer, request, MAX_LENGTH, HEAD_LENGTH, strict=True)
        encoded.append(e)
        items.append({"request": request, "seq_len": len(e["ids"]),
                      "input_ids": e["ids"], "markers": e["markers"],
                      "qtype": e["qtype"], "logits": case["pytorch_logits"]})
    for start in range(0, len(cases), 4):
        group = cases[start:start + 4]
        rows = [dict(g["request"], _encoded=encoded[start + i])
                for i, g in enumerate(group)]
        batch = collate(rows, include_targets=False)
        with torch.inference_mode():
            logits = model(**batch)
        for i, g in enumerate(group):
            recomputed.append(logits[i, :len(g["request"]["options"])].tolist())
    max_pub_diff = 0.0
    choices_match = 0
    for item, rec in zip(items, recomputed):
        pub = item["logits"]
        max_pub_diff = max(max_pub_diff, max(abs(a - b) for a, b in zip(pub, rec)))
        choices_match += int(np.argmax(pub) == np.argmax(rec))
        item["recomputed_logits"] = rec
    golden = {
        "model": "julia-1", "source": "SupersonicLabs/Julia-1-ONNX parity-cases.json",
        "revision": "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8",
        "encoding": "strict", "max_length": MAX_LENGTH, "head_length": HEAD_LENGTH,
        "count": len(items), "max_published_vs_recomputed": max_pub_diff,
        "choices_match": choices_match, "items": items,
    }
    out = ROOT / "tests/golden/julia-1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity100.json").write_text(json.dumps(golden))
    print(f"parity100: {choices_match}/100 argmax, max published-vs-recomputed {max_pub_diff:.3e}")

    # 2. Layer-state goldens on the 8 longest requests, captured in float64.
    #    Alongside, the f32 states are stored as the "conditioning band":
    #    |f32ref - f64ref| per element, i.e. the model's own precision error.
    order = sorted(range(len(items)), key=lambda i: -items[i]["seq_len"])[:8]
    tensors: dict[str, np.ndarray] = {}
    tensors32: dict[str, np.ndarray] = {}
    meta = []
    for rank, idx in enumerate(order):
        item = items[idx]
        e = encoded[idx]
        n = item["seq_len"]
        ids = torch.tensor([e["ids"]])
        attn = torch.ones_like(ids)
        model.float()
        with torch.inference_mode():
            enc32 = model.encoder(input_ids=ids, attention_mask=attn,
                                  output_hidden_states=True)
            states32 = list(enc32.hidden_states)
        model.double()
        with torch.inference_mode():
            enc = model.encoder(input_ids=ids, attention_mask=attn,
                                output_hidden_states=True)
            # hidden_states: embeddings output + one entry per encoder layer;
            # last_hidden_state is after final_norm.
            states = list(enc.hidden_states)
            typed = enc.last_hidden_state + model.type_emb(
                torch.tensor([e["qtype"]]))[:, None, :]
            h = typed
            head_states = []
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=~attn.bool())
                head_states.append(h)
            pos = torch.tensor([e["markers"]])[:, :, None].expand(-1, -1, h.shape[-1])
            markers = h.gather(1, pos)
            scores = model.scorer(markers).squeeze(-1)
        names = (["emb"] + [f"layer{l}" for l in range(len(states) - 1)]
                 + ["final", "typed", "head0", "head1", "markers", "logits"])
        arrays = ([s[0, :n].numpy() for s in states]
                  + [enc.last_hidden_state[0, :n].numpy(), typed[0, :n].numpy()]
                  + [s[0, :n].numpy() for s in head_states]
                  + [markers[0].numpy(), scores[0].numpy()])
        # Layer states are stored in float64: at a few positions the network
        # is so ill-conditioned that torch f32 itself deviates ~4e-2 from its
        # f64 result, so only an f64 reference can prove formula correctness.
        for name, arr in zip(names, arrays):
            key = f"case{rank}.{name}"
            tensors[key] = np.asarray(arr, dtype=np.float64)
        for l, s32 in enumerate(states32):
            tensors32[f"case{rank}.{'emb' if l == 0 else f'layer{l - 1}'}"] = (
                s32[0, :n].numpy().astype(np.float64))
        meta.append({"case": rank, "source_index": idx, "seq_len": n,
                     "n_options": len(e["markers"]), "request": item["request"]})
        print(f"layer golden case{rank}: source {idx}, seq_len {n}, "
              f"logits {scores[0].numpy().round(4).tolist()}")
    gdir = REPO.parent / "golden"
    gdir.mkdir(exist_ok=True)
    blob = bytearray()
    index = {}
    for name, arr in tensors.items():
        raw = arr.tobytes()
        index[name] = {"offset": len(blob), "shape": list(arr.shape)}
        blob += raw
    (gdir / "layers.bin").write_bytes(bytes(blob))
    blob32 = bytearray()
    index32 = {}
    for name, arr in tensors32.items():
        raw = arr.tobytes()
        index32[name] = {"offset": len(blob32), "shape": list(arr.shape)}
        blob32 += raw
    (gdir / "layers.f32.bin").write_bytes(bytes(blob32))
    (gdir / "layers.index.json").write_text(
        json.dumps({"cases": meta, "tensors": index, "tensors32": index32}))

    # 3. Tokenizer ids for every encoded text plus extra corpora.
    texts = {}
    for case in cases:
        r = case["request"]
        texts.setdefault(f"{r.get('type', 'choice')} question: {r['question']}", None)
        for opt in r["options"]:
            texts.setdefault(" " + opt, None)
        state = r["state"] if isinstance(r["state"], str) else json.dumps(
            r["state"], ensure_ascii=False)
        texts.setdefault(state, None)
    for t in corpus_texts():
        texts.setdefault(t, None)
    for t in MULTILINGUAL:
        texts.setdefault(t, None)
    ids_out = {}
    for text in texts:
        ids_out[text] = tokenizer(text, add_special_tokens=False)["input_ids"]
    (out / "tokenizer_ids.json").write_text(json.dumps(ids_out, ensure_ascii=False))
    print(f"tokenizer_ids: {len(ids_out)} texts")


if __name__ == "__main__":
    main()
