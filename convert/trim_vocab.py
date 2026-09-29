"""Trim a GLiNER2.5 Unigram vocabulary to a corpus (docs/RESEARCH.md).

build-corpus OUT --argmin DIR --texts FILE
                        every string in argmin content/families and
                        content/lessons plus the 1000 public corpus texts,
                        one text per line.
trim SRC DST CORPUS     trimmed HF checkpoint at DST plus trim-report.json;
                        export kleinhirn weights afterwards with
                        export_weights.py <name>.
eval ORIG TRIM CORPUS REPORT
                        tokenization and argmax parity on corpus texts and on
                        held-out k9 texts (agnews/emotiondair/banking77).

Kept tokens: every id the corpus tokenizes to, all single characters, all
byte-fallback pieces (<0xNN>), all added/special/schema tokens and the unk id.
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from check_wrapper import LABELS, TASK  # noqa: E402
from gliner2_export import (  # noqa: E402
    GLiNER2ClassificationExport,
    load_processor,
    prepare_tasks_with_processor,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/corpus/texts1000.json"
K9_TASKS = ROOT / "models/k9-data/tasks"
WS_RE = re.compile(r"\s+")
BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
EVAL_N = 300
MAX_OPTIONS = 16


def _iter_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)


def build_corpus(out: Path, argmin: Path, texts_path: Path):
    texts = set()
    corpus = json.loads(texts_path.read_text())
    texts.update(corpus["texts"] if isinstance(corpus, dict) else corpus)
    # Schema strings are part of every request: a label that segments
    # differently in the trimmed tokenizer changes its marker embedding.
    texts.add(TASK)
    texts.update(LABELS)
    for pool in ("families", "lessons"):
        for path in sorted((argmin / pool).glob("*.json")):
            texts.update(_iter_strings(json.loads(path.read_text())))
    lines = sorted(
        line for s in texts if (line := WS_RE.sub(" ", s).strip()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"corpus: {len(lines)} texts -> {out}")


def keep_ids(tok_json: dict, used: set[int]) -> tuple[list[int], dict]:
    vocab = tok_json["model"]["vocab"]
    kept, why = set(used), {"corpus": len(used), "single": 0, "byte": 0,
                          "added": 0}
    for i, (piece, _) in enumerate(vocab):
        if i in kept:
            continue
        if BYTE_RE.match(piece):
            kept.add(i)
            why["byte"] += 1
        elif len(piece.replace("▁", "")) == 1:
            kept.add(i)
            why["single"] += 1
    for a in tok_json.get("added_tokens") or []:
        if a["id"] not in kept:
            why["added"] += 1
        kept.add(a["id"])
    kept.add(tok_json["model"].get("unk_id", 3))
    return sorted(kept), why


def trim(src: Path, dst: Path, corpus_path: Path):
    from tokenizers import Tokenizer

    tok_json = json.loads((src / "tokenizer.json").read_text())
    lines = corpus_path.read_text().splitlines()
    used = set()
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    for enc in tok.encode_batch(lines):
        used.update(enc.ids)
    # Schema context produces pieces the standalone encode never emits
    # (e.g. a word after a literal '▁' loses its prefix), so collect the
    # ids from the real schema path as well.
    processor = load_processor(str(src))
    for text in lines:
        try:
            arrays = prepare_tasks_with_processor(
                processor, text, {TASK: LABELS}, 128, MAX_OPTIONS)
        except Exception:
            continue
        n = int(arrays["attention_mask"][0].sum())
        used.update(int(i) for i in arrays["input_ids"][0][:n])
    kept, why = keep_ids(tok_json, used)
    n_vocab = len(tok_json["model"]["vocab"])
    # Added-token ids can sit beyond the vocab array; keep the original
    # layout: vocab ids first (in order), extra ids appended after.
    kept = sorted(k for k in kept if k < n_vocab) + \
        sorted(k for k in kept if k >= n_vocab)
    remap = {old: new for new, old in enumerate(kept)}

    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if path.is_file() and path.name not in ("model.safetensors",
                                                "tokenizer.json"):
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    out_tok = json.loads((src / "tokenizer.json").read_text())
    out_tok["model"]["vocab"] = [tok_json["model"]["vocab"][i]
                                 for i in kept if i < n_vocab]
    for a in out_tok.get("added_tokens") or []:
        a["id"] = remap[a["id"]]
    (dst / "tokenizer.json").write_text(json.dumps(out_tok))

    from safetensors.torch import load_file, save_file
    tensors = load_file(str(src / "model.safetensors"))
    emb = tensors["encoder.embeddings.word_embeddings.weight"]
    assert kept[-1] < emb.shape[0], f"kept id {kept[-1]} >= {emb.shape[0]} rows"
    tensors["encoder.embeddings.word_embeddings.weight"] = \
        emb[torch.tensor(kept)].contiguous()
    save_file(tensors, str(dst / "model.safetensors"))

    cfg = json.loads((dst / "config.json").read_text())
    cfg["vocab_size"] = len(kept)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    report = {
        "src": str(src), "corpus": str(corpus_path), "corpusTexts": len(lines),
        "checkpointSha256": hashlib.sha256(
            (src / "model.safetensors").read_bytes()).hexdigest(),
        "vocabBefore": len(tok_json["model"]["vocab"]),
        "vocabAfter": len(kept), "kept": why,
        "embRows": int(emb.shape[0]),
    }
    (dst / "trim-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def _argmax(wrapper, processor, text, length=128):
    arrays = prepare_tasks_with_processor(
        processor, text, {TASK: LABELS}, length, MAX_OPTIONS)
    tensors = tuple(
        torch.from_numpy(arrays[k])
        for k in ("input_ids", "attention_mask", "marker_indices",
                  "marker_mask"))
    with torch.no_grad():
        logits, _ = wrapper(*tensors)
    mask = arrays["marker_mask"][0] > 0.5
    return int(np.argmax(logits[0].numpy()[mask]))


def _token_strs(tok, text):
    return [tok.id_to_token(i) for i in tok.encode(text).ids]


def evaluate(orig: Path, trimmed: Path, corpus_path: Path, report: Path):
    from gliner2 import AutoExtractor
    from tokenizers import Tokenizer

    rng = random.Random(0)
    corpus = corpus_path.read_text().splitlines()
    inside = rng.sample(corpus, min(EVAL_N, len(corpus)))
    outside = []
    for name in ("agnews", "emotiondair", "banking77"):
        path = K9_TASKS / f"{name}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            outside += [r["text"] for r in rng.sample(
                rows, min(EVAL_N // 3, len(rows)))]

    tok_o = Tokenizer.from_file(str(orig / "tokenizer.json"))
    tok_t = Tokenizer.from_file(str(trimmed / "tokenizer.json"))
    nat_o = AutoExtractor.from_pretrained(str(orig), map_location="cpu").eval()
    nat_t = AutoExtractor.from_pretrained(str(trimmed),
                                          map_location="cpu").eval()
    wrap_o, wrap_t = (GLiNER2ClassificationExport(nat_o),
                      GLiNER2ClassificationExport(nat_t))
    proc_o, proc_t = (load_processor(str(orig)), load_processor(str(trimmed)))

    result = {}
    for name, texts in (("inCorpus", inside), ("outOfCorpus", outside)):
        same_tok = same_arg = skipped = 0
        for text in texts:
            if _token_strs(tok_o, text) == _token_strs(tok_t, text):
                same_tok += 1
            try:
                same_arg += int(_argmax(wrap_o, proc_o, text)
                                == _argmax(wrap_t, proc_t, text))
            except Exception:
                skipped += 1
        n = len(texts)
        result[name] = {
            "n": n, "skipped": skipped,
            "identicalTokenization": same_tok / n,
            "changedTokenization": 1 - same_tok / n,
            "argmaxAgreement": same_arg / max(n - skipped, 1),
        }
        print(name, json.dumps(result[name]))
    report.parent.mkdir(parents=True, exist_ok=True)
    result["orig"] = str(orig)
    result["trimmed"] = str(trimmed)
    report.write_text(json.dumps(result, indent=1))


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args else ""
    if mode == "build-corpus":
        ap = argparse.ArgumentParser()
        ap.add_argument("out", type=Path)
        ap.add_argument("--argmin", type=Path, required=True)
        ap.add_argument("--texts", type=Path, default=CORPUS)
        ns = ap.parse_args(args)
        build_corpus(ns.out, ns.argmin, ns.texts)
    elif mode == "trim":
        trim(Path(args[0]), Path(args[1]), Path(args[2]))
    elif mode == "eval":
        evaluate(Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
