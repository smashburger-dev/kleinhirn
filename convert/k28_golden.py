"""PyTorch fp32 goldens for K28 models (K28.2 step 2).

Reference: PyTorch fp32 on the CPU, eager attention, one thread, deterministic
algorithms, every input alone (batch 1, no padding). Texts come from
tests/corpus/texts1000.json.

Per model, see docs/K28_2_WORKORDER.md:
- tests/golden/k28/<slug>/<task>.json        small goldens (in Git)
- models/k28/<slug>/golden/*.bin + index.json large goldens (never in Git):
  token logits, embedding vectors, layer states, tokenizer goldens
- models/k28/<slug>/tokenizer.json           only when the repo has none

Every file is written deterministically (sorted keys, fixed separators, no
timestamp). --verify recomputes everything and compares sha256 per file
against the files on disk without writing them.

K28.3 (--tokenizers-only): no weights are loaded. Per model one file
models/k28/<slug>/golden/tokenizer-cases.json holds the tokenizer goldens of
docs/K28_3_WORKORDER.md step 1 (singles, NLI pairs, truncation pairs, composed
long texts, extra texts), and bench/results/k28-tokenizers-wordpiece.json lists
the models per sha256 of tokenizer.json.

K28.5 (--pilot-k28.5): the 16 RoBERTa, XLM-R and DistilBERT models of
docs/K28_5_WORKORDER.md. DistilBERT takes no token_type_ids; the length limit
subtracts the RoBERTa position offset.

K28.6 (--pilot-k28.6): the 11 DeBERTa and ModernBERT models of docs/K28_6_WORKORDER.md. Tokenization
of every golden goes through tokenizer.json (tokenizers.Tokenizer.from_file); the file header says
tokenizerReference. --tag NAME writes bench/results/k28-golden[-verify]-NAME.json.

Usage: .venv-k28/bin/python convert/k28_golden.py <model-id>... | --pilot | --pilot-k28.5 | --pilot-k28.6 [--verify] [--tag NAME]
       .venv-k28/bin/python convert/k28_golden.py --tokenizers-only --wordpiece|--all-nonwordpiece [--verify]
"""
import argparse
import contextlib
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import (  # noqa: E402
    ROOT, load_entry, model_dir, position_offset, repo_dir, resolve_ids, slug, weight_file,
)

TEXTS_FILE = ROOT / "tests/corpus/texts1000.json"
GOLDEN_DIR = ROOT / "tests/golden/k28"
NLI_LABELS = ["politics", "sports", "technology", "business", "health"]
HYPOTHESIS = "This example is {}."
N_SEQ_CORPUS, N_SEQ_LONG = 170, 30
N_NLI_TEXTS = 40
N_RERANK_QUERIES, N_PASSAGES = 20, 10
N_TOKEN_TEXTS = 200
N_EMBED_TEXTS = 200
N_HIDDEN = 8
SEQ_LONG_TARGET = 512


def dumps(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def f32list(arr) -> list:
    """float32 values as Python floats (repr of the float32 value widened to f64)."""
    return [float(x) for x in np.asarray(arr, dtype=np.float32).reshape(-1)]


class Bins:
    """Concatenated little-endian arrays, one file per group, plus an index."""

    def __init__(self):
        self.buffers: dict[str, bytearray] = {}
        self.entries: list[dict] = []

    def add(self, file: str, name: str, arr, dtype) -> dict:
        arr = np.ascontiguousarray(np.asarray(arr).astype(np.dtype(dtype).newbyteorder("<")))
        raw = arr.tobytes()
        buf = self.buffers.setdefault(file, bytearray())
        self.entries.append({
            "file": file, "name": name, "dtype": np.dtype(dtype).name,
            "shape": list(arr.shape), "offset": len(buf), "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        })
        buf += raw
        return {"file": file, "name": name}

    def file_hashes(self) -> dict:
        return {f: sha256_bytes(bytes(b)) for f, b in sorted(self.buffers.items())}


def corpus_texts() -> list[str]:
    doc = json.loads(TEXTS_FILE.read_text())
    return doc["texts"] if isinstance(doc, dict) else doc


def composed_texts(texts, length):
    """Deterministic composed texts (rule of convert/golden.py, yields the indices too)."""
    rng = random.Random(1000 + length)
    seen = set()
    while True:
        n = rng.randint(max(3, length // 40), max(4, length // 20))
        idxs = tuple(sorted(rng.randrange(len(texts)) for _ in range(n)))
        if idxs in seen:
            continue
        seen.add(idxs)
        yield idxs, " ".join(texts[i] for i in idxs)


def load_tokenizer(model_id: str, entry: dict, allow_spm: bool = False):
    from transformers import AutoTokenizer

    repo = repo_dir(model_id)
    if (repo / "spm.model").exists() and not (repo / "tokenizer.json").exists():
        # spm.model only: AutoTokenizer would need sentencepiece, which .venv-k28 does not have (transformers
        # 5.0.0 then falls back to tiktoken and fails). The reference is tokenizer.json anyway, so
        # convert/k28_spm.py builds it from the model file and the helper fields come from tokenizer_config.json.
        if not allow_spm:
            raise SystemExit(f"{model_id}: spm.model only, the tokenizers-only run needs --reference json")
        from k28_spm import convert_model
        from transformers import PreTrainedTokenizerFast
        path = convert_model(model_id)
        cfg = json.loads((repo / "tokenizer_config.json").read_text())
        tok = PreTrainedTokenizerFast(tokenizer_file=str(path),
                                      model_max_length=int(cfg.get("model_max_length", 512)))
        return tok, sha256_bytes(path.read_bytes()), {}
    try:
        tok = AutoTokenizer.from_pretrained(str(repo))
    except KeyError:
        # transformers 5.0.0 rebuilds an XLM-R tokenizer from a Unigram vocab list and
        # fails on a BPE tokenizer.json (Setur/BRAGD). Load tokenizer.json as it is.
        from transformers import PreTrainedTokenizerFast
        if not (repo / "tokenizer.json").exists():
            raise
        tok = PreTrainedTokenizerFast.from_pretrained(str(repo))
    extra = {}
    if not (repo / "tokenizer.json").exists():
        out = model_dir(model_id) / "tokenizer.json"
        raw_path = out.with_name("tokenizer.json.tmp")
        tok.backend_tokenizer.save(str(raw_path))
        extra[str(out.relative_to(ROOT))] = raw_path.read_bytes()
        raw_path.unlink()
        tok_json = extra[str(out.relative_to(ROOT))]
    else:
        tok_json = (repo / "tokenizer.json").read_bytes()
    if not getattr(tok, "is_fast", False):
        raise SystemExit(f"{model_id}: no fast tokenizer")
    return tok, sha256_bytes(tok_json), extra


@contextlib.contextmanager
def json_tokenizer(model_id: str, extra: dict):
    """RawTokenizer on the tokenizer.json of the model (K28.6 step 0).

    The reference of tokenization is tokenizer.json through tokenizers.Tokenizer.from_file. A repo
    without the file gets the one AutoTokenizer converted (load_tokenizer wrote it into extra).
    """
    repo = repo_dir(model_id)
    tmp = None
    if (repo / "tokenizer.json").exists():
        path = repo / "tokenizer.json"
    elif extra:
        tmp = repo / "_tmp_tokenizer.json"
        tmp.write_bytes(next(iter(extra.values())))
        path = tmp
    else:
        path = model_dir(model_id) / "tokenizer.json"  # written by k28_spm.py
    try:
        yield RawTokenizer(path, repo)
    finally:
        if tmp is not None:
            tmp.unlink()


def model_length(cfg, tok, st_max, model_id) -> int:
    limits = [512, cfg.max_position_embeddings - position_offset(cfg, model_id)]
    if tok.model_max_length and tok.model_max_length < 10**9:
        limits.append(int(tok.model_max_length))
    if st_max:
        limits.append(int(st_max))
    return min(limits)


def encode(tok, first, second=None, length=512, truncation=True):
    enc = tok(first, second, truncation=truncation, max_length=length,
              return_offsets_mapping=True, return_token_type_ids=True)
    return {
        "input_ids": list(enc["input_ids"]),
        "token_type_ids": list(enc["token_type_ids"]),
        "attention_mask": list(enc["attention_mask"]),
        "offsets": [list(o) for o in enc["offset_mapping"]],
        "word_ids": [-1 if w is None else w for w in enc.word_ids()],
    }


def tensors_of(enc, model):
    out = {
        "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
        "attention_mask": torch.ones(1, len(enc["input_ids"]), dtype=torch.long),
    }
    if model.config.model_type != "distilbert":  # DistilBERT has no type embedding
        out["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
    return out


def hidden_states(model, enc):
    with torch.no_grad():
        out = model(**tensors_of(enc, model), output_hidden_states=True)
    return np.stack([h[0].numpy() for h in out.hidden_states]).astype(np.float32)


def add_hidden(bins, model, encs, tag):
    refs = []
    for n, enc in enumerate(encs[:N_HIDDEN]):
        hs = hidden_states(model, enc)
        refs.append(bins.add("hidden.bin", f"{tag}/{n}", hs, np.float32))
    return refs


def load_classifier(model_id: str, kind: str):
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

    cls = AutoModelForTokenClassification if kind == "token" else AutoModelForSequenceClassification
    model = cls.from_pretrained(str(repo_dir(model_id)), attn_implementation="eager",
                                dtype=torch.float32, trust_remote_code=False)
    return model.float().eval()


def logits_of(model, enc):
    with torch.no_grad():
        return model(**tensors_of(enc, model)).logits[0].numpy().astype(np.float32)


def sequence_task(model_id, entry, tok, L, header, texts, bins, files):
    model = load_classifier(model_id, "sequence")
    items, encs = [], []
    for i in range(N_SEQ_CORPUS):
        enc = encode(tok, texts[i], length=L)
        encs.append(enc)
        items.append(("corpus", i, None, enc))
    n_long = 0
    for idxs, text in composed_texts(texts, SEQ_LONG_TARGET):
        if n_long >= N_SEQ_LONG:
            break
        if len(tok(text)["input_ids"]) <= SEQ_LONG_TARGET // 2:
            continue
        enc = encode(tok, text, length=L)
        encs.append(enc)
        items.append(("composed", -1, {"indices": list(idxs), "text": text}, enc))
        n_long += 1
    out, argmax_hist = [], {}
    for source, i, extra, enc in items:
        logits = logits_of(model, enc)
        a = int(logits.argmax())
        argmax_hist[a] = argmax_hist.get(a, 0) + 1
        rec = {"source": source, "text_index": i, "input_ids": enc["input_ids"],
               "token_type_ids": enc["token_type_ids"], "logits": f32list(logits), "argmax": a}
        if extra:
            rec.update(extra)
        out.append(rec)
    hrefs = add_hidden(bins, model, encs, "sequence")
    doc = dict(header, items=out, hidden=hrefs, lengths={"model": L, "long_target": SEQ_LONG_TARGET})
    files[f"{task_json(model_id, header['task'])}"] = doc
    return {"argmax_classes": {str(k): v for k, v in sorted(argmax_hist.items())},
            "n": len(out), "n_long": n_long}, [e for _, _, _, e in items]


def entail_index(id2label):
    """Index of the entailment logit; the HF zero-shot pipeline uses startswith('entail')."""
    hits = [int(i) for i, name in id2label.items() if name.lower().startswith("entail")]
    if len(hits) != 1:
        raise SystemExit(f"NLI labels without a unique entailment label: {id2label}")
    return hits[0]


def nli_task(model_id, entry, tok, L, header, texts, bins, files):
    model = load_classifier(model_id, "sequence")
    ent = entail_index(header["id2label"])
    pairs, encs, per_text = [], [], []
    choice_hist = {}
    for t in range(N_NLI_TEXTS):
        ent_logits = []
        for label in NLI_LABELS:
            enc = encode(tok, texts[t], HYPOTHESIS.format(label), length=L, truncation="only_first")
            logits = logits_of(model, enc)
            encs.append(enc)
            ent_logits.append(float(logits[ent]))
            pairs.append({"text_index": t, "label": label, "input_ids": enc["input_ids"],
                          "token_type_ids": enc["token_type_ids"], "logits": f32list(logits),
                          "argmax": int(logits.argmax())})
        e = np.array(ent_logits, dtype=np.float32)
        p = np.exp(e - e.max())
        p = p / p.sum()
        choice = int(p.argmax())
        choice_hist[NLI_LABELS[choice]] = choice_hist.get(NLI_LABELS[choice], 0) + 1
        per_text.append({"text_index": t, "entail_logits": f32list(e),
                         "probabilities": f32list(p), "zero_shot_choice": choice})
    hrefs = add_hidden(bins, model, encs, "nli")
    argmax_hist = {}
    for p in pairs:
        argmax_hist[p["argmax"]] = argmax_hist.get(p["argmax"], 0) + 1
    doc = dict(header, entail_index=ent, labels=NLI_LABELS, hypothesis=HYPOTHESIS,
               items=pairs, texts=per_text, hidden=hrefs, lengths={"model": L})
    files[task_json(model_id, "nli")] = doc
    return {"argmax_classes": {str(k): v for k, v in sorted(argmax_hist.items())},
            "zero_shot_choices": choice_hist, "n_pairs": len(pairs)}, encs


def rerank_task(model_id, entry, tok, L, header, texts, bins, files):
    model = load_classifier(model_id, "sequence")
    pairs, encs, queries, best_all = [], [], [], []
    for q in range(N_RERANK_QUERIES):
        scores = []
        for p in range(N_PASSAGES):
            pi = 100 + N_PASSAGES * q + p
            enc = encode(tok, texts[q], texts[pi], length=L, truncation=True)
            logits = logits_of(model, enc)
            encs.append(enc)
            scores.append(float(logits[0]))
            pairs.append({"query": q, "passage": p, "text_index": q, "passage_text_index": pi,
                          "input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"],
                          "logit": float(logits[0])})
        s = np.array(scores, dtype=np.float32)
        best_all.append(int(s.argmax()))
        queries.append({"query": q, "scores": f32list(s), "best": int(s.argmax())})
    hrefs = add_hidden(bins, model, encs, "reranking")
    doc = dict(header, items=pairs, queries=queries, truncation="longest_first",
               hidden=hrefs, lengths={"model": L})
    files[task_json(model_id, "reranking")] = doc
    return {"best_passage_per_query": best_all}, encs


def token_task(model_id, entry, tok, L, header, texts, bins, files):
    from transformers import pipeline

    model = load_classifier(model_id, "token")
    pipe = pipeline("token-classification", model=model, tokenizer=tok.fast(L),
                    aggregation_strategy="simple", device="cpu")
    items, encs, hist, n_spans = [], [], {}, 0
    for i in range(N_TOKEN_TEXTS):
        enc = encode(tok, texts[i], length=L)
        encs.append(enc)
        logits = logits_of(model, enc)
        arg = logits.argmax(axis=-1)
        for a in arg:
            hist[int(a)] = hist.get(int(a), 0) + 1
        spans = [{"entity_group": s["entity_group"], "score": float(np.float32(s["score"])),
                  "word": s["word"], "start": int(s["start"]), "end": int(s["end"])}
                 for s in pipe(texts[i])]
        n_spans += len(spans)
        items.append({
            "text_index": i, "input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"],
            "offsets": enc["offsets"], "word_ids": enc["word_ids"], "argmax": arg.tolist(),
            "logits": bins.add("token-classification.bin", f"logits/{i}", logits, np.float32),
            "spans": spans,
        })
    hrefs = add_hidden(bins, model, encs, "token-classification")
    doc = dict(header, items=items, hidden=hrefs, aggregation_strategy="simple",
               lengths={"model": L})
    files[task_json(model_id, "token-classification")] = doc
    return {"argmax_classes": {str(k): v for k, v in sorted(hist.items())},
            "n_spans": n_spans}, encs


def embedding_task(model_id, entry, tok, L_unused, header, texts, bins, files):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(repo_dir(model_id)), device="cpu", trust_remote_code=False,
                             model_kwargs={"attn_implementation": "eager", "dtype": torch.float32})
    st.eval()
    st.float()
    modules = [type(m).__name__ for m in st]
    pool_idx = modules.index("Pooling")
    prompt = None
    if getattr(st, "default_prompt_name", None):
        prompt = st.prompts[st.default_prompt_name]
    items, encs, pooled_all, final_all = [], [], [], []
    max_len = int(st.max_seq_length)
    n_json_differs = 0
    for i in range(N_EMBED_TEXTS):
        text = (prompt or "") + texts[i]
        enc = encode(tok, text, length=max_len)
        ids, tt = enc["input_ids"], enc["token_type_ids"]
        with torch.no_grad():
            st_feats = st.tokenize([text])
            # the ids of tokenizer.json go straight into the module chain of sentence-transformers
            f = {"input_ids": torch.tensor([ids], dtype=torch.long),
                 "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            if "token_type_ids" in st_feats:
                f["token_type_ids"] = torch.tensor([tt], dtype=torch.long)
            pooled = None
            for k, m in enumerate(st):
                f = m(f)
                if k == pool_idx:
                    pooled = f["sentence_embedding"][0].numpy().astype(np.float32).copy()
            chain = f["sentence_embedding"][0].numpy().astype(np.float32)
            final = np.asarray(st.encode(texts[i], batch_size=1, convert_to_numpy=True,
                                         show_progress_bar=False), dtype=np.float32)
        if st_feats["input_ids"][0].tolist() != ids:
            n_json_differs += 1  # encode() would see other ids; the chain on tokenizer.json ids is the golden
        elif np.abs(final - chain).max() > 1e-6:
            raise SystemExit(f"{model_id}: encode() and module chain differ at text {i}")
        pooled_all.append(pooled)
        final_all.append(chain)
        encs.append({"input_ids": ids, "token_type_ids": tt})
        items.append({"text_index": i, "input_ids": ids, "token_type_ids": tt})
    pooled_all = np.stack(pooled_all)
    final_all = np.stack(final_all)
    pref = bins.add("embeddings.bin", "pooled", pooled_all, np.float32)
    fref = bins.add("embeddings.bin", "final", final_all, np.float32)
    hrefs = add_hidden(bins, st[0].auto_model, encs, "embeddings")
    norms = np.linalg.norm(final_all.astype(np.float64), axis=1)
    doc = dict(header, items=items, pooled=pref, final=fref, hidden=hrefs,
               modules=modules, prompt=prompt,
               pooling=json.loads((repo_dir(model_id) / "1_Pooling/config.json").read_text()),
               max_seq_length=int(st.max_seq_length), normalize="Normalize" in modules,
               lengths={"model": int(st.max_seq_length)})
    files[task_json(model_id, "embeddings")] = doc
    return {"final_norm_min": float(norms.min()), "final_norm_max": float(norms.max()),
            "texts_where_encode_has_other_ids": n_json_differs,
            "normalize": "Normalize" in modules, "dim": int(final_all.shape[1])}, None


def tokenizer_goldens(model_id, tok, L, texts, bins):
    """IDs, type ids, mask, offsets and word ids of all texts and the 200 NLI pairs."""
    sets = {}
    for length in sorted({128, L}):
        singles = [encode(tok, t, length=length) for t in texts]
        pairs = [encode(tok, texts[t], HYPOTHESIS.format(lab), length=length, truncation="only_first")
                 for t in range(N_NLI_TEXTS) for lab in NLI_LABELS]
        for tag, encs in (("single", singles), ("pair", pairs)):
            base = f"{tag}{length}"
            lens = np.array([len(e["input_ids"]) for e in encs], dtype=np.int32)
            bins.add("tokenizer.bin", f"{base}/lengths", lens, np.int32)
            for key in ("input_ids", "token_type_ids", "attention_mask", "word_ids"):
                bins.add("tokenizer.bin", f"{base}/{key}",
                         np.concatenate([np.array(e[key], dtype=np.int32) for e in encs]), np.int32)
            bins.add("tokenizer.bin", f"{base}/offsets",
                     np.concatenate([np.array(e["offsets"], dtype=np.int32).reshape(-1, 2) for e in encs]),
                     np.int32)
            sets[base] = {"n": len(encs), "tokens": int(lens.sum()), "max_len": int(lens.max())}
    return sets


EXTRA_TEXTS = [
    "Reset the [SEP] token, then [MASK] the [CLS] marker and [UNK] words: [PAD].",
    "Launch \U0001F680 and \U0001F468\u200d\U0001F469\u200d\U0001F467 with \U0001D54F, \U0001F3F3\ufe0f\u200d\U0001F308 and \U00020BB7 done.",
    "Cafe\u0301 re\u0301sume\u0301, Zu\u0308rich, \u0130stanbul, na\u0308ive, \u1e9e, \ufb01ne, \u00e5ngstro\u0308m, a\u0323\u0302.",
    "\u4e2d\u6587\u6d4b\u8bd5, \u65e5\u672c\u8a9e\u306e\u6587\u7ae0, \ud55c\uad6d\uc5b4 \ubb38\uc7a5 \uc548\ub155 abc\u4e2ddef\u6587.",
    "A " + "supercalifragilisticexpialidocious" * 4 + " and " + "x" * 101 + ", tab\there\nnew\r\nline\u00a0nbsp\u2003em\u0085nel\ufeffbom\u0000nul.",
]
STRATEGIES = ["longest_first", "only_first", "only_second"]


def spec_text(texts, spec):
    if "i" in spec:
        return texts[spec["i"]]
    if "join" in spec:
        return " ".join(texts[i] for i in spec["join"])
    return spec["s"]


def encode_case(tok, first, second, length, truncation):
    try:
        enc = encode(tok, first, second, length=length, truncation=truncation)
        seq = tok(first, second, truncation=truncation, max_length=length,
                  return_offsets_mapping=True)
        enc["sequence_ids"] = [-1 if x is None else x for x in seq.sequence_ids()]
        return dict(enc, ok=True)
    except Exception as exc:  # HF raises for impossible truncations; kept as a case
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def tokenizer_cases(tok, L, texts):
    """Specs plus HF results for every case of K28.3 step 1 (texts by reference)."""
    cases = []

    def add(kind, first, second, length, truncation):
        text2 = None if second is None else spec_text(texts, second)
        res = encode_case(tok, spec_text(texts, first), text2, length, truncation)
        cases.append(dict(kind=kind, first=first, second=second, max_length=length,
                          truncation=truncation, **res))

    for length in sorted({128, L}):
        for i in range(len(texts)):
            add("single", {"i": i}, None, length, "longest_first")
        for t in range(N_NLI_TEXTS):
            for lab in NLI_LABELS:
                add("nli", {"i": t}, {"s": HYPOTHESIS.format(lab)}, length, "only_first")
    for length in (32, 128):
        for strategy in STRATEGIES:
            for i in range(200):
                add("truncate", {"i": i}, {"i": i + 200}, length, strategy)
    n_long = 0
    for idxs, text in composed_texts(texts, SEQ_LONG_TARGET):
        if n_long >= N_SEQ_LONG:
            break
        if len(tok(text)["input_ids"]) <= SEQ_LONG_TARGET // 2:
            continue
        for length in sorted({128, L}):
            add("composed", {"join": list(idxs)}, None, length, "longest_first")
        n_long += 1
    for k, text in enumerate(EXTRA_TEXTS):
        add("extra", {"s": text}, None, 128, "longest_first")
        add("extra_pair", {"s": text}, {"s": EXTRA_TEXTS[(k + 1) % len(EXTRA_TEXTS)]}, 128,
            "longest_first")
    return cases


class RawEncoding:
    """An Encoding of the tokenizers library with the accessors encode() and encode_case() use."""

    def __init__(self, enc):
        self.enc = enc

    def __getitem__(self, key):
        return {"input_ids": self.enc.ids, "token_type_ids": self.enc.type_ids,
                "attention_mask": self.enc.attention_mask,
                "offset_mapping": self.enc.offsets}[key]

    def word_ids(self):
        return self.enc.word_ids

    def sequence_ids(self):
        return self.enc.sequence_ids


class RawTokenizer:
    """tokenizer.json through tokenizers.Tokenizer only, no transformers class in between.

    truncation and padding of the file are replaced by the call's own arguments, as the
    goldens of K28.3 step 1 do.
    """

    def __init__(self, path, repo=None):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.repo = repo
        self.tk = Tokenizer.from_file(str(path))
        self.tk.no_padding()

    def fast(self, model_max_length):
        """PreTrainedTokenizerFast on the same file, for the HF token-classification pipeline."""
        from transformers import PreTrainedTokenizerFast
        if self.repo is not None and (self.repo / "tokenizer.json").exists():
            tok = PreTrainedTokenizerFast.from_pretrained(str(self.repo))
        else:
            tok = PreTrainedTokenizerFast(tokenizer_file=str(self.path))
        tok.model_max_length = model_max_length
        return tok

    def __call__(self, first, second=None, truncation=False, max_length=None, **_):
        if not truncation or max_length is None:
            self.tk.no_truncation()
        else:
            strategy = "longest_first" if truncation is True else truncation
            self.tk.enable_truncation(max_length=max_length, strategy=strategy)
        return RawEncoding(self.tk.encode(first, second))


def tokenizer_only_files(model_id, texts, kind="wordpiece", reference="auto"):
    entry = load_entry(model_id)
    tok, tok_sha, extra = load_tokenizer(model_id, entry, allow_spm=reference == "json")
    from transformers import AutoConfig
    import tokenizers
    import transformers

    cfg = AutoConfig.from_pretrained(str(repo_dir(model_id)))
    st_max = None
    st_cfg = repo_dir(model_id) / "sentence_bert_config.json"
    if st_cfg.exists():
        st_max = json.loads(st_cfg.read_text()).get("max_seq_length")
    L = model_length(cfg, tok, st_max, model_id)
    repo_tj = repo_dir(model_id) / "tokenizer.json"
    own_tj = model_dir(model_id) / "tokenizer.json"  # written by k28_spm.py for an spm.model-only repo
    raw_tj = next(iter(extra.values())) if extra else (repo_tj if repo_tj.exists() else own_tj).read_bytes()
    tj = json.loads(raw_tj)
    if (tj["model"]["type"] == "WordPiece") != (kind == "wordpiece"):
        raise SystemExit(f"{model_id}: tokenizer model is {tj['model']['type']}, list is {kind}")
    if reference == "json":
        # tokenizer.json through the tokenizers library; a file written in this run (a repo
        # without one) is not on disk yet
        if extra:
            tj_path = repo_dir(model_id) / "_tmp_tokenizer.json"
            tj_path.write_bytes(raw_tj)
        else:
            tj_path = repo_tj if repo_tj.exists() else own_tj
        try:
            tok = RawTokenizer(tj_path)
        finally:
            if extra:
                tj_path.unlink()
    cases = tokenizer_cases(tok, L, texts)
    doc = {"header": {"model": model_id, "revision": entry["revision"], "maxLength": L,
                      "tokenizerJsonSha256": tok_sha,
                      "versions": {"transformers": transformers.__version__,
                                   "tokenizers": tokenizers.__version__}},
           "cases": cases}
    files = dict(extra)
    if reference == "json":
        doc["header"]["reference"] = "tokenizers.Tokenizer.from_file(tokenizer.json)"
    name = "tokenizer-cases-json.json" if reference == "json" else "tokenizer-cases.json"
    files = {} if reference == "json" else files
    files[str((model_dir(model_id) / "golden" / name).relative_to(ROOT))] = dumps(doc)
    summary = {"model": model_id, "length": L, "tokenizerJsonSha256": tok_sha,
               "cases": len(cases), "errorCases": sum(1 for c in cases if not c["ok"]),
               "files": {p: sha256_bytes(b) for p, b in sorted(files.items())}}
    return files, summary


def run_tokenizers_only(ids, verify, kind="wordpiece", reference="auto"):
    texts = corpus_texts()
    report, differing, total, by_sha = [], 0, 0, {}
    for model_id in ids:
        try:
            files, summary = tokenizer_only_files(model_id, texts, kind, reference)
        except (ImportError, ValueError) as exc:
            # a slow tokenizer that needs a package outside the frozen venv (spm.model)
            summary = {"model": model_id, "skipped": f"{type(exc).__name__}: {str(exc).strip()[:200]}"}
            report.append(summary)
            print(json.dumps(summary), flush=True)
            continue
        bad = []
        for rel, raw in files.items():
            path = ROOT / rel
            total += 1
            if verify:
                if not path.exists() or sha256_bytes(path.read_bytes()) != sha256_bytes(raw):
                    bad.append(rel)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        summary["differing"] = bad
        differing += len(bad)
        report.append(summary)
        by_sha.setdefault(summary["tokenizerJsonSha256"], []).append(model_id)
        print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)
    res = ROOT / "bench/results"
    if verify:
        out = res / f"k28-golden-verify-tokenizers-{kind}{'-json' if reference == 'json' else ''}.json"
    else:
        out = res / f"k28-golden-tokenizers-{kind}{'-json' if reference == 'json' else ''}.json"
        if (kind == "wordpiece") == (reference == "auto"):  # the list of the models: nonwordpiece from the json run
            (res / f"k28-tokenizers-{kind}.json").write_text(json.dumps(
                {"tokenizers": len(by_sha), "models": sum(map(len, by_sha.values())),
                 "bySha256": {k: sorted(v) for k, v in sorted(by_sha.items())}}, indent=1) + "\n")
    out.write_text(json.dumps({"files_total": total, "files_differing": differing,
                               "models": report}, indent=1, sort_keys=True) + "\n")
    print(f"files {total}, differing {differing}, distinct tokenizer.json {len(by_sha)}")
    if differing:
        sys.exit(1)


def task_json(model_id: str, task: str) -> str:
    return str((GOLDEN_DIR / slug(model_id) / f"{task}.json").relative_to(ROOT))


TASKS = {
    "sequence-classification": sequence_task,
    "nli": nli_task,
    "reranking": rerank_task,
    "token-classification": token_task,
    "embeddings": embedding_task,
}


def build_model(model_id: str, texts: list[str]):
    entry = load_entry(model_id)
    task = entry["task"]
    tok, tok_sha, extra_files = load_tokenizer(model_id, entry, allow_spm=True)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(repo_dir(model_id)))
    st_max = None
    st_cfg = repo_dir(model_id) / "sentence_bert_config.json"
    if task == "embeddings" and st_cfg.exists():
        st_max = json.loads(st_cfg.read_text()).get("max_seq_length")
    L = model_length(cfg, tok, st_max, model_id)
    import sentence_transformers
    import tokenizers
    import transformers

    header = {
        "model": model_id, "revision": entry["revision"],
        "weightFile": weight_file(entry), "weightSha256": entry["weightSha256"][0],
        "task": task, "id2label": {str(k): v for k, v in cfg.id2label.items()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__,
                     "tokenizers": tokenizers.__version__},
        "tokenizerJsonSha256": tok_sha, "maxLength": L, "tokenizerReference": "tokenizer.json",
    }
    bins = Bins()
    docs: dict[str, dict] = {}
    with json_tokenizer(model_id, extra_files) as raw_tok:
        plaus, _ = TASKS[task](model_id, entry, raw_tok, L, header, texts, bins, docs)
        tok_sets = tokenizer_goldens(model_id, raw_tok, L, texts, bins)

    files: dict[str, bytes] = dict(extra_files)
    gdir = model_dir(model_id) / "golden"
    for path, doc in docs.items():
        doc["bins"] = {f: h for f, h in bins.file_hashes().items() if f != "tokenizer.bin"}
        files[path] = dumps(doc)
    for name, buf in bins.buffers.items():
        files[str((gdir / name).relative_to(ROOT))] = bytes(buf)
    index = {"header": header, "files": bins.file_hashes(), "entries": bins.entries,
             "tokenizer_sets": tok_sets}
    files[str((gdir / "index.json").relative_to(ROOT))] = dumps(index)
    return files, {"model": model_id, "task": task, "length": L, "plausibility": plaus,
                   "files": {p: sha256_bytes(b) for p, b in sorted(files.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-k28.5", dest="k285", action="store_true")
    parser.add_argument("--pilot-k28.6", dest="k286", action="store_true")
    parser.add_argument("--tag", help="name part of the result file (default pilot, k28.5, k28.6)")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tokenizers-only", action="store_true")
    parser.add_argument("--wordpiece", action="store_true")
    parser.add_argument("--all-nonwordpiece", action="store_true")
    parser.add_argument("--tokenizer-json", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ids = resolve_ids(args.models, args.pilot, args.wordpiece, args.k285, args.all_nonwordpiece, args.k286)
    if args.tokenizers_only:
        run_tokenizers_only(ids, args.verify, "nonwordpiece" if args.all_nonwordpiece else "wordpiece",
                             "json" if args.tokenizer_json else "auto")
        return
    texts = corpus_texts()
    report, total, differing = [], 0, 0
    for model_id in ids:
        files, summary = build_model(model_id, texts)
        bad = []
        for rel, raw in files.items():
            path = ROOT / rel
            total += 1
            if args.verify:
                if not path.exists() or sha256_bytes(path.read_bytes()) != sha256_bytes(raw):
                    bad.append(rel)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        differing += len(bad)
        summary["differing"] = bad
        report.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)
    tag = args.tag or ("k28.6" if args.k286 else "k28.5" if args.k285 else "pilot")
    kind = "golden-verify" if args.verify else "golden"
    out = ROOT / f"bench/results/k28-{kind}-{tag}.json"
    out.write_text(json.dumps({"files_total": total, "files_differing": differing,
                               "models": report}, indent=1, sort_keys=True) + "\n")
    print(f"files {total}, differing {differing}")
    if differing:
        sys.exit(1)


if __name__ == "__main__":
    main()
