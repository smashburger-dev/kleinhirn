"""Executable specification of the K28 forward pass, numpy only (BERT, RoBERTa, XLM-R,
DistilBERT, DeBERTa-v2/v3 and ModernBERT rows).

Reads ONLY manifest.json and the shards (models/k28/<slug>/f32) and computes in
float64. The forward hangs on `spec` and `head` from the manifest, not on the
family name, so a passing run checks the config reading (src/plan/hf.ts), the
name table and the converter (src/convert/) together, before any kernel exists.

Gates (--check), against the goldens of convert/k28_golden.py:
- G2: largest logit deviation <= 1e-4 (embeddings: smallest cosine >= 0.99999),
  argmax agreement, zero-shot choice (NLI), best passage (reranking), argmax
  per token (token classification), largest deviation per layer on the 8
  layer-state cases.
- G3: every tensor of the f16 manifest is bit-identical to
  numpy.astype(float16) of the f32 tensor.

Usage: .venv-k28/bin/python convert/k28_forward.py <model-id>... | --pilot | --pilot-k28.5 | --pilot-k28.6 [--tag NAME] --check
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import ROOT, model_dir, resolve_ids, slug  # noqa: E402

GOLDEN_DIR = ROOT / "tests/golden/k28"
RESULT = ROOT / "bench/results/k28-forward-pilot.json"
RESULT_K285 = ROOT / "bench/results/k28-forward-k28.5.json"
LOGIT_TOL = 1e-4
COSINE_MIN = 0.99999


def read_manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text())


def load_raw(path: Path, dtype: str) -> tuple[dict, list[tuple[dict, np.ndarray]]]:
    """Manifest plus (entry, array) per manifest entry; verifies the shard sha256."""
    manifest = read_manifest(path)
    shards = []
    for s in manifest["shards"]:
        raw = (path / s["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != s["sha256"] or len(raw) != s["bytes"]:
            raise ValueError(f"{path / s['file']}: sha256 or size differs from the manifest")
        shards.append(raw)
    np_dtype = {"f32": np.float32, "f16": np.float16}[dtype]
    out = []
    for t in manifest["tensors"]:
        if t["dtype"] != dtype:
            raise ValueError(f"{t['name']}: dtype {t['dtype']}, expected {dtype}")
        arr = np.frombuffer(shards[t["shard"]], dtype=np_dtype,
                            count=int(np.prod(t["shape"])), offset=t["offset"]).reshape(t["shape"])
        out.append((t, arr))
    return manifest, out


def load_tensors(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """name -> f64 array; embedding row chunks joined by rowStart."""
    manifest, entries = load_raw(path, "f32")
    tensors: dict[str, np.ndarray] = {}
    for t, arr in entries:
        arr = arr.astype(np.float64)
        tensors[t["name"]] = np.concatenate([tensors[t["name"]], arr]) if t["name"] in tensors else arr
    return manifest, tensors


def layer_norm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def act(name, x):
    if name == "none":
        return x
    if name == "gelu":
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    raise ValueError(f"unknown activation {name}")


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def opt(T, name):
    """A tensor of the manifest, or 0 when the model has no such bias."""
    return T[name] if name in T else 0.0


def rel_index(n, span, max_pos):
    """Bucketed relative positions q - k as in HF make_log_bucket_position (float32 ops)."""
    q = np.arange(n)
    rel = q[:, None] - q[None, :]
    mid = span // 2
    abs_pos = np.where((rel < mid) & (rel > -mid), mid - 1, np.abs(rel))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pos = np.ceil(np.log(abs_pos.astype(np.float32) / np.float32(mid))
                          / np.log(np.float32((max_pos - 1) / mid)) * np.float32(mid - 1)) + mid
    return np.where(abs_pos <= mid, rel, log_pos * np.sign(rel)).astype(np.int64)


def deberta_attention(T, p, spec, q, k, v):
    """Disentangled attention (c2p and p2c), pos_key and pos_query from the manifest."""
    rel = spec["attention"]["rel"]
    span, heads, d, n = rel["buckets"], spec["heads"], spec["headDim"], q.shape[1]
    bucket = rel_index(n, span, rel["maxPositions"])
    c2p_pos = np.clip(bucket + span, 0, 2 * span - 1)
    p2c_pos = np.clip(-bucket + span, 0, 2 * span - 1)  # [j, i] = -bucket(j - i) + span
    split = lambda t: t.reshape(2 * span, heads, d).transpose(1, 0, 2)  # noqa: E731
    pos_k, pos_q = split(T[p + "pos_key"]), split(T[p + "pos_query"])
    c2p = np.take_along_axis(q @ pos_k.transpose(0, 2, 1), np.broadcast_to(c2p_pos, (heads, n, n)), axis=-1)
    p2c = np.take_along_axis(k @ pos_q.transpose(0, 2, 1), np.broadcast_to(p2c_pos, (heads, n, n)), axis=-1)
    scale = np.sqrt(d * (1 + len(rel["types"])))
    return softmax((q @ k.transpose(0, 2, 1) + c2p + p2c.transpose(0, 2, 1)) / scale) @ v


def conv_cols(x, kernel):
    """im2col over one sequence: row i holds x[i + t - pad] for t = 0..kernel-1, zero outside."""
    n, pad = x.shape[0], (kernel - 1) // 2
    padded = np.concatenate([np.zeros((pad, x.shape[1]), x.dtype), x, np.zeros((pad, x.shape[1]), x.dtype)])
    return np.concatenate([padded[t:t + n] for t in range(kernel)], axis=1)


def deberta_conv(T, spec, emb, layer0, eps):
    """HF ConvLayer after layer 0: LN(layer0 + act(conv1d(emb))), the mask is all ones without padding.
    conv.weight is [H, kernel * H] (column t * H + ci = W[c, ci, t])."""
    conv = spec["conv"]
    out = act(conv["act"], linear(conv_cols(emb, conv["kernel"]), T["conv.weight"], T["conv.bias"]))
    return layer_norm(layer0 + out, T["conv.ln.weight"], T["conv.ln.bias"], eps)


def encoder_post(T, spec, ids, tt, capture=None):
    """Post-norm encoder (BERT rows, DeBERTa), one sequence, no padding."""
    emb, blk = spec["embed"], spec["block"]
    relative = spec["attention"]["kind"] == "deberta-relative"
    n = len(ids)
    x = T["embeddings.word.weight"][ids]
    if emb["positions"] == "absolute":
        x = x + T["embeddings.position.weight"][np.arange(n) + emb["positionOffset"]]
    if emb["typeVocab"] > 0:
        x = x + T["embeddings.type.weight"][tt]
    x = layer_norm(x, T["embeddings.LayerNorm.weight"], T["embeddings.LayerNorm.bias"], emb["norm"]["eps"])
    if emb["project"]:
        x = linear(x, T["embeddings.project.weight"], T["embeddings.project.bias"])
    # maskMultiply: the mask is all ones without padding
    if capture is not None:
        capture.append(x.copy())
    heads, d = spec["heads"], spec["headDim"]
    emb_out = x
    for l in range(spec["layers"]):
        p = f"layers.{l}."
        q, k, v = np.split(linear(x, T[p + "qkv.weight"], T[p + "qkv.bias"]), 3, axis=-1)
        split = lambda t: t.reshape(n, heads, d).transpose(1, 0, 2)  # noqa: E731
        q, k, v = split(q), split(k), split(v)
        if relative:
            ctx = deberta_attention(T, p, spec, q, k, v)
        else:
            ctx = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(d)) @ v
        ctx = ctx.transpose(1, 0, 2).reshape(n, -1)
        attn = linear(ctx, T[p + "attn_out.weight"], T[p + "attn_out.bias"])
        x = layer_norm(x + attn, T[p + "attn_ln.weight"], T[p + "attn_ln.bias"], blk["norm"]["eps"])
        mid = act(spec["ffn"]["act"], linear(x, T[p + "ffn_in.weight"], T[p + "ffn_in.bias"]))
        x = layer_norm(x + linear(mid, T[p + "ffn_out.weight"], T[p + "ffn_out.bias"]),
                       T[p + "ffn_ln.weight"], T[p + "ffn_ln.bias"], blk["norm"]["eps"])
        if l == 0 and "conv" in spec:
            x = deberta_conv(T, spec, emb_out, x, blk["norm"]["eps"])
        if capture is not None:
            capture.append(x.copy())
    return x


def rope_tables(n, theta, d):
    """cos and sin of the default RoPE, [n, d] with both halves equal (rotate-half layout)."""
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    freqs = np.arange(n)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def encoder_pre(T, spec, ids, tt, capture=None):
    """ModernBERT: pre-norm, RoPE (global and local theta), sliding window, GeGLU."""
    blk, att = spec["block"], spec["attention"]
    n, heads, d, eps = len(ids), spec["heads"], spec["headDim"], blk["norm"]["eps"]
    x = layer_norm(T["embeddings.word.weight"][ids], T["embeddings.norm.weight"],
                   opt(T, "embeddings.norm.bias"), eps)
    if capture is not None:
        capture.append(x.copy())
    rope = att["rope"]
    tables = {"g": rope_tables(n, rope["thetaGlobal"], d), "l": rope_tables(n, rope["thetaLocal"], d)}
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    for l in range(spec["layers"]):
        p = f"layers.{l}."
        is_global = l % att["window"]["globalEvery"] == 0
        h = x if (blk["firstNormIdentity"] and l == 0) else layer_norm(
            x, T[p + "attn_norm.weight"], opt(T, p + "attn_norm.bias"), eps)
        qkv = linear(h, T[p + "wqkv.weight"], opt(T, p + "wqkv.bias")).reshape(n, 3, heads, d)
        q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))  # [heads, n, d]
        cos, sin = tables["g" if is_global else "l"]
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        scores = q @ k.transpose(0, 2, 1) * att["scale"]
        if not is_global:
            scores = np.where(dist <= att["window"]["half"], scores, -np.inf)
        ctx = (softmax(scores) @ v).transpose(1, 0, 2).reshape(n, -1)
        x = x + linear(ctx, T[p + "attn_out.weight"], opt(T, p + "attn_out.bias"))
        h = layer_norm(x, T[p + "mlp_norm.weight"], opt(T, p + "mlp_norm.bias"), eps)
        inp, gate = np.split(linear(h, T[p + "mlp_in.weight"], opt(T, p + "mlp_in.bias")), 2, axis=-1)
        x = x + linear(act(spec["ffn"]["act"], inp) * gate, T[p + "mlp_out.weight"], opt(T, p + "mlp_out.bias"))
        if capture is not None:
            capture.append(x.copy())
    return layer_norm(x, T["final_norm.weight"], opt(T, "final_norm.bias"), eps)


def encoder(T, spec, ids, tt, capture=None):
    if spec["block"]["order"] == "pre":
        return encoder_pre(T, spec, ids, tt, capture)
    return encoder_post(T, spec, ids, tt, capture)


def run_steps(T, steps, x):
    for s in steps:
        if s["op"] == "dense":
            x = act(s["act"], linear(x, T[s["name"] + ".weight"], opt(T, s["name"] + ".bias")))
        elif s["op"] == "norm":
            x = layer_norm(x, T[s["name"] + ".weight"], opt(T, s["name"] + ".bias"), s["eps"])
        else:
            raise ValueError(f"unsupported head step {s['op']}")
    return x


def head_forward(T, head, x):
    """Returns logits [C] (classify), [n, C] (token) or (pooled, final) (embed)."""
    if head["type"] == "classify":
        pooled = x[0] if head["pool"] == "first" else x.mean(axis=0)
        return run_steps(T, head["steps"], pooled)
    if head["type"] == "token":
        return run_steps(T, head["steps"], x)
    if head["type"] == "embed":
        pooled = {"mean": x.mean(axis=0), "cls": x[0], "max": x.max(axis=0)}[head["pool"]]
        final = run_steps(T, head["steps"], pooled)
        if head["normalize"]:
            final = final / np.linalg.norm(final)
        return pooled, final
    raise ValueError(f"unknown head type {head['type']}")


class Goldens:
    def __init__(self, model_id: str, task: str):
        self.doc = json.loads((GOLDEN_DIR / slug(model_id) / f"{task}.json").read_text())
        gdir = model_dir(model_id) / "golden"
        index = json.loads((gdir / "index.json").read_text())
        self.entries = {e["name"] + "@" + e["file"]: e for e in index["entries"]}
        self.gdir = gdir
        self.cache: dict[str, bytes] = {}

    def array(self, ref: dict) -> np.ndarray:
        e = self.entries[ref["name"] + "@" + ref["file"]]
        raw = self.cache.setdefault(e["file"], (self.gdir / e["file"]).read_bytes())
        return np.frombuffer(raw, dtype=e["dtype"], count=int(np.prod(e["shape"])),
                             offset=e["offset"]).reshape(e["shape"]).astype(np.float64)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_layers(T, spec, gold, items):
    """Largest deviation per layer state over the layer-state cases."""
    worst = np.zeros(spec["layers"] + 1)
    for item, ref in zip(items, gold.doc["hidden"]):
        cap = []
        encoder(T, spec, np.array(item["input_ids"]), np.array(item["token_type_ids"]), cap)
        want = gold.array(ref)
        for i, got in enumerate(cap):
            worst[i] = max(worst[i], float(np.abs(got - want[i]).max()))
    return worst


def check_model(model_id: str) -> dict:
    mdir = model_dir(model_id)
    manifest, T = load_tensors(mdir / "f32")
    spec, head, task = manifest["spec"], manifest["head"], manifest["task"]
    gold = Goldens(model_id, task)
    doc = gold.doc
    report = {"model": model_id, "task": task, "family": spec["family"]}
    tol_ok = True

    def ids(item):
        return np.array(item["input_ids"]), np.array(item["token_type_ids"])

    if task in ("sequence-classification", "nli"):
        dev, agree = 0.0, 0
        logits_all = []
        for item in doc["items"]:
            x = encoder(T, spec, *ids(item))
            logits = head_forward(T, head, x)
            logits_all.append(logits)
            dev = max(dev, float(np.abs(logits - np.array(item["logits"])).max()))
            agree += int(logits.argmax() == item["argmax"])
        report.update(max_abs_logit_diff=dev, argmax_agreement=agree / len(doc["items"]))
        tol_ok = dev <= LOGIT_TOL and agree == len(doc["items"])
        if task == "nli":
            ent, ok = doc["entail_index"], 0
            for t in doc["texts"]:
                e = np.array([logits_all[t["text_index"] * 5 + j][ent] for j in range(5)])
                ok += int(e.argmax() == t["zero_shot_choice"])
            report["zero_shot_agreement"] = ok / len(doc["texts"])
            tol_ok = tol_ok and ok == len(doc["texts"])
        check_items = doc["items"]
    elif task == "reranking":
        dev, scores = 0.0, {}
        for item in doc["items"]:
            logit = float(head_forward(T, head, encoder(T, spec, *ids(item)))[0])
            dev = max(dev, abs(logit - item["logit"]))
            scores.setdefault(item["query"], []).append(logit)
        ok = sum(int(np.argmax(scores[q["query"]]) == q["best"]) for q in doc["queries"])
        report.update(max_abs_logit_diff=dev, best_passage_agreement=ok / len(doc["queries"]))
        tol_ok = dev <= LOGIT_TOL and ok == len(doc["queries"])
        check_items = doc["items"]
    elif task == "token-classification":
        dev, agree, total = 0.0, 0, 0
        for item in doc["items"]:
            logits = head_forward(T, head, encoder(T, spec, *ids(item)))
            want = gold.array(item["logits"])
            dev = max(dev, float(np.abs(logits - want).max()))
            agree += int((logits.argmax(axis=-1) == np.array(item["argmax"])).sum())
            total += len(item["argmax"])
        report.update(max_abs_logit_diff=dev, argmax_per_token_agreement=agree / total, tokens=total)
        tol_ok = dev <= LOGIT_TOL and agree == total
        check_items = doc["items"]
    elif task == "embeddings":
        pooled_ref, final_ref = gold.array(doc["pooled"]), gold.array(doc["final"])
        cos_final, cos_pooled, dev = 1.0, 1.0, 0.0
        for i, item in enumerate(doc["items"]):
            pooled, final = head_forward(T, head, encoder(T, spec, *ids(item)))
            cos_final = min(cos_final, cosine(final, final_ref[i]))
            cos_pooled = min(cos_pooled, cosine(pooled, pooled_ref[i]))
            dev = max(dev, float(np.abs(final - final_ref[i]).max()))
        report.update(min_cosine_final=cos_final, min_cosine_pooled=cos_pooled, max_abs_diff_final=dev)
        tol_ok = cos_final >= COSINE_MIN and cos_pooled >= COSINE_MIN
        check_items = doc["items"]
    else:
        raise ValueError(task)

    layers = check_layers(T, spec, gold, check_items)
    report["layer_state_max_diff"] = [float(v) for v in layers]
    report["layer_state_worst"] = float(layers.max())
    report["g2_pass"] = bool(tol_ok)

    # G3: f16 manifest against numpy's cast of the f32 tensors.
    m32, e32 = load_raw(mdir / "f32", "f32")
    m16, e16 = load_raw(mdir / "f16", "f16")
    bad = []
    if [t["name"] for t, _ in e32] != [t["name"] for t, _ in e16]:
        bad.append("tensor lists differ")
    for (t32, a32), (t16, a16) in zip(e32, e16):
        if t32["shape"] != t16["shape"] or not np.array_equal(a32.astype(np.float16).view(np.uint16),
                                                              a16.view(np.uint16)):
            bad.append(t32["name"])
    report.update(f16_tensors=len(e16), f16_differing=len(bad), f16_differing_names=bad[:5],
                  g3_pass=not bad)
    return report


def result_path(args) -> Path:
    if args.tag:
        return ROOT / f"bench/results/k28-forward-{args.tag}.json"
    return ROOT / "bench/results/k28-forward-k28.6.json" if args.k286 else RESULT_K285 if args.k285 else RESULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-k28.5", dest="k285", action="store_true")
    parser.add_argument("--pilot-k28.6", dest="k286", action="store_true")
    parser.add_argument("--tag", help="result file bench/results/k28-forward-NAME.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        raise SystemExit("use --check")
    rows = [check_model(m) for m in resolve_ids(args.models, args.pilot, k285=args.k285, k286=args.k286)]
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k != "layer_state_max_diff"}))
    result_path(args).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if not all(r["g2_pass"] and r["g3_pass"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
