"""Independent f16 simulation of the K28 engine (K28.S3, decision 3 of docs/K28_S3_WORKORDER.md).

Reads the f16 manifest (models/k28/<slug>/f16) and computes the forward pass of the engine in numpy
float32: every buffer the engine keeps in f16 is rounded to f16 after the operation that writes it
(matmul and head outputs, LayerNorm outputs, attention context, embedding, residual adds, RoPE, GeGLU),
every accumulation (matmul, LayerNorm statistics, softmax, pooling) stays in float32 like the kernels.
Scores and softmax stay float32 (workgroup memory). The sentinel for masked keys plays no role: the
goldens are unpadded sequences, the engine's padding keys get weight exactly 0.

The largest logit deviation against the golden is delta_sim. A real f16 run is judged against it:
delta_run <= 3 * delta_sim, else the gate says "f16 deviates more than the simulation". For embeddings the
smallest cosine of the simulation tells whether f16 storage alone explains a cosine below 0.999.

Output: bench/results/k28-f16-sim/<slug>.json (one file per model), one JSON line per model on stdout.
When a result file of an f16 parity run exists, delta_run and delta_run / delta_sim are added.

Usage: .venv-k28/bin/python convert/k28_f16_sim.py <model-id>... | --all-available
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k28_forward as F  # noqa: E402
from k28_common import K28_DIR, ROOT, model_dir, slug  # noqa: E402

OUT_DIR = ROOT / "bench/results/k28-f16-sim"
RESULTS = ROOT / "bench/results"
f32 = np.float32


def r(x):
    """Round to f16 and back: a value stored in an f16 buffer of the engine."""
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=f32).astype(np.float16).astype(f32)


def load_f16(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """name -> float32 array of the f16 tensors; embedding row chunks joined by rowStart."""
    manifest, entries = F.load_raw(path, "f16")
    tensors: dict[str, np.ndarray] = {}
    for t, arr in entries:
        arr = arr.astype(f32)
        tensors[t["name"]] = np.concatenate([tensors[t["name"]], arr]) if t["name"] in tensors else arr
    return manifest, tensors


def opt(T, name):
    return T[name] if name in T else f32(0.0)


def layer_norm(x, w, b, eps):
    """The layernorm kernel: float32 sums, variance as E[x^2] - mean^2, no rounding here."""
    n = x.shape[-1]
    mean = x.sum(axis=-1, keepdims=True, dtype=f32) / f32(n)
    sq = (x * x).sum(axis=-1, keepdims=True, dtype=f32)
    var = sq / f32(n) - mean * mean
    inv = f32(1.0) / np.sqrt(var + f32(eps))
    return (x - mean) * inv * w + b


def gelu(v):
    """Abramowitz-Stegun 7.1.26 erf form, as in matmul.wgsl ACT 2 and geglu.wgsl."""
    u = v * f32(0.7071067811865476)
    t = f32(1.0) / (f32(1.0) + f32(0.3275911) * np.abs(u))
    p = (((((f32(1.061405429) * t - f32(1.453152027)) * t) + f32(1.421413741)) * t
          - f32(0.284496736)) * t + f32(0.254829592)) * t
    e = f32(1.0) - p * np.exp(-u * u)
    return f32(0.5) * v * (f32(1.0) + np.where(v >= 0, e, -e))


def act(name, v):
    if name == "none":
        return v
    if name == "relu":
        return np.maximum(v, f32(0.0))
    if name == "gelu":
        return gelu(v)
    if name == "tanh":
        return np.tanh(v)
    if name == "silu":
        return v / (f32(1.0) + np.exp(-v))
    raise ValueError(f"unknown activation {name}")


def dense(x, w, b, a="none"):
    """matmul kernel: float32 accumulation, bias and activation in float32, one rounding on write."""
    return r(act(a, x @ w.T + b))


def softmax_ctx(scores, v):
    """Kernel softmax: float32, normalization folded into the epilogue (ctx = (e @ v) / total)."""
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    total = e.sum(axis=-1, keepdims=True, dtype=f32)
    return r((e @ v) * (f32(1.0) / total))


def rel_table(n, span, max_pos):
    """relPosTable of src/plan/build.ts (double precision), clipped to [0, 2 * span - 1]."""
    q = np.arange(n)
    rel = q[:, None] - q[None, :]
    mid = span >> 1
    ab = np.abs(rel)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pos = np.ceil(np.log(np.maximum(ab, 1) / mid) / np.log((max_pos - 1) / mid) * (mid - 1)) + mid
    out = np.where(ab > mid, log_pos * np.sign(rel), rel)
    return np.clip(out + span, 0, 2 * span - 1).astype(np.int64)


def split_heads(t, n, heads, d):
    return t.reshape(n, heads, d).transpose(1, 0, 2)


def encoder_post(T, spec, ids, tt):
    emb, blk = spec["embed"], spec["block"]
    relative = spec["attention"]["kind"] == "deberta-relative"
    n, heads, d, hidden = len(ids), spec["heads"], spec["headDim"], spec["hidden"]
    v = T["embeddings.word.weight"][ids]
    if emb["positions"] == "absolute":
        v = v + T["embeddings.position.weight"][np.arange(n) + emb["positionOffset"]]
        if emb["typeVocab"] > 0:
            v = v + T["embeddings.type.weight"][tt]
    x = r(layer_norm(v, T["embeddings.LayerNorm.weight"], T["embeddings.LayerNorm.bias"], emb["norm"]["eps"]))
    if emb["project"]:
        x = dense(x, T["embeddings.project.weight"], T["embeddings.project.bias"])
    eps = blk["norm"]["eps"]
    emb_out = x
    if relative:
        rel = spec["attention"]["rel"]
        span = rel["buckets"]
        idx = rel_table(n, span, rel["maxPositions"])
        scale = f32(np.sqrt((1 + len(rel["types"])) * (hidden / heads)))
    else:
        scale = f32((hidden / heads) ** -0.5)
    for l in range(spec["layers"]):
        p = f"layers.{l}."
        qkv = dense(x, T[p + "qkv.weight"], T[p + "qkv.bias"])
        q, k, vv = (split_heads(t, n, heads, d) for t in np.split(qkv, 3, axis=-1))
        if relative:
            pos_k = T[p + "pos_key"].reshape(2 * span, heads, d).transpose(1, 0, 2)
            pos_q = T[p + "pos_query"].reshape(2 * span, heads, d).transpose(1, 0, 2)
            c2p = np.take_along_axis(q @ pos_k.transpose(0, 2, 1), np.broadcast_to(idx, (heads, n, n)), axis=-1)
            # engine: q_i . k_j + q_i . posKey[idx[i,j]] + k_j . posQuery[idx[i,j]]
            p2c_ij = np.take_along_axis(
                np.einsum("hjd,hpd->hjp", k, pos_q), np.broadcast_to(idx.T, (heads, n, n)), axis=-1
            ).transpose(0, 2, 1)
            scores = (q @ k.transpose(0, 2, 1) + c2p + p2c_ij) / scale
        else:
            scores = q @ k.transpose(0, 2, 1) * scale
        ctx = softmax_ctx(scores, vv).transpose(1, 0, 2).reshape(n, -1)
        attn = dense(ctx, T[p + "attn_out.weight"], T[p + "attn_out.bias"])
        tmp = r(layer_norm(x + attn, T[p + "attn_ln.weight"], T[p + "attn_ln.bias"], eps))
        mid = dense(tmp, T[p + "ffn_in.weight"], T[p + "ffn_in.bias"], spec["ffn"]["act"])
        ffn = dense(mid, T[p + "ffn_out.weight"], T[p + "ffn_out.bias"])
        x = r(layer_norm(tmp + ffn, T[p + "ffn_ln.weight"], T[p + "ffn_ln.bias"], eps))
        if l == 0 and "conv" in spec:
            # convIn: im2col is a copy, the matmul rounds once; conv: the add kernel rounds the sum to f16,
            # then LN(sum) * mask (all ones without padding) is written to x
            conv = spec["conv"]
            out = dense(F.conv_cols(emb_out, conv["kernel"]), T["conv.weight"], T["conv.bias"], conv["act"])
            x = r(layer_norm(r(out + x), T["conv.ln.weight"], T["conv.ln.bias"], eps))
    return x


def rope_tables(n, theta, d):
    """Float32 table of the engine: computed in double like ropeTable (build.ts), stored as float32."""
    half = d // 2
    freqs = np.arange(n)[:, None] * theta ** (-(2.0 * np.arange(half)) / d)[None, :]
    return np.cos(freqs).astype(f32), np.sin(freqs).astype(f32)


def encoder_pre(T, spec, ids, tt):
    blk, att = spec["block"], spec["attention"]
    n, heads, d, eps = len(ids), spec["heads"], spec["headDim"], blk["norm"]["eps"]
    x = r(layer_norm(T["embeddings.word.weight"][ids], T["embeddings.norm.weight"],
                     opt(T, "embeddings.norm.bias"), eps))
    rope = att["rope"]
    tables = {"g": rope_tables(n, rope["thetaGlobal"], d), "l": rope_tables(n, rope["thetaLocal"], d)}
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    scale = f32((spec["hidden"] / heads) ** -0.5)
    half = d // 2
    for l in range(spec["layers"]):
        p = f"layers.{l}."
        is_global = l % att["window"]["globalEvery"] == 0
        h = x if (blk["firstNormIdentity"] and l == 0) else r(layer_norm(
            x, T[p + "attn_norm.weight"], opt(T, p + "attn_norm.bias"), eps))
        qkv = dense(h, T[p + "wqkv.weight"], opt(T, p + "wqkv.bias")).reshape(n, 3, heads, d)
        q, k, vv = (qkv[:, i].transpose(1, 0, 2) for i in range(3))
        cos, sin = tables["g" if is_global else "l"]
        cos, sin = cos[None], sin[None]  # [1, n, d/2]

        def rot(t):
            a, b = t[..., :half], t[..., half:]
            return np.concatenate([r(a * cos - b * sin), r(b * cos + a * sin)], axis=-1)

        q, k = rot(q), rot(k)
        scores = q @ k.transpose(0, 2, 1) * scale
        if not is_global:
            scores = np.where(dist <= att["window"]["half"], scores, f32(-1e30))
        ctx = softmax_ctx(scores, vv).transpose(1, 0, 2).reshape(n, -1)
        x = r(x + dense(ctx, T[p + "attn_out.weight"], opt(T, p + "attn_out.bias")))
        h = r(layer_norm(x, T[p + "mlp_norm.weight"], opt(T, p + "mlp_norm.bias"), eps))
        mid = dense(h, T[p + "mlp_in.weight"], opt(T, p + "mlp_in.bias"))
        inp, gate = np.split(mid, 2, axis=-1)
        g = r(gelu(inp) * gate)
        x = r(x + dense(g, T[p + "mlp_out.weight"], opt(T, p + "mlp_out.bias")))
    return r(layer_norm(x, T["final_norm.weight"], opt(T, "final_norm.bias"), eps))


def encoder(T, spec, ids, tt):
    return encoder_pre(T, spec, ids, tt) if spec["block"]["order"] == "pre" else encoder_post(T, spec, ids, tt)


def run_steps(T, steps, x):
    for s in steps:
        if s["op"] == "dense":
            x = dense(x, T[s["name"] + ".weight"], opt(T, s["name"] + ".bias"), s["act"])
        elif s["op"] == "norm":
            x = r(layer_norm(x, T[s["name"] + ".weight"], opt(T, s["name"] + ".bias"), s["eps"]))
        else:
            raise ValueError(f"unsupported head step {s['op']}")
    return x


def head_forward(T, head, x):
    """Logits [C], [n, C] or (pooled, final); the final output is an f16 buffer in the engine."""
    if head["type"] == "token":
        return run_steps(T, head["steps"], x)
    pool = head["pool"]
    if pool in ("first", "cls"):
        pooled = x[0]
    elif pool == "mean":
        pooled = r(x.sum(axis=0, dtype=f32) / f32(len(x)))
    else:
        pooled = x.max(axis=0)
    final = run_steps(T, head["steps"], pooled[None])[0]
    if head["type"] == "classify":
        return final
    if head["normalize"]:
        final = final / np.linalg.norm(final.astype(np.float64))
    return pooled, final


def simulate(model_id: str) -> dict:
    mdir = model_dir(model_id)
    manifest, T = load_f16(mdir / "f16")
    spec, head, task = manifest["spec"], manifest["head"], manifest["task"]
    gold = F.Goldens(model_id, task)
    doc = gold.doc
    out = {"model": slug(model_id), "task": task, "family": spec["family"], "items": len(doc["items"])}

    def ids(item):
        return np.array(item["input_ids"]), np.array(item["token_type_ids"])

    if task in ("sequence-classification", "nli"):
        dev = max(float(np.abs(head_forward(T, head, encoder(T, spec, *ids(i))).astype(np.float64)
                              - np.array(i["logits"])).max()) for i in doc["items"])
        out["delta_sim"] = dev
    elif task == "reranking":
        dev = max(abs(float(head_forward(T, head, encoder(T, spec, *ids(i)))[0]) - i["logit"])
                  for i in doc["items"])
        out["delta_sim"] = dev
    elif task == "token-classification":
        dev = 0.0
        for item in doc["items"]:
            logits = head_forward(T, head, encoder(T, spec, *ids(item))).astype(np.float64)
            dev = max(dev, float(np.abs(logits - gold.array(item["logits"])).max()))
        out["delta_sim"] = dev
    elif task == "embeddings":
        pooled_ref, final_ref = gold.array(doc["pooled"]), gold.array(doc["final"])
        cos_f, cos_p, dev = 1.0, 1.0, 0.0
        for i, item in enumerate(doc["items"]):
            pooled, final = head_forward(T, head, encoder(T, spec, *ids(item)))
            final = final.astype(np.float64)
            cos_f = min(cos_f, F.cosine(final, final_ref[i]))
            cos_p = min(cos_p, F.cosine(pooled.astype(np.float64), pooled_ref[i]))
            dev = max(dev, float(np.abs(final - final_ref[i]).max()))
        out["delta_sim"] = dev
        out["min_cosine_final"] = cos_f
        if not head["steps"]:
            out["min_cosine_pooled"] = cos_p
    else:
        raise ValueError(task)
    out.update(run_comparison(out))
    return out


def latest_f16_parity(sl: str):
    best = None
    for p in RESULTS.glob(f"k28-parity-{sl}-f16-*.json"):
        if p.name.endswith("-text.json") or p.name.endswith("-layers.json"):
            continue
        doc = json.loads(p.read_text())
        t = int(str(doc.get("run_id", "0")).split("-")[-1])
        if best is None or t > best[0]:
            best = (t, p.name, doc)
    return best


def run_comparison(sim: dict) -> dict:
    """delta_run of the newest f16 parity file and its ratio to delta_sim (the 0.3 to 3 window)."""
    best = latest_f16_parity(sim["model"])
    if not best:
        return {}
    m = best[2]["metrics"]
    res = {"run_file": best[1]}
    if sim["task"] == "embeddings":
        res["delta_run"] = m.get("maxAbsDiffFinal")
        res["min_cosine_final_run"] = m.get("minCosineFinal")
        if "min_cosine_final" in sim and res["min_cosine_final_run"] is not None:
            res["cosine_loss_ratio"] = (1 - res["min_cosine_final_run"]) / max(1 - sim["min_cosine_final"], 1e-12)
    else:
        res["delta_run"] = m.get("maxAbsLogitDiff")
    if res.get("delta_run") is not None:
        res["ratio"] = res["delta_run"] / sim["delta_sim"]
    return res


def available() -> list[str]:
    ids = []
    for e in json.loads((ROOT / "data/k28/models.json").read_text())["models"]:
        d = K28_DIR / slug(e["id"])
        if (d / "f16/manifest.json").exists() and (d / "golden/index.json").exists() \
                and (ROOT / f"tests/golden/k28/{slug(e['id'])}").exists():
            ids.append(e["id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all-available", action="store_true")
    args = ap.parse_args()
    ids = available() if args.all_available else args.models
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        res = simulate(mid)
        (OUT_DIR / f"{slug(mid)}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
