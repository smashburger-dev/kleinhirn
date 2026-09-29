"""Executable specification of the kleinhirn forward pass, numpy only.

Loads ONLY manifest tensors (models/<model>/f32/) and applies the formulas
from docs/ARCHITECTURE.md: embedding LayerNorm + mask, relative attention
with c2p/p2c over make_log_bucket_position, scale sqrt(64 * 3), exact GELU
(erf), LayerNorm eps 1e-7, classification head, temperature, per-group
softmax. This file is the template for the WGSL engine; keep it readable.

Gate (--check): against the f32 goldens, max logit deviation <= 1e-4 and
100 % argmax on all 1000 corpus texts, layer states in L128 and L256 on valid
positions <= 1e-4, two-task cases likewise.

Usage:
  .venv/bin/python convert/manifest_forward.py --check [model ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-7
F32_MIN = np.finfo(np.float32).min
MASK_FILL = -1e4


def load_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Read manifest.json and return name -> f32 ndarray (row chunks joined)."""
    manifest = json.loads((model_dir / "manifest.json").read_text())
    shard_data = [(model_dir / s["file"]).read_bytes() for s in manifest["shards"]]
    tensors: dict[str, np.ndarray] = {}
    for t in manifest["tensors"]:
        if t["dtype"] != "f32":
            raise ValueError(f"gate uses the f32 manifest; got {t['dtype']} for {t['name']}")
        arr = np.frombuffer(
            shard_data[t["shard"]], dtype=np.float32,
            count=int(np.prod(t["shape"])), offset=t["offset"],
        ).reshape(t["shape"])
        if t["name"] in tensors:
            tensors[t["name"]] = np.concatenate([tensors[t["name"]], arr], axis=0)
        else:
            tensors[t["name"]] = arr.copy()
    return tensors


def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
    """DeBERTa-v2 position bucketing (transformers modeling_deberta_v2)."""
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where(
        (relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    log_pos = (
        np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1))
        + mid
    )
    return np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int64)


def relative_position_table(length: int, bucket_size: int, max_position: int):
    """idx[i, j] = clamp(bucket(i - j) + span, 0, 2 * span - 1); span = bucket_size."""
    d = np.arange(length)[:, None] - np.arange(length)[None, :]
    r = make_log_bucket_position(d, bucket_size, max_position)
    return np.clip(r + bucket_size, 0, bucket_size * 2 - 1)


def layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def linear(x, weight, bias):
    return x @ weight.T + bias


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(x, axis=-1):
    exps = np.exp(x - x.max(axis=axis, keepdims=True))
    return exps / exps.sum(axis=axis, keepdims=True)


def forward(tensors, encoder, input_ids, attention_mask, marker_indices,
            marker_mask, marker_groups=None, capture=None):
    """One classification pass. capture: optional dict for per-layer states."""
    length = input_ids.shape[0]
    heads = encoder["heads"]
    head_dim = encoder["hiddenSize"] // heads
    span = encoder["positionBuckets"]
    scale = np.sqrt(head_dim * 3.0)
    idx = relative_position_table(length, span, encoder["maxRelativePositions"])

    emb = tensors["embeddings.word.weight"][input_ids]
    x = layer_norm(emb, tensors["embeddings.LayerNorm.weight"],
                   tensors["embeddings.LayerNorm.bias"])
    if capture is not None:
        capture["embedding_ln"] = x.copy()
    x = x * attention_mask[:, None]
    if capture is not None:
        capture["embedding_masked"] = x.copy()

    pair_mask = attention_mask[:, None] * attention_mask[None, :]  # [L, L]

    for layer in range(encoder["layers"]):
        qkv = linear(x, tensors[f"layers.{layer}.qkv.weight"],
                     tensors[f"layers.{layer}.qkv.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)

        def heads_out(t):  # [rows, H*D] -> [H, rows, D]
            return t.reshape(t.shape[0], heads, head_dim).transpose(1, 0, 2)

        q, k, v = heads_out(q), heads_out(k), heads_out(v)
        pos_key = heads_out(tensors[f"layers.{layer}.pos_key"])   # [H, 512, D]
        pos_query = heads_out(tensors[f"layers.{layer}.pos_query"])

        scores = q @ k.transpose(0, 2, 1)
        # c2p[i, j] = q_i . pos_key[idx[i, j]]; p2c[i, j] = k_j . pos_query[idx[i, j]]
        c2p_full = q @ pos_key.transpose(0, 2, 1)               # [H, L, 512]
        idx_b = np.broadcast_to(idx[None], (heads, length, length))
        c2p = np.take_along_axis(c2p_full, idx_b, axis=2)
        p2c_full = (k @ pos_query.transpose(0, 2, 1)).transpose(0, 2, 1)  # [H, 512, L]
        p2c = np.take_along_axis(p2c_full, idx_b, axis=1)
        scores = (scores + c2p + p2c) / scale

        scores = np.where(pair_mask[None].astype(bool), scores, F32_MIN)
        probs = softmax(scores)
        context = probs @ v                                     # [H, L, D]
        context = context.transpose(1, 0, 2).reshape(length, -1)
        attn = linear(context, tensors[f"layers.{layer}.attn_out.weight"],
                      tensors[f"layers.{layer}.attn_out.bias"])
        x = layer_norm(attn + x, tensors[f"layers.{layer}.attn_ln.weight"],
                       tensors[f"layers.{layer}.attn_ln.bias"])

        mid = gelu(linear(x, tensors[f"layers.{layer}.ffn_in.weight"],
                          tensors[f"layers.{layer}.ffn_in.bias"]))
        out = linear(mid, tensors[f"layers.{layer}.ffn_out.weight"],
                     tensors[f"layers.{layer}.ffn_out.bias"])
        x = layer_norm(out + x, tensors[f"layers.{layer}.ffn_ln.weight"],
                       tensors[f"layers.{layer}.ffn_ln.bias"])
        if capture is not None:
            capture[f"layer_{layer}"] = x.copy()

    states = x[marker_indices]
    hidden = np.maximum(linear(states, tensors["head.fc1.weight"],
                             tensors["head.fc1.bias"]), 0.0)
    logits = (hidden @ tensors["head.fc2.weight"].T
              + tensors["head.fc2.bias"]).reshape(-1)
    logits = logits / encoder["temperature"]
    logits = np.where(marker_mask > 0.5, logits, MASK_FILL)
    return logits


def group_probabilities(logits, groups, mask, counts):
    """Per-group softmax over valid markers (native decode order)."""
    out = []
    for g, n in enumerate(counts):
        vals = logits[(groups == g) & (mask > 0.5)]
        assert len(vals) == n
        exps = np.exp(vals - vals.max())
        out.append({"logits": vals, "probabilities": exps / exps.sum(),
                    "label_index": int(vals.argmax())})
    return out


def check_model(name: str):
    model_dir = ROOT / "models" / name / "f32"
    tensors = load_tensors(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    encoder = dict(manifest["encoder"])
    encoder["temperature"] = manifest["head"]["temperature"]
    golden_dir = ROOT / "tests" / "golden" / name
    report = {"model": name}

    # Corpus-1000 logits + argmax.
    corpus = json.loads((golden_dir / "texts1000_l128k16.json").read_text())
    max_logit = 0.0
    argmax_ok = 0
    for item in corpus["items"]:
        ids = np.zeros(128, dtype=np.int64)
        ids[: item["seq_len"]] = item["input_ids"]
        mask = np.zeros(128, dtype=np.int64)
        mask[: item["seq_len"]] = 1
        markers = np.array(item["marker_indices"] + [0] * (16 - len(item["marker_indices"])))
        mmask = np.array(item["marker_mask"] + [0.0] * (16 - len(item["marker_mask"])))
        groups = np.array(item["marker_groups"] + [0] * (16 - len(item["marker_groups"])))
        logits = forward(tensors, encoder, ids, mask, markers, mmask)
        per = group_probabilities(logits, groups, mmask, [len(item["logits"])])[0]
        max_logit = max(max_logit, float(np.abs(per["logits"] - item["logits"]).max()))
        argmax_ok += int(per["label_index"] == item["label_index"])
    report["corpus"] = {"n": corpus["count"], "max_abs_logit_diff": max_logit,
                       "argmax_agreement": argmax_ok / corpus["count"]}

    # Layer states L128 + L256 on valid positions.
    index = json.loads((ROOT / "models" / name / "golden" / "layers.index.json").read_text())
    blob = (ROOT / "models" / name / "golden" / "layers.bin").read_bytes()
    worst = 0.0
    checked = 0
    for case in index["cases"]:
        for bucket, entry in case["buckets"].items():
            if entry is None:
                continue
            ids = np.array(entry["input_ids"], dtype=np.int64)
            mask = np.array(entry["attention_mask"], dtype=np.int64)
            markers = np.array(entry["marker_indices"] + [0] * (16 - len(entry["marker_indices"])))
            mmask = np.array(entry["marker_mask"] + [0.0] * (16 - len(entry["marker_mask"])))
            capture = {}
            forward(tensors, encoder, ids, mask, markers, mmask, capture=capture)
            seq = entry["seq_len"]
            for tname, spec in entry["tensors"].items():
                if tname == "embedding_ln":
                    continue  # pre-mask golden; engine masks itself
                ref = np.frombuffer(blob, dtype=np.float32,
                                    count=int(np.prod(spec["shape"])),
                                    offset=spec["offset"]).reshape(spec["shape"])
                got = capture[tname]
                diff = float(np.abs(got[:seq] - ref[:seq]).max())
                worst = max(worst, diff)
                checked += 1
            # embedding_ln is compared separately (pre-mask).
            ref = np.frombuffer(blob, dtype=np.float32,
                                count=int(np.prod(entry["tensors"]["embedding_ln"]["shape"])),
                                offset=entry["tensors"]["embedding_ln"]["offset"]).reshape(
                entry["tensors"]["embedding_ln"]["shape"])
            worst = max(worst, float(np.abs(capture["embedding_ln"][:seq] - ref[:seq]).max()))
    report["layers"] = {"tensors_checked": checked, "max_abs_diff_valid_positions": worst}

    # Two-task cases.
    two = json.loads((golden_dir / "two_tasks_l128k16.json").read_text())
    max_logit2 = 0.0
    argmax2 = 0
    total2 = 0
    for item in two["items"]:
        ids = np.zeros(128, dtype=np.int64)
        ids[: item["seq_len"]] = item["input_ids"]
        mask = np.zeros(128, dtype=np.int64)
        mask[: item["seq_len"]] = 1
        markers = np.array(item["marker_indices"] + [0] * (16 - len(item["marker_indices"])))
        mmask = np.array(item["marker_mask"] + [0.0] * (16 - len(item["marker_mask"])))
        groups = np.array(item["marker_groups"] + [0] * (16 - len(item["marker_groups"])))
        logits = forward(tensors, encoder, ids, mask, markers, mmask)
        counts = [len(t["labels"]) for t in item["tasks"]]
        for ref, got in zip(item["tasks"], group_probabilities(logits, groups, mmask, counts)):
            max_logit2 = max(max_logit2, float(np.abs(got["logits"] - ref["logits"]).max()))
            argmax2 += int(got["label_index"] == ref["label_index"])
            total2 += 1
    report["two_tasks"] = {"n": total2, "max_abs_logit_diff": max_logit2,
                           "argmax_agreement": argmax2 / total2}

    report["pass"] = (
        max_logit <= 1e-4 and argmax_ok == corpus["count"]
        and worst <= 1e-4 and max_logit2 <= 1e-4 and argmax2 == total2
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("models", nargs="*", default=["small-upstream", "base-upstream", "multi-upstream"])
    args = parser.parse_args()
    if args.check:
        results = [check_model(name) for name in args.models]
        for r in results:
            print(json.dumps(r))
        if not all(r["pass"] for r in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
