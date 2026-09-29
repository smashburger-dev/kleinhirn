"""Julia 1 weight export into the kleinhirn manifest format (K8).

Reads models/julia-1/repo/model.safetensors (pinned HF revision, see
models/julia-1/REVISIONS.txt) and writes models/julia-1/<precision>/
manifest.json plus weights-<n>.bin shards under 95 MB, tensor offsets aligned
to 256 bytes, sha256 per shard and of the source safetensors.

Layout (safetensors -> manifest):
- encoder.embeddings.tok_embeddings.weight [256000,384] -> embeddings.word.weight,
  keepOnCpu row chunks (the table dominates the checkpoint: ~98M params)
- encoder.embeddings.norm.weight -> embeddings.norm.weight (LayerNorm, no bias)
- encoder.layers.N.attn_norm.weight -> layers.N.attn_norm.weight (absent at N=0:
  ModernBERT layer 0 uses Identity)
- encoder.layers.N.attn.Wqkv.weight [1152,384] -> layers.N.wqkv.weight (no bias)
- encoder.layers.N.attn.Wo.weight -> layers.N.attn_out.weight (no bias)
- encoder.layers.N.mlp_norm.weight -> layers.N.mlp_norm.weight (no bias)
- encoder.layers.N.mlp.Wi.weight [2304,384] -> layers.N.mlp_in.weight (no bias;
  GeGLU input|gate halves)
- encoder.layers.N.mlp.Wo.weight [384,1152] -> layers.N.mlp_out.weight (no bias)
- encoder.final_norm.weight -> final_norm.weight
- type_emb.weight [3,384] -> type_emb.weight
- head.layers.N.* -> head.N.{in_proj,out_proj,norm1,norm2,linear1,linear2}
  (.weight and .bias; nn.TransformerEncoderLayer, pre-norm, RELU FFN)
- scorer.0 -> scorer.norm (LayerNorm with bias), scorer.1 -> scorer.fc1,
  scorer.3 -> scorer.fc2
- act_head.* and temperature are exported for completeness (not needed for
  decision logits)

Usage: .venv-julia/bin/python convert/export_julia.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "models/julia-1/repo"
SHARD_LIMIT = 95_000_000
ALIGN = 256
EMBED_CHUNK_ROWS = 60_000


def collect() -> tuple[list[tuple[str, np.ndarray, dict]], dict]:
    sd = load_file(str(REPO / "model.safetensors"))
    cfg = json.loads((REPO / "encoder/config.json").read_text())
    tensors: list[tuple[str, np.ndarray, dict]] = []

    def put(name, arr, **meta):
        tensors.append((name, np.ascontiguousarray(arr.astype(np.float32)), meta))

    emb = sd["encoder.embeddings.tok_embeddings.weight"]
    for start in range(0, emb.shape[0], EMBED_CHUNK_ROWS):
        put("embeddings.word.weight", emb[start:start + EMBED_CHUNK_ROWS],
            keepOnCpu=True, rowStart=start,
            rowEnd=min(start + EMBED_CHUNK_ROWS, emb.shape[0]))
    put("embeddings.norm.weight", sd["encoder.embeddings.norm.weight"])
    n_layers = cfg["num_hidden_layers"]
    for i in range(n_layers):
        p = f"encoder.layers.{i}"
        if f"{p}.attn_norm.weight" in sd:
            put(f"layers.{i}.attn_norm.weight", sd[f"{p}.attn_norm.weight"])
        put(f"layers.{i}.wqkv.weight", sd[f"{p}.attn.Wqkv.weight"])
        put(f"layers.{i}.attn_out.weight", sd[f"{p}.attn.Wo.weight"])
        put(f"layers.{i}.mlp_norm.weight", sd[f"{p}.mlp_norm.weight"])
        put(f"layers.{i}.mlp_in.weight", sd[f"{p}.mlp.Wi.weight"])
        put(f"layers.{i}.mlp_out.weight", sd[f"{p}.mlp.Wo.weight"])
    put("final_norm.weight", sd["encoder.final_norm.weight"])
    put("type_emb.weight", sd["type_emb.weight"])
    for i in range(2):
        p = f"head.layers.{i}"
        put(f"head.{i}.in_proj.weight", sd[f"{p}.self_attn.in_proj_weight"])
        put(f"head.{i}.in_proj.bias", sd[f"{p}.self_attn.in_proj_bias"])
        for src, dst in (("self_attn.out_proj", "out_proj"),
                         ("norm1", "norm1"), ("norm2", "norm2"),
                         ("linear1", "linear1"), ("linear2", "linear2")):
            put(f"head.{i}.{dst}.weight", sd[f"{p}.{src}.weight"])
            put(f"head.{i}.{dst}.bias", sd[f"{p}.{src}.bias"])
    put("scorer.norm.weight", sd["scorer.0.weight"])
    put("scorer.norm.bias", sd["scorer.0.bias"])
    put("scorer.fc1.weight", sd["scorer.1.weight"])
    put("scorer.fc1.bias", sd["scorer.1.bias"])
    put("scorer.fc2.weight", sd["scorer.3.weight"])
    put("scorer.fc2.bias", sd["scorer.3.bias"])
    put("act_head.fc1.weight", sd["act_head.0.weight"])
    put("act_head.fc1.bias", sd["act_head.0.bias"])
    put("act_head.fc2.weight", sd["act_head.2.weight"])
    put("act_head.fc2.bias", sd["act_head.2.bias"])
    put("temperature", sd["temperature"])
    encoder = {
        "arch": "modernbert-julia",
        "layers": n_layers,
        "hiddenSize": cfg["hidden_size"],
        "heads": cfg["num_attention_heads"],
        "headDim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "intermediate": cfg["intermediate_size"],
        "normEps": cfg["norm_eps"],
        "ropeTheta": cfg["rope_parameters"]["full_attention"]["rope_theta"],
        "localAttention": cfg["local_attention"] // 2,
        "globalEvery": cfg["global_attn_every_n_layers"],
        "maxPos": cfg["max_position_embeddings"],
        "vocab": cfg["vocab_size"],
        "headLayers": 2,
        "headFfn": 1536,
        "options": 20,
        "qtypes": 3,
        "clsToken": cfg["cls_token_id"],
        "sepToken": cfg["sep_token_id"],
        "maskToken": cfg["mask_token_id"],
        "padToken": cfg["pad_token_id"],
    }
    return tensors, encoder


def export(precision: str, dtype):
    tensors, encoder = collect()
    out_dir = ROOT / "models/julia-1" / precision
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    entries = []
    shards = []
    shard_index = 0
    shard_start = 0

    def flush_shard():
        nonlocal blob, shard_start, shard_index
        if shard_start == len(blob):
            return
        payload = bytes(blob[shard_start:])
        file_name = f"weights-{shard_index}.bin"
        (out_dir / file_name).write_bytes(payload)
        shards.append({"file": file_name, "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()})
        shard_index += 1
        shard_start = len(blob)

    def pad():
        while (len(blob) - shard_start) % ALIGN:
            blob.append(0)

    for name, value, meta in tensors:
        cast = value.astype(dtype)
        raw = cast.tobytes()
        if len(blob) - shard_start + len(raw) > SHARD_LIMIT:
            flush_shard()
        pad()
        entry = {"name": name, "dtype": precision, "shape": list(value.shape),
                 "shard": shard_index, "offset": len(blob) - shard_start,
                 "byteLength": len(raw)}
        entry.update(meta)
        entries.append(entry)
        blob += raw
    flush_shard()
    manifest = {
        "format": "kleinhirn-weights-1",
        "source": {
            "repo": "SupersonicLabs/Julia-1",
            "revision": "a85b127321d580d65176c89ced8273f305745d85",
            "checkpointSha256": hashlib.sha256(
                (REPO / "model.safetensors").read_bytes()).hexdigest(),
        },
        "encoder": encoder,
        "head": {"type": "julia-decision", "temperature": 1.0},
        "tokenizer": "tokenizer.json",
        "tensors": entries,
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(s["bytes"] for s in shards)
    print(f"{precision}: {len(entries)} tensors, {len(shards)} shards, "
          f"{total / 1e6:.1f} MB")


def main():
    for precision, dtype in (("f32", np.float32), ("f16", np.float16)):
        export(precision, dtype)
    import shutil
    for precision in ("f32", "f16"):
        dst = ROOT / "models/julia-1" / precision / "tokenizer.json"
        if not dst.exists():
            shutil.copy(REPO / "tokenizer/tokenizer.json", dst)


if __name__ == "__main__":
    main()
