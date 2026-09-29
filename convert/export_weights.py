"""Weight export into the kleinhirn manifest format (docs/ARCHITECTURE.md).

Per model and precision (f32, f16): models/<model>/<precision>/manifest.json
plus weights-<n>.bin shards under 95 MB, tensor offsets aligned to 256 bytes,
sha256 per shard and of the source safetensors.

Layout per encoder layer N (HF: encoder.encoder.layer.N.*):
- qkv.weight [1152, 384] / qkv.bias [1152]: query|key|value rows fused
- attn_out / attn_ln: attention.output.dense / attention.output.LayerNorm
- ffn_in / ffn_out / ffn_ln: intermediate.dense / output.dense / output.LayerNorm
- pos_key / pos_query [512, 384]: key_proj / query_proj applied to the
  LayerNorm'ed rel_embeddings, precomputed in f32 then cast (share_att_key).
Global tensors: embeddings.word.weight (kept on CPU), embeddings.LayerNorm,
rel_embeddings (post-LayerNorm; the encoder.encoder.LayerNorm weights are
folded in and not exported), head.fc1/head.fc2 of the classification head
Linear(384->768) -> ReLU -> Linear(768->1).

Usage: .venv/bin/python convert/export_weights.py [model ...]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from gliner2 import AutoExtractor

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "small-upstream": {
        "checkpoint": ROOT / "models/small-upstream/checkpoint",
        "source": {
            "repo": "fastino/gliner2.5-small-v1",
            "revision": "7e6f537f10337497069276892a5ef435028252ce",
        },
    },
    "base-upstream": {
        "checkpoint": ROOT / "models/base-upstream/checkpoint",
        "source": {
            "repo": "fastino/gliner2.5-base-v1",
            "revision": "b0c10b23313ec3ff028821dff298dd743e010706",
        },
    },
    "multi-upstream": {
        "checkpoint": ROOT / "models/multi-upstream/checkpoint",
        "source": {
            "repo": "fastino/gliner2.5-multi-v1",
            "revision": "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        },
    },
    "multi-trimmed": {
        "checkpoint": ROOT / "models/multi-trimmed/checkpoint",
        "source": {
            "repo": "fastino/gliner2.5-multi-v1",
            "revision": "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
            "trim": "convert/trim_vocab.py corpus=data/trim/corpus.txt",
        },
    },
}
SHARD_LIMIT = 95_000_000
ALIGN = 256
EMBED_CHUNK_ROWS = 60_000


def collect_tensors(checkpoint: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Extract encoder + head tensors in manifest order, f32."""
    native = AutoExtractor.from_pretrained(str(checkpoint), map_location="cpu").eval()
    encoder = native.encoder
    cfg = encoder.config
    tensors: dict[str, np.ndarray] = {}

    def put(name, value):
        tensors[name] = np.ascontiguousarray(value.detach().to(torch.float32).numpy())

    put("embeddings.word.weight", encoder.embeddings.word_embeddings.weight)
    put("embeddings.LayerNorm.weight", encoder.embeddings.LayerNorm.weight)
    put("embeddings.LayerNorm.bias", encoder.embeddings.LayerNorm.bias)
    # rel_embeddings already LayerNorm'ed (norm_rel_ebd == "layer_norm"); the
    # encoder.encoder.LayerNorm weights are folded in here.
    rel_ln = encoder.encoder.get_rel_embedding().detach().to(torch.float32)
    put("rel_embeddings", rel_ln)
    for i, layer in enumerate(encoder.encoder.layer):
        attn = layer.attention
        w = torch.cat([
            attn.self.query_proj.weight, attn.self.key_proj.weight, attn.self.value_proj.weight,
        ], dim=0)
        b = torch.cat([
            attn.self.query_proj.bias, attn.self.key_proj.bias, attn.self.value_proj.bias,
        ], dim=0)
        put(f"layers.{i}.qkv.weight", w)
        put(f"layers.{i}.qkv.bias", b)
        put(f"layers.{i}.attn_out.weight", attn.output.dense.weight)
        put(f"layers.{i}.attn_out.bias", attn.output.dense.bias)
        put(f"layers.{i}.attn_ln.weight", attn.output.LayerNorm.weight)
        put(f"layers.{i}.attn_ln.bias", attn.output.LayerNorm.bias)
        put(f"layers.{i}.ffn_in.weight", layer.intermediate.dense.weight)
        put(f"layers.{i}.ffn_in.bias", layer.intermediate.dense.bias)
        put(f"layers.{i}.ffn_out.weight", layer.output.dense.weight)
        put(f"layers.{i}.ffn_out.bias", layer.output.dense.bias)
        put(f"layers.{i}.ffn_ln.weight", layer.output.LayerNorm.weight)
        put(f"layers.{i}.ffn_ln.bias", layer.output.LayerNorm.bias)
        # share_att_key: position projections reuse the attention projections.
        put(f"layers.{i}.pos_key", attn.self.key_proj(rel_ln).detach())
        put(f"layers.{i}.pos_query", attn.self.query_proj(rel_ln).detach())
    fc1w = native.classifier[0].weight
    fc2w = native.classifier[3].weight
    put("head.fc1.weight", fc1w)
    put("head.fc1.bias", native.classifier[0].bias)
    put("head.fc2.weight", fc2w)
    put("head.fc2.bias", native.classifier[3].bias)
    head_hid, hidden = fc1w.shape

    encoder_info = {
        "arch": "deberta-v2",
        "hiddenSize": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediateSize": cfg.intermediate_size,
        "vocabSize": cfg.vocab_size,
        "positionBuckets": cfg.position_buckets,
        "maxRelativePositions": (
            cfg.max_relative_positions if cfg.max_relative_positions > 0
            else cfg.max_position_embeddings
        ),
        "posAttType": cfg.pos_att_type,
        "shareAttKey": cfg.share_att_key,
        "layerNormEps": cfg.layer_norm_eps,
        "attSpan": cfg.position_buckets,
        "relEmbeddingRows": cfg.position_buckets * 2,
    }
    head_info = {
        "type": "gliner2-classification",
        "temperature": float(native.boundary_settings.classification_temperature),
        "hiddenSize": int(head_hid),
        "structure": f"Linear({hidden} -> {head_hid}) + ReLU + Linear({head_hid} -> 1)",
        "tensors": [
            {"name": "head.fc1.weight", "shape": [int(head_hid), int(hidden)]},
            {"name": "head.fc1.bias", "shape": [int(head_hid)]},
            {"name": "head.fc2.weight", "shape": [1, int(head_hid)]},
            {"name": "head.fc2.bias", "shape": [1]},
        ],
        "decode": "logits = head(state_at_marker) / temperature; masked softmax per task group",
    }
    return tensors, {"encoder": encoder_info, "head": head_info}


def export_model(name: str, checkpoint: Path, source: dict):
    tensors, info = collect_tensors(checkpoint)
    st_path = checkpoint / "model.safetensors"
    source["checkpointSha256"] = hashlib.sha256(st_path.read_bytes()).hexdigest()
    for precision, dtype in (("f32", np.float32), ("f16", np.float16)):
        out_dir = ROOT / "models" / name / precision
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        shards = []
        blob = bytearray()
        shard_index = 0
        shard_start = 0

        def flush_shard():
            nonlocal blob, shard_start, shard_index
            if shard_start == len(blob):
                return
            payload = bytes(blob[shard_start:])
            file_name = f"weights-{shard_index}.bin"
            (out_dir / file_name).write_bytes(payload)
            shards.append({
                "file": file_name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
            shard_index += 1
            shard_start = len(blob)

        def pad():
            while (len(blob) - shard_start) % ALIGN:
                blob.append(0)

        def write_tensor(tensor_name, value, row_start=None, row_end=None):
            nonlocal blob
            cast = value.astype(dtype)
            raw = cast.tobytes()
            pad()
            if len(blob) - shard_start + len(raw) > SHARD_LIMIT:
                flush_shard()
                pad()
            entry = {
                "name": tensor_name,
                "dtype": precision,
                "shape": list(value.shape),
                "shard": shard_index,
                "offset": len(blob) - shard_start,
                "byteLength": len(raw),
            }
            if tensor_name == "embeddings.word.weight":
                entry["keepOnCpu"] = True
                entry["rowStart"] = row_start
                entry["rowEnd"] = row_end
            entries.append(entry)
            blob += raw

        for tensor_name, value in tensors.items():
            if tensor_name == "embeddings.word.weight":
                # The embedding table exceeds the shard limit alone: split it
                # into row chunks that loaders concatenate by rowStart.
                rows = value.shape[0]
                chunk = EMBED_CHUNK_ROWS
                for row_start in range(0, rows, chunk):
                    row_end = min(row_start + chunk, rows)
                    write_tensor(tensor_name, value[row_start:row_end], row_start, row_end)
            else:
                write_tensor(tensor_name, value)
        flush_shard()
        manifest = {
            "format": "kleinhirn-weights",
            "version": 1,
            "source": source,
            "encoder": info["encoder"],
            "head": info["head"],
            "tokenizer": "tokenizer.json",
            "tensors": entries,
            "shards": shards,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        shutil.copy2(checkpoint / "tokenizer.json", out_dir / "tokenizer.json")
        total = sum(s["bytes"] for s in shards)
        print(name, precision, "->", out_dir.relative_to(ROOT),
              f"{len(entries)} tensors, {len(shards)} shards, {total / 1e6:.1f} MB")


def main():
    torch.set_num_threads(4)
    names = sys.argv[1:] or list(MODELS)
    for name in names:
        spec = MODELS[name]
        export_model(name, spec["checkpoint"], dict(spec["source"]))


if __name__ == "__main__":
    main()
