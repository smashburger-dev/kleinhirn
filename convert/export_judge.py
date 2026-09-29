"""Countdown judge weight export into the kleinhirn manifest format (K17).

Reads models/countdown-judge/countdown_judge.pt (upstream
interference-search, pinned revision in REVISIONS.txt) and writes
models/countdown-judge/<precision>/manifest.json plus weights-0.bin.

Layout (state_dict -> manifest):
- cls [1,1,64] -> cls (class token prepended after the input projection)
- inp.* -> inp.* (Linear 29 -> 64)
- tf.layers.N.self_attn.in_proj_* -> layers.N.in_proj.* (packed q|k|v, 192x64)
- tf.layers.N.self_attn.out_proj.* -> layers.N.out_proj.*
- tf.layers.N.norm{1,2}.* -> layers.N.norm{1,2}.*
- tf.layers.N.linear{1,2}.* -> layers.N.linear{1,2}.* (FFN 64 -> 256 -> 64,
  ReLU; nn.TransformerEncoderLayer, pre-norm, batch_first)
- out.0.* -> out.norm.* (LayerNorm), out.1.* -> out.fc.* (Linear 64 -> 1,
  alive logit; sigmoid is applied on the CPU like upstream's load_judge)

Usage: .venv/bin/python convert/export_judge.py [--src PATH]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/countdown-judge/countdown_judge.pt"
ALIGN = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()
    src = Path(args.src)
    sd = torch.load(src, map_location="cpu")

    tensors: list[tuple[str, np.ndarray]] = [
        ("cls", sd["cls"].reshape(64).numpy()),
        ("inp.weight", sd["inp.weight"].numpy()),
        ("inp.bias", sd["inp.bias"].numpy()),
    ]
    for i in range(2):
        p = f"tf.layers.{i}"
        tensors += [
            (f"layers.{i}.in_proj.weight", sd[f"{p}.self_attn.in_proj_weight"].numpy()),
            (f"layers.{i}.in_proj.bias", sd[f"{p}.self_attn.in_proj_bias"].numpy()),
            (f"layers.{i}.out_proj.weight", sd[f"{p}.self_attn.out_proj.weight"].numpy()),
            (f"layers.{i}.out_proj.bias", sd[f"{p}.self_attn.out_proj.bias"].numpy()),
            (f"layers.{i}.norm1.weight", sd[f"{p}.norm1.weight"].numpy()),
            (f"layers.{i}.norm1.bias", sd[f"{p}.norm1.bias"].numpy()),
            (f"layers.{i}.norm2.weight", sd[f"{p}.norm2.weight"].numpy()),
            (f"layers.{i}.norm2.bias", sd[f"{p}.norm2.bias"].numpy()),
            (f"layers.{i}.linear1.weight", sd[f"{p}.linear1.weight"].numpy()),
            (f"layers.{i}.linear1.bias", sd[f"{p}.linear1.bias"].numpy()),
            (f"layers.{i}.linear2.weight", sd[f"{p}.linear2.weight"].numpy()),
            (f"layers.{i}.linear2.bias", sd[f"{p}.linear2.bias"].numpy()),
        ]
    tensors += [
        ("out.norm.weight", sd["out.0.weight"].numpy()),
        ("out.norm.bias", sd["out.0.bias"].numpy()),
        ("out.fc.weight", sd["out.1.weight"].numpy()),
        ("out.fc.bias", sd["out.1.bias"].numpy()),
    ]

    encoder = {
        "arch": "setjudge-countdown",
        "layers": 2,
        "hiddenSize": 64,
        "heads": 4,
        "headDim": 16,
        "ffn": 256,
        "nFeat": 29,
        "maxN": 8,  # 1 cls token + 7 numbers
        "normEps": 1e-5,
    }
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()

    for precision, dtype in (("f32", np.float32), ("f16", np.float16)):
        out_dir = ROOT / "models/countdown-judge" / precision
        out_dir.mkdir(parents=True, exist_ok=True)
        blob = bytearray()
        entries = []
        for name, value in tensors:
            cast = value.astype(dtype)
            raw = cast.tobytes()
            while len(blob) % ALIGN:
                blob.append(0)
            entries.append({
                "name": name, "dtype": precision,
                "shape": list(value.shape), "shard": 0,
                "offset": len(blob), "byteLength": len(raw),
            })
            blob += raw
        (out_dir / "weights-0.bin").write_bytes(bytes(blob))
        manifest = {
            "format": "kleinhirn-weights-1",
            "source": {
                "repo": "github.com/Badtheorylabs/interference-search",
                "revision": "afedcc23aac17bb039df8c72a834b4a43537eac3",
                "checkpointSha256": src_sha,
            },
            "encoder": encoder,
            "head": {"type": "setjudge-alive", "temperature": 1.0,
                     "hiddenSize": 64},
            "shards": [{
                "file": "weights-0.bin", "bytes": len(blob),
                "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
            }],
            "tensors": entries,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"{precision}: {len(entries)} tensors, "
              f"{len(blob)} bytes -> {out_dir}")


if __name__ == "__main__":
    main()
