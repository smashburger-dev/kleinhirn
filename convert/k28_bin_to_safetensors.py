"""Convert a pytorch_model.bin checkpoint to safetensors (K28.2 step 4).

Only for repos without model.safetensors. Loads with torch.load(...,
weights_only=True) and writes models/k28/<slug>/model.safetensors, which
tools/k28_convert.ts reads in place of repo/model.safetensors. The repo
directory stays unchanged.

Usage: .venv-k28/bin/python convert/k28_bin_to_safetensors.py <model-id> [--bin PATH] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import model_dir, repo_dir  # noqa: E402


def convert(bin_path: Path, out_path: Path) -> int:
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # save_file rejects tensors that share memory; the copy also makes them contiguous.
    tensors = {k: v.detach().clone().contiguous() for k, v in state.items() if torch.is_tensor(v)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"format": "pt"})
    return len(tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bin_path = args.bin or repo_dir(args.model) / "pytorch_model.bin"
    out_path = args.out or model_dir(args.model) / "model.safetensors"
    print(f"{convert(bin_path, out_path)} tensors -> {out_path}")


if __name__ == "__main__":
    main()
