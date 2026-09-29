"""Executable specification of the Countdown judge forward pass, numpy only.

Loads ONLY manifest tensors (models/countdown-judge/f32/) and applies the
formulas from interference_search/judge.py: 29 features per number relative
to the target (src/search/judge.ts mirrors them), Linear(29->64), one learned
class token prepended, two nn.TransformerEncoderLayer blocks (pre-norm, full
masked self-attention over 8 positions, 4 heads of dim 16, RELU FFN 64->256
->64, biases everywhere, eps 1e-5), then LayerNorm + Linear(64->1) on the
class token. Upstream applies sigmoid outside the module; this spec emits
the raw logit like judge_golden.py stores it.

Gate (--check): against tests/golden/search/judge-parity.json. Like the K8
gate, an absolute bound against torch f32 is not attainable: torch's own
f32 output deviates up to ~1.3e-5 from its f64 result on some states. The
check therefore requires max |spec64 - ref64| <= 1e-5 (the spec must be
closer to the exact reference than the reference's own f32 noise, ~1.3e-5)
and reports the f32-vs-f32 distance for documentation.

Usage:
  .venv/bin/python convert/judge_forward.py [--check]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from julia_manifest_forward import load_tensors, ln_bias, mha

ROOT = Path(__file__).resolve().parents[1]
MODS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
MAX_N = 8  # 1 cls + 7 numbers


def num_feats(x: int, t: int) -> list[float]:
    lx, lt = np.log1p(x), np.log1p(t)
    f = [lx / 10, (lx - lt) / 5, float(x == t), float(x > t),
         float(t % x == 0), float(x % t == 0),
         min(abs(x - t), 1000) / 1000]
    f += [float(x % m == t % m) for m in MODS]
    f += [(x % m) / m for m in MODS]
    return f


def judge_forward(t: dict[str, np.ndarray], state: list[int], target: int,
                  dtype=np.float32) -> float:
    n_feat = 29
    rows = np.zeros((MAX_N, n_feat), dtype=dtype)
    mask = np.zeros(MAX_N, dtype=dtype)
    mask[0] = 1.0  # class token always valid
    for i, x in enumerate(state):
        rows[i + 1] = num_feats(x, target)
        mask[i + 1] = 1.0
    h = rows @ t["inp.weight"].astype(dtype).T + t["inp.bias"].astype(dtype)
    h[0] = t["cls"].astype(dtype)
    for i in range(2):
        p = f"layers.{i}"
        normed = ln_bias(h, t[f"{p}.norm1.weight"], t[f"{p}.norm1.bias"])
        h = h + mha(normed, t[f"{p}.in_proj.weight"], t[f"{p}.in_proj.bias"],
                    t[f"{p}.out_proj.weight"], t[f"{p}.out_proj.bias"],
                    4, mask, None)
        normed = ln_bias(h, t[f"{p}.norm2.weight"], t[f"{p}.norm2.bias"])
        ffn = np.maximum(0.0, normed @ t[f"{p}.linear1.weight"].T
                         + t[f"{p}.linear1.bias"])
        h = h + ffn @ t[f"{p}.linear2.weight"].T + t[f"{p}.linear2.bias"]
    s = ln_bias(h[:1], t["out.norm.weight"], t["out.norm.bias"])
    return float((s @ t["out.fc.weight"].T + t["out.fc.bias"])[0, 0])


def check() -> None:
    tensors, _ = load_tensors(ROOT / "models/countdown-judge/f32")
    tensors64 = {k: v.astype(np.float64) for k, v in tensors.items()}
    golden = json.loads(
        (ROOT / "tests/golden/search/judge-parity.json").read_text())
    n = len(golden["items"])
    worst32 = worst64 = worst_ref = 0.0
    for item in golden["items"]:
        lg32 = judge_forward(tensors, item["s"], item["t"])
        lg64 = judge_forward(tensors64, item["s"], item["t"],
                             dtype=np.float64)
        worst32 = max(worst32, abs(lg32 - item["logit"]))
        worst64 = max(worst64, abs(lg64 - item["logit64"]))
        worst_ref = max(worst_ref, abs(item["logit"] - item["logit64"]))
    status = "PASS" if worst64 <= 1e-5 else "FAIL"
    print(f"judge-parity: {n} states, max |spec64 - ref64| {worst64:.3e} "
          f"(gate <= 1e-5) {status}")
    print(f"  reference f32 self-noise |ref32 - ref64| {worst_ref:.3e}, "
          f"max |spec32 - ref32| {worst32:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    check()
