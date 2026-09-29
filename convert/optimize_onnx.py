"""K20: offline-fused ORT graphs for the ONNX Runtime Web comparison.

Runs onnxruntime.transformers.optimizer on the ORT graphs we benchmark
against. onnxruntime 1.29 has no deberta or modernbert model type; both
encoders go through the documented BERT path (model_type "bert", default
opt_level 1: ORT basic optimizations, then the Python fusions). fp16 uses
the optimizer's own conversion of the fused graph (keep_io_types=True).

- small-upstream: models/small-upstream/onnx/model_f32.onnx (fixed L128
  K16, convert/export_onnx.py) -> model_f32_opt.onnx, model_f16_opt.onnx
- julia-1: models/julia-1/onnx/model.onnx (upstream dynamo export with
  external data) -> model_opt.onnx(.data), model_opt_f16.onnx(.data)

Per graph: op-type census, fused-operator statistics and CPU parity
against the PyTorch goldens (small: 968 corpus goldens, julia: the 100
upstream parity cases, compared with our recomputed PyTorch fp32 logits).
Report: bench/results/k20-optimize-<model>.json

Usage: .venv/bin/python convert/optimize_onnx.py [small-upstream|julia-1 ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model import OnnxModel

ROOT = Path(__file__).resolve().parents[1]
# Both encoders: hidden 384, 6 heads (deberta-v3-xsmall, mmBERT-small).
NUM_HEADS, HIDDEN = 6, 384
LENGTH, MAX_OPTIONS = 128, 16
SMALL_INPUTS = ["input_ids", "attention_mask", "marker_indices", "marker_mask"]


def census(model):
    ops = Counter(
        n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}:{n.op_type}"
        for n in model.graph.node)
    return {"nodes": len(model.graph.node), "ops": dict(ops.most_common())}


def small_feeds(item):
    arrays = [np.zeros((1, LENGTH), np.int32), np.zeros((1, LENGTH), np.int32),
              np.zeros((1, MAX_OPTIONS), np.int32), np.zeros((1, MAX_OPTIONS), np.float32)]
    arrays[0][0, : item["seq_len"]] = item["input_ids"]
    arrays[1][0, : item["seq_len"]] = 1
    n = len(item["marker_indices"])
    arrays[2][0, :n] = item["marker_indices"]
    arrays[3][0, :n] = item["marker_mask"]
    return dict(zip(SMALL_INPUTS, arrays)), n, np.asarray(item["logits"][:n])


def julia_feeds(item):
    k = len(item["markers"])
    feeds = {
        "input_ids": np.asarray([item["input_ids"]], np.int64),
        "attention_mask": np.ones((1, item["seq_len"]), np.int64),
        "marker_pos": np.asarray([item["markers"]], np.int64),
        "marker_mask": np.ones((1, k), bool),
        "qtype": np.asarray([item["qtype"]], np.int64),
    }
    return feeds, k, np.asarray(item["recomputed_logits"])


def parity(path, items, feeds_for):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_logit, agree = 0.0, 0
    for item in items:
        feeds, n, ref = feeds_for(item)
        logits = session.run(["logits"], feeds)[0][0, :n]
        max_logit = max(max_logit, float(np.abs(logits - ref).max()))
        agree += int(logits.argmax() == ref.argmax())
    return {"n": len(items), "max_abs_logit_diff": max_logit,
            "argmax_agreement": agree / len(items)}


CONFIGS = {
    "small-upstream": {
        "src": "models/small-upstream/onnx/model_f32.onnx",
        "f32": "models/small-upstream/onnx/model_f32_opt.onnx",
        "f16": "models/small-upstream/onnx/model_f16_opt.onnx",
        "external": False,
        "golden": "tests/golden/small-upstream/texts1000_l128k16.json",
        "feeds": small_feeds,
    },
    "julia-1": {
        "src": "models/julia-1/onnx/model.onnx",
        "f32": "models/julia-1/onnx/model_opt.onnx",
        "f16": "models/julia-1/onnx/model_opt_f16.onnx",
        "external": True,
        "golden": "tests/golden/julia-1/parity100.json",
        "feeds": julia_feeds,
    },
}


def file_bytes(path):
    data = Path(str(path) + ".data")
    return path.stat().st_size + (data.stat().st_size if data.exists() else 0)


def run(name):
    cfg = CONFIGS[name]
    src, f32, f16 = (ROOT / cfg[k] for k in ("src", "f32", "f16"))
    items = json.loads((ROOT / cfg["golden"]).read_text())["items"]
    before = census(onnx.load(str(src), load_external_data=False))

    fused = optimizer.optimize_model(
        str(src), model_type="bert", num_heads=NUM_HEADS, hidden_size=HIDDEN,
        optimization_options=FusionOptions("bert"))
    stats = fused.get_fused_operator_statistics()
    after = census(fused.model)
    fused.save_model_to_file(str(f32), use_external_data_format=cfg["external"])

    half = OnnxModel(onnx.load(str(f32)))
    half.convert_float_to_float16(keep_io_types=True)
    half.save_model_to_file(str(f16), use_external_data_format=cfg["external"])
    after_f16 = census(onnx.load(str(f16), load_external_data=False))

    report = {
        "model": name,
        "onnxruntime": ort.__version__,
        "optimizer": {"model_type": "bert", "opt_level": "default (1)",
                      "num_heads": NUM_HEADS, "hidden_size": HIDDEN},
        "census": {"source": before, "fused_f32": after, "fused_f16": after_f16},
        "fused_operator_statistics": stats,
        "attention_fused": bool(stats.get("Attention") or stats.get("MultiHeadAttention")),
        "bytes": {"source": file_bytes(src), "fused_f32": file_bytes(f32),
                  "fused_f16": file_bytes(f16)},
        "parity_cpu": {k: parity(p, items, cfg["feeds"])
                       for k, p in (("source", src), ("fused_f32", f32), ("fused_f16", f16))},
        "files": {"fused_f32": cfg["f32"], "fused_f16": cfg["f16"]},
    }
    out = ROOT / "bench" / "results" / f"k20-optimize-{name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in (
        "model", "fused_operator_statistics", "attention_fused", "bytes", "parity_cpu")}, indent=1))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    for model_name in sys.argv[1:] or list(CONFIGS):
        run(model_name)
