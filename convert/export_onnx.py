"""ONNX export of the unpatched GLiNER2 classification wrapper (K2 baseline).

Fixed shape: input_ids/attention_mask [1, 128] int32, marker_indices [1, 16]
int32, marker_mask [1, 16] f32, opset 17. Written to models/<model>/onnx/ as
model_f32.onnx and (via onnxconverter-common, keep_io_types=True)
model_f16.onnx.

Checks: onnxruntime CPU vs the PyTorch goldens. f32: max logit deviation
<= 1e-4. f16: argmax agreement >= 99.5 %. Both on the 1000 corpus goldens.

Usage: .venv/bin/python convert/export_onnx.py [model ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from gliner2 import AutoExtractor
from onnxruntime.transformers import float16 as ort_f16

from transformers.models.deberta_v2.modeling_deberta_v2 import (  # noqa: E402
    DebertaV2Encoder,
    build_relative_position,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_wrapper import LABELS, TASK  # noqa: E402
from gliner2_export import (  # noqa: E402
    GLiNER2ClassificationExport,
    prepare_tasks_with_processor,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "small-upstream": ROOT / "models/small-upstream/checkpoint",
}
LENGTH = 128
MAX_OPTIONS = 16
INPUT_NAMES = ["input_ids", "attention_mask", "marker_indices", "marker_mask"]
SAMPLE_TEXT = "The rocket launched successfully."
SAMPLE_LABELS = ["science", "sports", "politics"]


def export_f32(native, wrapper, out_path: Path):
    arrays = prepare_tasks_with_processor(
        native.processor, SAMPLE_TEXT, {"topic": SAMPLE_LABELS}, LENGTH, MAX_OPTIONS)
    tensors = tuple(torch.from_numpy(arrays[k]) for k in INPUT_NAMES)
    # The sequence length is fixed at export: relative_pos is constant, so we
    # freeze get_rel_pos (same idea as the CoreML build_rpos freeze). It also
    # keeps index math out of the fp16 conversion, which mis-casts it.
    # Verified: the ORT check below compares against the unpatched goldens.
    dummy = torch.zeros(1, 1, LENGTH, 1)
    rel_pos = build_relative_position(
        dummy, dummy,
        bucket_size=native.encoder.config.position_buckets,
        max_position=native.encoder.config.max_position_embeddings,
    )
    original = DebertaV2Encoder.get_rel_pos
    with torch.no_grad():
        reference = wrapper(*tensors)[0][0, : len(SAMPLE_LABELS)]
    DebertaV2Encoder.get_rel_pos = lambda self, h, query_states=None, relative_pos=None: rel_pos
    try:
        with torch.no_grad():
            patched = wrapper(*tensors)[0][0, : len(SAMPLE_LABELS)]
            error = float((patched - reference).abs().max())
            if error > 1e-4:
                raise RuntimeError(f"Frozen get_rel_pos changed logits: {error}")
        torch.onnx.export(
            wrapper, tensors, str(out_path),
            input_names=INPUT_NAMES,
            output_names=["logits", "probabilities"],
            opset_version=17,
            dynamo=False,
        )
    finally:
        DebertaV2Encoder.get_rel_pos = original
    onnx.checker.check_model(str(out_path))


def ort_inputs(item):
    ids = np.zeros((1, LENGTH), dtype=np.int32)
    ids[0, : item["seq_len"]] = item["input_ids"]
    mask = np.zeros((1, LENGTH), dtype=np.int32)
    mask[0, : item["seq_len"]] = 1
    n = len(item["marker_indices"])
    markers = np.zeros((1, MAX_OPTIONS), dtype=np.int32)
    markers[0, :n] = item["marker_indices"]
    mmask = np.zeros((1, MAX_OPTIONS), dtype=np.float32)
    mmask[0, :n] = item["marker_mask"]
    return dict(zip(INPUT_NAMES, (ids, mask, markers, mmask))), n


def check_model(session, items):
    max_logit = 0.0
    agree = 0
    for item in items:
        feeds, n = ort_inputs(item)
        logits = session.run(["logits"], feeds)[0][0, :n]
        max_logit = max(max_logit, float(np.abs(logits - item["logits"]).max()))
        agree += int(int(logits.argmax()) == item["label_index"])
    return {"n": len(items), "max_abs_logit_diff": max_logit,
            "argmax_agreement": agree / len(items)}


def export_model(name: str, checkpoint: Path):
    torch.set_num_threads(4)
    native = AutoExtractor.from_pretrained(str(checkpoint), map_location="cpu").eval()
    wrapper = GLiNER2ClassificationExport(native).eval()
    out_dir = ROOT / "models" / name / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = json.loads((ROOT / "tests" / "golden" / name / "texts1000_l128k16.json").read_text())
    items = gold["items"]

    f32_path = out_dir / "model_f32.onnx"
    export_f32(native, wrapper, f32_path)
    session = ort.InferenceSession(str(f32_path), providers=["CPUExecutionProvider"])
    f32_report = check_model(session, items)

    f16_path = out_dir / "model_f16.onnx"
    model = onnx.load(str(f32_path))
    # The transformer-aware converter keeps blocked ops (LayerNorm etc.) in
    # fp32 and inserts matching boundary casts; onnxconverter-common produced
    # type-mismatched graphs on this model.
    fp16_model = ort_f16.convert_float_to_float16(
        model, keep_io_types=True, force_fp16_initializers=True)
    onnx.save_model(fp16_model, str(f16_path))
    session16 = ort.InferenceSession(str(f16_path), providers=["CPUExecutionProvider"])
    f16_report = check_model(session16, items)

    report = {
        "model": name,
        "f32": {**f32_report, "pass": f32_report["max_abs_logit_diff"] <= 1e-4},
        "f16": {**f16_report, "pass": f16_report["argmax_agreement"] >= 0.995},
        "files": {
            "f32": str(f32_path.relative_to(ROOT)),
            "f16": str(f16_path.relative_to(ROOT)),
            "f32_bytes": f32_path.stat().st_size,
            "f16_bytes": f16_path.stat().st_size,
        },
    }
    print(json.dumps(report))
    if not (report["f32"]["pass"] and report["f16"]["pass"]):
        sys.exit(1)


def main():
    names = sys.argv[1:] or list(MODELS)
    for name in names:
        export_model(name, MODELS[name])


if __name__ == "__main__":
    main()
