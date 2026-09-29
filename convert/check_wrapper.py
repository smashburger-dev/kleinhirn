"""Gate 1: unpatched fp32 wrapper vs native classify_text.

For each case the wrapper runs on collator-prepared arrays (L128, K16) and its
softmax confidence must match ``native.classify_text`` within 1e-4 and pick the
same label. Cases: the three upstream EXAMPLES plus the first 50 texts of the
public corpus (tests/corpus/texts1000.json). Runs for each given model.

Usage: .venv/bin/python convert/check_wrapper.py [model ...] [--texts PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gliner2 import AutoExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gliner2_export import (  # noqa: E402
    GLiNER2ClassificationExport,
    prepare_tasks_with_processor,
)

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ROOT / "tests/corpus/texts1000.json"
MODELS = {
    "small-upstream": ROOT / "models/small-upstream/checkpoint",
    "base-upstream": ROOT / "models/base-upstream/checkpoint",
    "multi-upstream": ROOT / "models/multi-upstream/checkpoint",
}
EXAMPLES = [
    ("The rocket launched successfully.", "topic", ["science", "sports", "politics"]),
    ("The team won the football championship.", "topic", ["science", "sports", "politics"]),
    ("The budget was approved by parliament.", "topic", ["science", "sports", "politics"]),
]
# Neutral demo labels for the public golden corpus; the task name and labels
# are schema fixtures only, they carry no benchmark meaning.
TASK = "topic"
LABELS = ["science", "sports", "politics", "technology", "business",
          "culture", "health", "education", "environment", "other"]
SECOND_TASK = "sentiment"
SECOND_LABELS = ["positive", "negative", "neutral"]
LENGTH = 128
MAX_OPTIONS = 16
PILOT_N = 50


def group_probs(logits: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Per-group softmax over valid markers; mirrors the native decode."""
    out = []
    for g in range(int(groups[mask > 0.5].max()) + 1 if (mask > 0.5).any() else 0):
        sel = (groups == g) & (mask > 0.5)
        vals = logits[sel]
        exps = np.exp(vals - vals.max())
        out.append(exps / exps.sum())
    return out


def check_model(name: str, checkpoint: Path, cases: list) -> dict:
    torch.set_num_threads(4)
    native = AutoExtractor.from_pretrained(str(checkpoint), map_location="cpu").eval()
    wrapper = GLiNER2ClassificationExport(native).eval()
    checked = skipped = 0
    max_conf_err = 0.0
    label_mismatch = []
    for text, task, labels in cases:
        try:
            arrays = prepare_tasks_with_processor(native.processor, text, {task: labels}, LENGTH, MAX_OPTIONS)
        except ValueError:
            skipped += 1
            continue
        tensors = tuple(
            torch.from_numpy(arrays[k])
            for k in ("input_ids", "attention_mask", "marker_indices", "marker_mask")
        )
        with torch.no_grad():
            logits = wrapper(*tensors)[0][0].numpy()
        probs = group_probs(logits, arrays["marker_groups"][0], arrays["marker_mask"][0])[0]
        confidence = float(probs.max())
        choice = labels[int(probs.argmax())]
        native_out = native.classify_text(text, {task: labels}, include_confidence=True, max_len=LENGTH)[task]
        err = abs(confidence - float(native_out["confidence"]))
        max_conf_err = max(max_conf_err, err)
        if choice != native_out["label"]:
            label_mismatch.append({"text": text, "wrapper": choice, "native": native_out["label"]})
        checked += 1
    return {
        "model": name,
        "checked": checked,
        "skipped": skipped,
        "max_confidence_error": max_conf_err,
        "label_mismatches": label_mismatch,
        "pass": max_conf_err <= 1e-4 and not label_mismatch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    ap.add_argument("--texts", type=Path, default=TEXTS)
    args = ap.parse_args()
    corpus = json.loads(args.texts.read_text())
    texts = corpus["texts"] if isinstance(corpus, dict) else corpus
    cases = EXAMPLES + [(t, TASK, LABELS) for t in texts[:PILOT_N]]
    results = []
    for name in args.models:
        result = check_model(name, MODELS[name], cases)
        print(json.dumps(result))
        results.append(result)
    if not all(r["pass"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
