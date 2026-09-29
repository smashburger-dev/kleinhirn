"""Wide-schema goldens for >16-label classification (banking77, 72 labels).

One schema per row embeds ALL 72 hypothesis labels; the PyTorch wrapper is
the reference for marker logits. Bucket: L1280 with 80 marker slots (the
engine keeps buffer sizes far under the 128 MiB binding limit even at
K=80, so no raised device limits are needed). Rows whose schema exceeds
the bucket are recorded with seq_len as None.

Usage: .venv/bin/python convert/wide_golden.py [model ...]
"""
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
MODELS = {
    "base-upstream": ROOT / "models/base-upstream/checkpoint",
    "multi-upstream": ROOT / "models/multi-upstream/checkpoint",
}
TASKS_FILE = ROOT / "models/k9-data/tasks/banking77.jsonl"
LENGTH = 1280
MAX_MARKERS = 80


def main():
    torch.set_num_threads(4)
    names = sys.argv[1:] or list(MODELS)
    rows = [json.loads(line) for line in TASKS_FILE.open()][:20]
    for name in names:
        native = AutoExtractor.from_pretrained(
            str(MODELS[name]), map_location="cpu").eval()
        wrapper = GLiNER2ClassificationExport(native).eval()
        items = []
        over = 0
        for r in rows:
            labels = list(r["labels"])
            try:
                arrays = prepare_tasks_with_processor(
                    native.processor, r["text"], {"intent": labels},
                    LENGTH, MAX_MARKERS)
            except ValueError:
                over += 1
                continue
            tensors = tuple(
                torch.from_numpy(arrays[k]) for k in (
                    "input_ids", "attention_mask",
                    "marker_indices", "marker_mask"))
            with torch.no_grad():
                logits, _ = wrapper(*tensors)
            logits = logits[0].numpy()
            seq = int(arrays["seq_len"])
            n = len(labels)
            valid = logits[arrays["marker_mask"][0] > 0.5]
            assert len(valid) == n
            items.append({
                "example_id": r["example_id"],
                "labels": labels, "target_index": r["target_index"],
                "seq_len": seq,
                "input_ids": arrays["input_ids"][0, :seq].tolist(),
                "attention_mask": arrays["attention_mask"][0, :seq].tolist(),
                "marker_indices": arrays["marker_indices"][0, :n].tolist(),
                "marker_mask": arrays["marker_mask"][0, :n].tolist(),
                "marker_groups": arrays["marker_groups"][0, :n].tolist(),
                "logits": valid.astype(np.float32).tolist(),
                "label_index": int(valid.argmax()),
            })
        out = {
            "task": "intent", "bucket": {"length": LENGTH, "max_options": MAX_MARKERS},
            "count": len(items), "over_bucket": over, "items": items,
        }
        out_dir = ROOT / "tests" / "golden" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"banking77_l{LENGTH}k{MAX_MARKERS}.json"
        path.write_text(json.dumps(out))
        print(name, "wide:", len(items), "goldens,", over, "over bucket")


if __name__ == "__main__":
    main()
