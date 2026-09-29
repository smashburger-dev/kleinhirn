"""Golden generation for kleinhirn.

Per model (small-upstream, base-upstream, multi-upstream), unpatched fp32
wrapper over the public 1000-text corpus (tests/corpus/texts1000.json):

1. tests/golden/<model>/texts1000_l128k16.json
   All 1000 corpus texts, task "topic" with the 10 demo labels, bucket
   L128 K16. Per text: unpadded input_ids/attention_mask (seq_len),
   marker_indices/marker_mask/marker_groups, logits and probabilities
   (f32, valid labels only), label_index.
2. tests/golden/<model>/two_tasks_l128k16.json
   20 corpus texts with topic + sentiment in ONE schema sequence.
   Marker groups follow the collator's cls_group_index (dense per-task
   ids); softmax is per task. Native classify_text reference is embedded.
3. models/<model>/golden/layers.{bin,index.json}
   8 texts x {L128, L256}: embedding output after LayerNorm (pre-mask and
   post-mask), hidden state after each of the 12 layers, final logits.
   >=2 texts have >129 input ids so the logarithmic DeBERTa buckets at
   |i-j| >= 129 are exercised (L256 only, they do not fit L128).
4. tests/golden/<model>/tokenizer_ids.json
   Pure tokenizer ids (no schema, add_special_tokens=False) for all 1000
   corpus texts.

Usage: .venv/bin/python convert/golden.py [model ...] [--texts PATH] [--long]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from gliner2 import AutoExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_wrapper import (  # noqa: E402
    LABELS, MAX_OPTIONS, SECOND_LABELS, SECOND_TASK, TASK,
)
from gliner2_export import (  # noqa: E402
    GLiNER2ClassificationExport,
    load_processor,
    prepare_tasks_with_processor,
)

ROOT = Path(__file__).resolve().parents[1]
TEXTS_FILE = ROOT / "tests/corpus/texts1000.json"
MODELS = {
    "small-upstream": ROOT / "models/small-upstream/checkpoint",
    "base-upstream": ROOT / "models/base-upstream/checkpoint",
    "multi-upstream": ROOT / "models/multi-upstream/checkpoint",
}
EVENT_LABELS = LABELS


def per_group(logits, groups, mask, task_label_counts):
    """Split valid marker logits into dense task groups -> logits, probs, argmax."""
    out = []
    offset = 0
    for g, n in enumerate(task_label_counts):
        sel = (groups == g) & (mask > 0.5)
        vals = logits[sel]
        assert len(vals) == n, f"group {g}: expected {n} markers, got {len(vals)}"
        exps = np.exp(vals - vals.max())
        out.append({
            "logits": vals.astype(np.float32).tolist(),
            "probabilities": (exps / exps.sum()).astype(np.float32).tolist(),
            "label_index": int(vals.argmax()),
        })
        offset += n
    assert offset == int(mask.sum())
    return out


def run_wrapper(wrapper, arrays):
    tensors = tuple(
        torch.from_numpy(arrays[k])
        for k in ("input_ids", "attention_mask", "marker_indices", "marker_mask")
    )
    with torch.no_grad():
        logits, _ = wrapper(*tensors)
    return logits[0].numpy()


def corpus_goldens(native, wrapper, texts):
    goldens = []
    skipped = 0
    for text in texts:
        try:
            arrays = prepare_tasks_with_processor(
                native.processor, text, {TASK: EVENT_LABELS}, 128, MAX_OPTIONS)
        except ValueError:
            skipped += 1
            continue
        seq = arrays["seq_len"]
        logits = run_wrapper(wrapper, arrays)
        task_out = per_group(
            logits, arrays["marker_groups"][0], arrays["marker_mask"][0],
            [len(EVENT_LABELS)],
        )[0]
        goldens.append({
            "title": text,
            "seq_len": seq,
            "input_ids": arrays["input_ids"][0, :seq].tolist(),
            "attention_mask": arrays["attention_mask"][0, :seq].tolist(),
            "marker_indices": arrays["marker_indices"][0, :len(EVENT_LABELS)].tolist(),
            "marker_mask": arrays["marker_mask"][0, :len(EVENT_LABELS)].tolist(),
            "marker_groups": arrays["marker_groups"][0, :len(EVENT_LABELS)].tolist(),
            "logits": task_out["logits"],
            "probabilities": task_out["probabilities"],
            "label_index": task_out["label_index"],
        })
    return {"task": TASK, "labels": EVENT_LABELS, "bucket": {"length": 128, "max_options": MAX_OPTIONS},
            "count": len(goldens), "skipped": skipped, "items": goldens}


def two_task_goldens(native, wrapper, texts):
    tasks = {TASK: EVENT_LABELS, SECOND_TASK: SECOND_LABELS}
    goldens = []
    skipped = 0
    for text in texts:
        if len(goldens) >= 20:
            break
        try:
            arrays = prepare_tasks_with_processor(native.processor, text, tasks, 128, MAX_OPTIONS)
        except ValueError:
            skipped += 1
            continue
        seq = arrays["seq_len"]
        logits = run_wrapper(wrapper, arrays)
        per_task = per_group(
            logits, arrays["marker_groups"][0], arrays["marker_mask"][0],
            [len(EVENT_LABELS), len(SECOND_LABELS)],
        )
        reference = native.classify_text(text, tasks, include_confidence=True, max_len=128)
        n_markers = len(EVENT_LABELS) + len(SECOND_LABELS)
        goldens.append({
            "title": text,
            "seq_len": seq,
            "input_ids": arrays["input_ids"][0, :seq].tolist(),
            "attention_mask": arrays["attention_mask"][0, :seq].tolist(),
            "marker_indices": arrays["marker_indices"][0, :n_markers].tolist(),
            "marker_mask": arrays["marker_mask"][0, :n_markers].tolist(),
            "marker_groups": arrays["marker_groups"][0, :n_markers].tolist(),
            "tasks": [
                {"task": TASK, "labels": EVENT_LABELS, **per_task[0],
                 "native": reference[TASK]},
                {"task": SECOND_TASK, "labels": SECOND_LABELS, **per_task[1],
                 "native": reference[SECOND_TASK]},
            ],
        })
    return {"tasks": tasks, "bucket": {"length": 128, "max_options": MAX_OPTIONS},
            "count": len(goldens), "skipped": skipped, "items": goldens}


def capture_forward(native, input_ids, attention_mask):
    """Run the encoder once, returning post-LN embeddings (pre/post mask),
    the hidden state after each layer and the final state."""
    captured = {}
    hooks = []
    hooks.append(native.encoder.embeddings.LayerNorm.register_forward_hook(
        lambda m, i, o: captured.__setitem__("embedding_ln", o.detach().numpy())))
    hooks.append(native.encoder.embeddings.register_forward_hook(
        lambda m, i, o: captured.__setitem__("embedding_masked", o.detach().numpy())))
    for index, layer in enumerate(native.encoder.encoder.layer):
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, index=index: captured.__setitem__(f"layer_{index}", o[0].detach().numpy())))
    try:
        with torch.no_grad():
            out = native.encoder(
                input_ids=torch.from_numpy(input_ids).long(),
                attention_mask=torch.from_numpy(attention_mask).long(),
            )
        captured["last_hidden_state"] = out.last_hidden_state.numpy()
    finally:
        for h in hooks:
            h.remove()
    return captured


def layer_goldens(native, wrapper, texts, model_dir):
    # Six plain texts plus two composites that exceed 129 input ids in L256
    # (logarithmic relative buckets only kick in above |i-j| = 128).
    plain = texts[:6]
    composites = []
    start = 0
    while len(composites) < 2:
        text = " ".join(texts[start:start + 9])
        try:
            arrays = prepare_tasks_with_processor(
                native.processor, text, {TASK: EVENT_LABELS}, 256, MAX_OPTIONS)
            if arrays["seq_len"] > 129:
                composites.append(text)
        except ValueError:
            pass
        start += 1
        if start > 200:
            raise RuntimeError("could not compose a >129 token text")
    texts_ = plain + composites
    index = {"model": model_dir.name, "tensors": {}, "cases": []}
    blob = bytearray()
    for text in texts_:
        case = {"title": text, "buckets": {}}
        for length in (128, 256):
            try:
                arrays = prepare_tasks_with_processor(
                    native.processor, text, {TASK: EVENT_LABELS}, length, MAX_OPTIONS)
            except ValueError:
                case["buckets"][f"L{length}"] = None
                continue
            logits = run_wrapper(wrapper, arrays)
            captured = capture_forward(native, arrays["input_ids"], arrays["attention_mask"])
            names = ["embedding_ln", "embedding_masked"] + [f"layer_{i}" for i in range(12)]
            tensors = {}
            for name in names:
                arr = np.ascontiguousarray(captured[name][0], dtype=np.float32)
                tensors[name] = {"offset": len(blob), "shape": list(arr.shape), "dtype": "f32"}
                blob += arr.tobytes()
            task_out = per_group(
                logits, arrays["marker_groups"][0], arrays["marker_mask"][0],
                [len(EVENT_LABELS)],
            )[0]
            case["buckets"][f"L{length}"] = {
                "seq_len": arrays["seq_len"],
                "input_ids": arrays["input_ids"][0].tolist(),
                "attention_mask": arrays["attention_mask"][0].tolist(),
                "marker_indices": arrays["marker_indices"][0, :len(EVENT_LABELS)].tolist(),
                "marker_mask": arrays["marker_mask"][0, :len(EVENT_LABELS)].tolist(),
                "marker_groups": arrays["marker_groups"][0, :len(EVENT_LABELS)].tolist(),
                "tensors": tensors,
                "logits": task_out["logits"],
                "probabilities": task_out["probabilities"],
                "label_index": task_out["label_index"],
            }
        case["skipped_L128"] = case["buckets"]["L128"] is None
        index["cases"].append(case)
    index["tensor_order"] = ["embedding_ln", "embedding_masked"] + [f"layer_{i}" for i in range(12)]
    index["blob_bytes"] = len(blob)
    golden_dir = model_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / "layers.bin").write_bytes(bytes(blob))
    (golden_dir / "layers.index.json").write_text(json.dumps(index))
    return index


def composed_texts(texts, length):
    """Deterministically composed corpus texts (seed depends on the bucket).
    The caller filters by seq_len so every emitted case is used at most once."""
    rng = random.Random(1000 + length)
    seen = set()
    while True:
        n = rng.randint(max(3, length // 40), max(4, length // 20))
        idxs = tuple(sorted(rng.randrange(len(texts)) for _ in range(n)))
        if idxs in seen:
            continue
        seen.add(idxs)
        yield " ".join(texts[i] for i in idxs)


def long_bucket_goldens(native, wrapper, texts, length, count=200):
    """`count` logit cases composed from corpus texts for one long bucket.
    Only texts exceeding length // 2 tokens count, so every case really
    needs this bucket and cannot have run in the next smaller one."""
    goldens = []
    tried = 0
    for text in composed_texts(texts, length):
        tried += 1
        if tried > count * 60:
            raise RuntimeError(f"could not compose {count} texts for L{length}")
        try:
            arrays = prepare_tasks_with_processor(
                native.processor, text, {TASK: EVENT_LABELS}, length, MAX_OPTIONS)
        except ValueError:
            continue
        seq = arrays["seq_len"]
        if seq <= length // 2:
            continue
        logits = run_wrapper(wrapper, arrays)
        task_out = per_group(
            logits, arrays["marker_groups"][0], arrays["marker_mask"][0],
            [len(EVENT_LABELS)],
        )[0]
        goldens.append({
            "title": text,
            "seq_len": seq,
            "input_ids": arrays["input_ids"][0, :seq].tolist(),
            "attention_mask": arrays["attention_mask"][0, :seq].tolist(),
            "marker_indices": arrays["marker_indices"][0, :len(EVENT_LABELS)].tolist(),
            "marker_mask": arrays["marker_mask"][0, :len(EVENT_LABELS)].tolist(),
            "marker_groups": arrays["marker_groups"][0, :len(EVENT_LABELS)].tolist(),
            "logits": task_out["logits"],
            "probabilities": task_out["probabilities"],
            "label_index": task_out["label_index"],
        })
        if len(goldens) >= count:
            break
    return {"task": TASK, "labels": EVENT_LABELS,
            "bucket": {"length": length, "max_options": MAX_OPTIONS},
            "count": len(goldens), "items": goldens}


def long_layer_goldens(native, wrapper, texts, model_dir):
    """8 long texts (seq_len 300-512) with layer states and logits, each run
    in the L512 and the L1024 bucket. Written next to the L128/L256 layer
    goldens as layers_long.{bin,index.json} (not in git)."""
    texts_ = []
    for text in composed_texts(texts, 512):
        try:
            arrays = prepare_tasks_with_processor(
                native.processor, text, {TASK: EVENT_LABELS}, 512, MAX_OPTIONS)
        except ValueError:
            continue
        if arrays["seq_len"] > 300:
            texts_.append(text)
        if len(texts_) >= 8:
            break
    if len(texts_) < 8:
        raise RuntimeError("could not compose 8 long layer texts")

    index = {"model": model_dir.name, "tensors": {}, "cases": []}
    blob = bytearray()
    names = ["embedding_ln", "embedding_masked"] + [f"layer_{i}" for i in range(12)]
    for text in texts_:
        case = {"title": text, "buckets": {}}
        for length in (512, 1024):
            arrays = prepare_tasks_with_processor(
                native.processor, text, {TASK: EVENT_LABELS}, length, MAX_OPTIONS)
            logits = run_wrapper(wrapper, arrays)
            captured = capture_forward(
                native, arrays["input_ids"], arrays["attention_mask"])
            tensors = {}
            for name in names:
                arr = np.ascontiguousarray(captured[name][0], dtype=np.float32)
                tensors[name] = {"offset": len(blob), "shape": list(arr.shape), "dtype": "f32"}
                blob += arr.tobytes()
            task_out = per_group(
                logits, arrays["marker_groups"][0], arrays["marker_mask"][0],
                [len(EVENT_LABELS)],
            )[0]
            case["buckets"][f"L{length}"] = {
                "seq_len": arrays["seq_len"],
                "input_ids": arrays["input_ids"][0].tolist(),
                "attention_mask": arrays["attention_mask"][0].tolist(),
                "marker_indices": arrays["marker_indices"][0, :len(EVENT_LABELS)].tolist(),
                "marker_mask": arrays["marker_mask"][0, :len(EVENT_LABELS)].tolist(),
                "marker_groups": arrays["marker_groups"][0, :len(EVENT_LABELS)].tolist(),
                "tensors": tensors,
                "logits": task_out["logits"],
                "probabilities": task_out["probabilities"],
                "label_index": task_out["label_index"],
            }
        index["cases"].append(case)
    index["tensor_order"] = names
    index["blob_bytes"] = len(blob)
    golden_dir = model_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / "layers_long.bin").write_bytes(bytes(blob))
    (golden_dir / "layers_long.index.json").write_text(json.dumps(index))
    return index


def tokenizer_goldens(processor, texts):
    tokenizer = processor.tokenizer
    items = []
    for text in texts:
        items.append({
            "text": text,
            "ids": tokenizer(text, add_special_tokens=False)["input_ids"],
        })
    return {"add_special_tokens": False, "vocab_size": int(tokenizer.vocab_size),
            "count": len(items), "items": items}


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    ap.add_argument("--texts", type=Path, default=TEXTS_FILE)
    ap.add_argument("--long", action="store_true",
                    help="only the long-bucket and long-layer goldens")
    args = ap.parse_args()
    corpus = json.loads(args.texts.read_text())
    texts = corpus["texts"] if isinstance(corpus, dict) else corpus
    for name in args.models:
        checkpoint = MODELS[name]
        native = AutoExtractor.from_pretrained(str(checkpoint), map_location="cpu").eval()
        wrapper = GLiNER2ClassificationExport(native).eval()
        out_dir = ROOT / "tests/golden" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.long:
            gold = corpus_goldens(native, wrapper, texts)
            (out_dir / "texts1000_l128k16.json").write_text(json.dumps(gold))
            print(name, "corpus:", gold["count"], "goldens,", gold["skipped"], "skipped")

            two = two_task_goldens(native, wrapper, texts)
            (out_dir / "two_tasks_l128k16.json").write_text(json.dumps(two, indent=1))
            print(name, "two-task:", two["count"], "goldens,", two["skipped"], "skipped")

            idx = layer_goldens(native, wrapper, texts, checkpoint.parent)
            print(name, "layers:", len(idx["cases"]), "cases")

            ids = tokenizer_goldens(native.processor, texts)
            (out_dir / "tokenizer_ids.json").write_text(json.dumps(ids, ensure_ascii=False))
            print(name, "tokenizer:", ids["count"], "texts")

        for length in (256, 512, 1024):
            long = long_bucket_goldens(native, wrapper, texts, length)
            (out_dir / f"long200_l{length}k16.json").write_text(json.dumps(long))
            print(name, f"long L{length}:", long["count"], "goldens")

        idx = long_layer_goldens(native, wrapper, texts, checkpoint.parent)
        print(name, "long layers:", len(idx["cases"]), "cases")


if __name__ == "__main__":
    main()
