"""K9 GLiNER2.5 route: zero-shot classification, options as labels,
state-plus-question as text. Runs only on tasks whose Julia gate passed.

Usage:
  .venv/bin/python bench/k9/run_gliner.py --task agnews --model small
  .venv/bin/python bench/k9/run_gliner.py --task typed --model multi --limit 100
"""
import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

MODELS = {
    'small': ('fastino/gliner2.5-small-v1',
              '3ec6d3dd7e1e93a7cf9b46096fa47aeda61c711c'),
    'base': ('fastino/gliner2.5-base-v1',
             'b0c10b23313ec3ff028821dff298dd743e010706'),
    'multi': ('fastino/gliner2.5-multi-v1',
              '235cf92d6d4318da9bfca0d08975c8fa7250d13b'),
}


def gliner_text(row):
    state = row['state']
    state_text = state if isinstance(state, str) else json.dumps(
        state, ensure_ascii=False)
    return state_text + '\n' + row['question']


def load_average():
    out = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
    return out.split('load average')[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['typed', 'agnews', 'emotiondair', 'banking77'])
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()

    model_id, revision = MODELS[args.model]
    output = args.output or (common.ROOT / 'bench' / 'results' /
                             f'k9-{args.task}-gliner-{args.model}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    load_start = load_average()

    pairs = common.load_task_pairs(args.task, limit=args.limit)

    from gliner2.classification import ClassificationSchema, Classifier
    print(f'loading {model_id}@{revision[:8]}...', flush=True)
    classifier = Classifier.from_pretrained(
        model_id, revision=revision, device='cpu').eval()
    schemas = {}

    def schema(labels):
        key = tuple(labels)
        if key not in schemas:
            schemas[key] = ClassificationSchema().single('label', list(labels))
        return schemas[key]

    predictions, stats = [], collections.Counter()
    t0 = time.monotonic()
    for i, (row, meta) in enumerate(pairs):
        started = time.perf_counter()
        try:
            result = classifier.classify(gliner_text(row), schema(row['options']))
            probs = {label: float(result.probabilities('label')[label])
                     for label in row['options']}
            index = max(range(len(row['options'])),
                        key=lambda j: probs[row['options'][j]])
            status = 'ok'
        except Exception:
            status, index, probs = 'failed', None, None
        latency = time.perf_counter() - started
        predicted = meta['keys'][index] if index is not None else None
        correct = predicted == meta['gold']
        stats['correct' if correct else 'wrong'] += 1
        predictions.append(dict(id=meta['id'], type=meta.get('type', 'choice'),
                                gold=meta['gold'], predicted=predicted,
                                correct=correct, status=status,
                                probabilities=probs,
                                latency_ms=round(latency * 1000, 2)))
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(pairs)} correct={stats["correct"]}', flush=True)
    elapsed = time.monotonic() - t0

    total = len(predictions)
    correct = stats['correct']
    by_type = collections.defaultdict(lambda: dict(count=0, correct=0))
    for p in predictions:
        by_type[p['type']]['count'] += 1
        by_type[p['type']]['correct'] += int(p['correct'])
    result = dict(
        task=args.task, route='gliner-pytorch-cpu', model=model_id,
        revision=revision, zero_shot=True,
        total=total, correct=correct,
        accuracy=correct / total if total else 0,
        failed=sum(1 for p in predictions if p['status'] != 'ok'),
        by_type={k: dict(v, accuracy=v['correct'] / v['count'])
                 for k, v in by_type.items()},
        elapsed_seconds=round(elapsed, 1), load_average=load_start,
    )
    output.write_text(json.dumps(result, indent=1) + '\n')
    pred_path = output.with_suffix('.predictions.jsonl')
    with pred_path.open('w') as stream:
        for p in predictions:
            stream.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'by_type'},
                     indent=1))


if __name__ == '__main__':
    main()
