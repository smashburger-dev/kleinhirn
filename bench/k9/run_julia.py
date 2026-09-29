"""K9 gate runner: Julia-1 on PyTorch CPU, one request per forward.

Usage:
  .venv-julia/bin/python bench/k9/run_julia.py --task typed
  .venv-julia/bin/python bench/k9/run_julia.py --task agnews --output out.json
  .venv-julia/bin/python bench/k9/run_julia.py --task massive --locales en-US de-DE
"""
import argparse
import collections
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common

CHECKPOINT = common.ROOT / 'models' / 'julia-1' / 'repo'
WEIGHTS_SHA256 = 'df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72'

GATES = {
    'typed': dict(correct=1463, total=2000, accuracy=0.7315),
    'agnews': dict(correct=94, total=100, accuracy=0.94),
    'emotiondair': dict(correct=86, total=100, accuracy=0.86),
    'banking77': dict(correct=64, total=100, accuracy=0.64),
    'massive': dict(correct=110573, total=154648, accuracy=0.7149979307847499),
}


class PyReducer:
    """Pure-Python stand-in for the Bend reducer (argmax only, as route_many uses)."""
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def argmax(scores):
        best, index = -math.inf, 0
        for i, score in enumerate(scores):
            if score > best:
                best, index = score, i
        return index


def build_engine(threads):
    import torch
    torch.set_num_threads(threads)
    from julia.router.engine import FastEngine
    return FastEngine(str(CHECKPOINT), device='cpu', transformer_backend='torch',
                      strict_encoding=True, max_length=1024, head_length=512,
                      batch_size=16, marker_only_head=False)


def load_average():
    out = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
    return out.split('load average')[-1].strip()


def evaluate_single(engine, pairs):
    """One example per forward, matching the published typed-decisions protocol."""
    predictions, stats = [], collections.Counter()
    t0 = time.monotonic()
    for i, (row, meta) in enumerate(pairs):
        started = time.perf_counter()
        try:
            answer = engine.predict([row])[0]
            status = 'ok'
            index = answer['index']
            probs = answer['probabilities']
        except Exception:
            status, index, probs = 'abstained', None, None
        latency = time.perf_counter() - started
        predicted = meta['keys'][index] if index is not None else None
        correct = predicted == meta['gold']
        stats['correct' if correct else 'wrong'] += 1
        predictions.append(dict(id=meta['id'], type=meta.get('type', 'choice'),
                                gold=meta['gold'], predicted=predicted,
                                correct=correct, status=status,
                                probabilities=probs,
                                latency_ms=round(latency * 1000, 2)))
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(pairs)} correct={stats["correct"]}', flush=True)
    return predictions, stats, time.monotonic() - t0


def evaluate_routed(engine, pairs, width, survivors):
    """Banking77: Router groups of `width`, `survivors` per group, final <=16 call."""
    import julia.router.router as rr
    rr.BendReducer = PyReducer  # Bend lib is Linux-only; Router needs argmax only
    router = rr.Router(engine, width=width, survivors=survivors)
    predictions, stats = [], collections.Counter()
    t0 = time.monotonic()
    for row, meta in pairs:
        started = time.perf_counter()
        try:
            result = router.route(row)
            status = 'ok'
            index = result.index
            candidates = list(result.candidates)
        except Exception:
            status, index, candidates = 'abstained', None, None
        latency = time.perf_counter() - started
        predicted = meta['keys'][index] if index is not None else None
        correct = predicted == meta['gold']
        stats['correct' if correct else 'wrong'] += 1
        predictions.append(dict(id=meta['id'], type='choice', gold=meta['gold'],
                                predicted=predicted, correct=correct, status=status,
                                candidates=candidates,
                                latency_ms=round(latency * 1000, 2)))
    return predictions, stats, time.monotonic() - t0


def evaluate_noul_shortlist(engine, pairs, top_k):
    """Banking77 shortlist reconstruction: rank all labels by noul p(true),
    keep the top-k, then one native choice call over the shortlist."""
    predictions, stats = [], collections.Counter()
    t0 = time.monotonic()
    for row, meta in pairs:
        started = time.perf_counter()
        try:
            rank_rows = [dict(state=row['state'], question=label, type='noul',
                              options=['false', 'true']) for label in row['options']]
            answers = engine.predict(rank_rows)
            scored = sorted(range(len(answers)),
                            key=lambda i: answers[i]['probabilities'][1],
                            reverse=True)
            shortlist = sorted(scored[:top_k])
            final = engine.predict([dict(row, options=[row['options'][i]
                                                       for i in shortlist])])[0]
            status, index = 'ok', shortlist[final['index']]
        except Exception:
            status, index = 'abstained', None
        latency = time.perf_counter() - started
        predicted = meta['keys'][index] if index is not None else None
        correct = predicted == meta['gold']
        stats['correct' if correct else 'wrong'] += 1
        predictions.append(dict(id=meta['id'], type='choice', gold=meta['gold'],
                                predicted=predicted, correct=correct, status=status,
                                latency_ms=round(latency * 1000, 2)))
    return predictions, stats, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['typed', 'agnews', 'emotiondair', 'banking77', 'massive'])
    ap.add_argument('--output', type=Path)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--locales', nargs='*')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--width', type=int, default=20)
    ap.add_argument('--survivors', type=int, default=4)
    ap.add_argument('--shortlist', choices=['router', 'noul'], default='router')
    ap.add_argument('--top-k', type=int, default=16)
    args = ap.parse_args()

    output = args.output or (common.ROOT / 'bench' / 'results' /
                             f'k9-{args.task}-julia-pytorch.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    load_start = load_average()
    print('building engine...', flush=True)
    engine = build_engine(args.threads)

    if args.task == 'typed':
        pairs = common.load_task_pairs('typed', limit=args.limit)
        predictions, stats, elapsed = evaluate_single(engine, pairs)
        by_type = collections.defaultdict(lambda: dict(count=0, correct=0))
        for p in predictions:
            by_type[p['type']]['count'] += 1
            by_type[p['type']]['correct'] += int(p['correct'])
        extra = dict(by_type={k: dict(v, accuracy=v['correct'] / v['count'])
                              for k, v in by_type.items()})
    elif args.task == 'massive':
        locales = args.locales or common.MASSIVE_LOCALES
        predictions, stats = [], collections.Counter()
        per_locale = {}
        t0 = time.monotonic()
        for locale in locales:
            pairs = common.load_task_pairs('massive', locales=[locale],
                                           limit=args.limit)
            preds, st, _ = evaluate_single(engine, pairs)
            predictions.extend(preds)
            stats.update(st)
            n = len(preds)
            c = sum(1 for p in preds if p['correct'])
            per_locale[locale] = dict(count=n, correct=c, accuracy=c / n)
            print(f'{locale}: {c}/{n} = {c / n:.4f}', flush=True)
        elapsed = time.monotonic() - t0
        macro = sum(v['accuracy'] for v in per_locale.values()) / len(per_locale)
        extra = dict(per_locale=per_locale, macro_accuracy=macro)
    else:
        pairs = common.load_task_pairs(args.task, limit=args.limit)
        if args.task == 'banking77':
            if args.shortlist == 'noul':
                predictions, stats, elapsed = evaluate_noul_shortlist(
                    engine, pairs, args.top_k)
            else:
                predictions, stats, elapsed = evaluate_routed(
                    engine, pairs, args.width, args.survivors)
        else:
            predictions, stats, elapsed = evaluate_single(engine, pairs)
        extra = dict(seed=common.BTZSC_SEED, dataset_revision=common.BTZSC_REVISION)

    total = len(predictions)
    correct = stats['correct']
    gate = GATES[args.task]
    result = dict(
        task=args.task, route='julia-pytorch-cpu', model='Julia-1',
        weights_sha256=WEIGHTS_SHA256,
        engine=dict(device='cpu', transformer_backend='torch', strict_encoding=True,
                    max_length=1024, head_length=512, batch_size=16,
                    marker_only_head=False, threads=args.threads,
                    router=(dict(width=args.width, survivors=args.survivors)
                            if args.task == 'banking77' else None)),
        total=total, correct=correct, accuracy=correct / total if total else 0,
        abstained=sum(1 for p in predictions if p['status'] == 'abstained'),
        gate=dict(published=gate['accuracy'], published_counts=gate,
                  tolerance_pp=1.0,
                  deviation_pp=round((correct / total - gate['accuracy']) * 100, 3)
                  if total == gate['total'] else None),
        elapsed_seconds=round(elapsed, 1),
        load_average=load_start,
        **extra,
    )
    output.write_text(json.dumps(result, indent=1) + '\n')
    pred_path = output.with_suffix('.predictions.jsonl')
    with pred_path.open('w') as stream:
        for p in predictions:
            stream.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_locale'},
                     indent=1))


if __name__ == '__main__':
    main()
