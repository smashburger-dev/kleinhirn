"""Shared task loading for the K9 suite.

Replicates the published protocols documented in docs/BENCHMARKS.md:
- typed-decisions expansion from julia-1/scripts/reproduce_typed.py
- BTZSC sampling from jev-benchmarks @0d610cc src/jev_benchmarks/data.py
- MASSIVE scenario classification over 52 locales (test split)
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'models' / 'k9-data'
BTZSC_REVISION = 'fef2a2ac62b69c58670047dddf045c53d7c3cb5e'
BTZSC_SEED = 20260917
JEV_QUESTION = 'Which single label best describes the input text?'
BTZSC_DATASETS = [('agnews', 'topic'), ('emotiondair', 'emotion'), ('banking77', 'intent')]
TYPED_PARQUET = DATA / 'typed-test.parquet'
TYPED_SHA256 = '4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c'
MASSIVE_DIR = DATA / 'massive'
MASSIVE_LOCALES = [
    'af-ZA', 'am-ET', 'ar-SA', 'az-AZ', 'bn-BD', 'ca-ES', 'cy-GB', 'da-DK', 'de-DE',
    'el-GR', 'en-US', 'es-ES', 'fa-IR', 'fi-FI', 'fr-FR', 'he-IL', 'hi-IN', 'hu-HU',
    'hy-AM', 'id-ID', 'is-IS', 'it-IT', 'ja-JP', 'jv-ID', 'ka-GE', 'km-KH', 'kn-IN',
    'ko-KR', 'lv-LV', 'ml-IN', 'mn-MN', 'ms-MY', 'my-MM', 'nb-NO', 'nl-NL', 'pl-PL',
    'pt-PT', 'ro-RO', 'ru-RU', 'sl-SL', 'sq-AL', 'sv-SE', 'sw-KE', 'ta-IN', 'te-IN',
    'th-TH', 'tl-PH', 'tr-TR', 'ur-PK', 'vi-VN', 'zh-CN', 'zh-TW',
]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_typed(path=TYPED_PARQUET):
    """Expand the pinned 400 cases into 2,000 requests, as reproduce_typed.py."""
    import pyarrow.parquet as pq
    if digest(path) != TYPED_SHA256:
        raise ValueError('Typed-decisions parquet does not match the pinned hash')
    rows, metadata = [], []
    for case in pq.read_table(path).to_pylist():
        state = json.loads(case['state'])
        gold = json.loads(case['gold'])
        for question_id, question in json.loads(case['questions']).items():
            kind = question['type']
            criteria = question.get('criteria')
            if criteria is None and kind == 'noul':
                criteria = {'false': 'false', 'true': 'true'}
            if isinstance(criteria, list):
                criteria = {str(i): value for i, value in enumerate(criteria)}
            if not isinstance(criteria, dict):
                raise ValueError('Missing option descriptions')
            keys = ['false', 'true'] if kind == 'noul' else list(criteria)
            rows.append(dict(state=state, question=question['instructions'],
                             type=kind, options=[criteria[key] for key in keys]))
            metadata.append(dict(id=case['id'] + ':' + question_id, type=kind,
                                 keys=keys, gold=str(gold[question_id]['label'])))
    return rows, metadata


def _class_count(texts):
    first = texts[0]
    for index in range(1, len(texts)):
        if texts[index] != first:
            return index
    raise ValueError('could not infer class count from repeated BTZSC texts')


def _balanced_indices(targets, limit, seed):
    by_class = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen, classes = [], sorted(by_class)
    while len(chosen) < min(limit, len(targets)):
        progress = False
        for class_id in classes:
            if by_class[class_id] and len(chosen) < limit:
                chosen.append(by_class[class_id].pop())
                progress = True
        if not progress:
            break
    return sorted(chosen)


def load_btzsc(name, samples=100):
    """Jev pilot sampling: test split, single-positive rows, balanced 100."""
    from datasets import load_dataset
    offset = [n for n, _ in BTZSC_DATASETS].index(name)
    rows = load_dataset('btzsc/btzsc', name=name, split='test',
                        revision=BTZSC_REVISION,
                        cache_dir=str(DATA / 'cache'))
    binary = [int(v) for v in rows['labels']]
    texts = [str(v) for v in rows['text']]
    n_classes = _class_count(texts)
    total = len(rows) // n_classes
    labels = tuple(str(rows[i]['hypothesis']) for i in range(n_classes))
    valid, targets = [], []
    for s in range(total):
        values = binary[s * n_classes:(s + 1) * n_classes]
        if sum(values) == 1:
            valid.append(s)
            targets.append(values.index(1))
    selected = _balanced_indices(targets, samples, BTZSC_SEED + offset)
    return [dict(example_id=f'{name}:{valid[p]}', text=texts[valid[p] * n_classes],
                 labels=labels, target_index=targets[p]) for p in selected]


def classify_row(example):
    """Jev request mapped to the local row interface. State is stored
    pre-serialized (Python json.dumps separators) so browser and PyTorch
    tokenize byte-identical text."""
    return dict(state=json.dumps({'text': example['text']}, ensure_ascii=False),
                question=JEV_QUESTION, type='choice',
                options=list(example['labels']))


def massive_scenarios():
    """18 canonical scenario names from the parquet ClassLabel metadata."""
    import pyarrow.parquet as pq
    table = pq.read_table(MASSIVE_DIR / 'en-US-test.parquet')
    info = json.loads(table.schema.metadata[b'huggingface'].decode())
    return info['info']['features']['scenario']['names']


def load_massive(locale, scenarios):
    import pyarrow.parquet as pq
    table = pq.read_table(MASSIVE_DIR / f'{locale}-test.parquet')
    return [dict(example_id=f'{locale}:{row["id"]}', utt=row['utt'],
                 scenario=scenarios[row['scenario']]) for row in table.to_pylist()]


def massive_row(example, scenarios):
    return dict(state=json.dumps({'text': example['utt']}, ensure_ascii=False),
                question=JEV_QUESTION, type='choice', options=list(scenarios))


TASK_DIR = DATA / 'tasks'


def _read_jsonl(path):
    import json as _json
    with Path(path).open() as stream:
        return [_json.loads(line) for line in stream if line.strip()]


def load_task_pairs(task, locales=None, limit=None):
    """Read a frozen manifest (export_tasks.py) into (row, meta) pairs."""
    if task == 'typed':
        entries = _read_jsonl(TASK_DIR / 'typed.jsonl')
        pairs = [(e['row'], e['meta']) for e in entries]
    elif task in ('agnews', 'emotiondair', 'banking77'):
        examples = _read_jsonl(TASK_DIR / f'{task}.jsonl')
        pairs = [(classify_row(e),
                  dict(id=e['example_id'], keys=list(e['labels']),
                       gold=e['labels'][e['target_index']]))
                 for e in examples]
    elif task == 'massive':
        import json as _json
        scenarios = _json.loads(
            (TASK_DIR / 'massive-scenarios.json').read_text())
        pairs = []
        for locale in (locales or MASSIVE_LOCALES):
            for e in _read_jsonl(TASK_DIR / f'massive-{locale}.jsonl'):
                pairs.append((massive_row(e, scenarios),
                              dict(id=e['example_id'], keys=scenarios,
                                   gold=e['scenario'])))
    else:
        raise ValueError(f'unknown task {task}')
    return pairs[:limit] if limit else pairs
