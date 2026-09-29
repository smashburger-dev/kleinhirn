"""Export frozen K9 task manifests to models/k9-data/tasks/*.jsonl.
Run once under .venv-julia (needs datasets and pyarrow). All later runners
read these manifests instead of re-sampling."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main():
    out = common.DATA / 'tasks'
    out.mkdir(parents=True, exist_ok=True)

    rows, metadata = common.load_typed()
    with (out / 'typed.jsonl').open('w') as stream:
        for row, meta in zip(rows, metadata):
            # Pre-serialize dict state the way julia/data.py does so the
            # browser tokenizer sees the identical text as PyTorch.
            row = dict(row, state=json.dumps(row['state'], ensure_ascii=False))
            stream.write(json.dumps(dict(row=row, meta=meta),
                                    ensure_ascii=False) + '\n')
    print('typed:', len(rows))

    for name, _ in common.BTZSC_DATASETS:
        examples = common.load_btzsc(name)
        with (out / f'{name}.jsonl').open('w') as stream:
            for e in examples:
                stream.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(name, ':', len(examples))

    scenarios = common.massive_scenarios()
    (out / 'massive-scenarios.json').write_text(json.dumps(scenarios) + '\n')
    for locale in common.MASSIVE_LOCALES:
        examples = common.load_massive(locale, scenarios)
        with (out / f'massive-{locale}.jsonl').open('w') as stream:
            for e in examples:
                stream.write(json.dumps(e, ensure_ascii=False) + '\n')
    print('massive: 52 locales x', len(examples))


def read_manifest(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == '__main__':
    main()
