"""MASSIVE format probe, round 5: scenario options built from the dataset's
own intent membership (scenario name plus its intent names), richer glosses,
and JEV question. en-US TRAIN split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    table = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet')
    info = __import__('json').loads(
        table.schema.metadata[b'huggingface'].decode())
    intent_names = info['info']['features']['intent']['names']
    rows = table.to_pylist()[:n]
    scenarios = common.massive_scenarios()
    members = {s: [] for s in scenarios}
    full = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet').to_pylist()
    for r in full:
        s = scenarios[r['scenario']]
        i = intent_names[r['intent']]
        if i not in members[s]:
            members[s].append(i)
    from run_julia import build_engine
    engine = build_engine(4)

    variants = [
        ('names+intents',
         lambda s: f'{s}: ' + ', '.join(members[s])),
        ('intents-only',
         lambda s: ', '.join(members[s])),
        ('names+natural-intents',
         lambda s: f'{s}: ' + ', '.join(i.replace('_', ' ') for i in members[s])),
        ('natural-intents-only',
         lambda s: ', '.join(i.replace('_', ' ') for i in members[s])),
        ('sentence-intents',
         lambda s: f'This request is about {s} (' + ', '.join(
             i.replace('_', ' ') for i in members[s]) + ').'),
    ]
    for label, of in variants:
        options = [of(s) for s in scenarios]
        correct = 0
        for r in rows:
            req = dict(state={'text': r['utt']}, question=common.JEV_QUESTION,
                       type='choice', options=options)
            try:
                a = engine.predict([req])[0]
                if a['index'] == r['scenario']:
                    correct += 1
            except Exception:
                pass
        print(f'{label}: {correct}/{len(rows)}  sample: {options[0][:80]!r}',
              flush=True)


if __name__ == '__main__':
    main()
