"""MASSIVE format probe, round 4: BTZSC-hypothesis phrasing for the 18
scenario options (the style that reproduced agnews/emotiondair exactly).
en-US TRAIN split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common

GLOSS = {
    'alarm': 'alarms',
    'audio': 'audio volume',
    'calendar': 'calendars',
    'cooking': 'cooking',
    'datetime': 'dates and times',
    'email': 'emails',
    'general': 'general conversation',
    'iot': 'IoT and smart home devices',
    'lists': 'lists',
    'music': 'music',
    'news': 'the news',
    'play': 'playing media',
    'qa': 'asking questions',
    'recommendation': 'recommendations',
    'social': 'social media',
    'takeaway': 'takeaway food',
    'transport': 'transport or travels',
    'weather': 'the weather',
}


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    table = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet')
    rows = table.to_pylist()[:n]
    scenarios = common.massive_scenarios()
    from run_julia import build_engine
    engine = build_engine(4)

    templates = [
        'The example utterance is about {g}.',
        'The example utterance is related to {g}.',
        'This example utterance is about {g}.',
        'The example utterance is a query about {g}.',
        'The intent of this example utterance is about {g}.',
    ]
    for t in templates:
        for q in [common.JEV_QUESTION,
                  'Which intent best describes this request?']:
            options = [t.format(g=GLOSS[s]) for s in scenarios]
            correct = 0
            for r in rows:
                req = dict(state={'text': r['utt']}, question=q, type='choice',
                           options=options)
                try:
                    a = engine.predict([req])[0]
                    if a['index'] == r['scenario']:
                        correct += 1
                except Exception:
                    pass
            print(f'{q[:40]!r} + {t[:45]!r}: {correct}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
