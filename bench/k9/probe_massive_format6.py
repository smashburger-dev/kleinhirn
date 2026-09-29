"""MASSIVE format probe, round 6: full-sentence scenario descriptions written
from the member intents, JEV question, dict/raw state. en-US TRAIN only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common

FULL = {
    'alarm': 'Setting, checking, or removing alarms',
    'audio': 'Adjusting audio volume or playback output',
    'calendar': 'Checking, adding, or removing calendar events',
    'cooking': 'Recipes and cooking instructions',
    'datetime': 'Checking the date or time, or converting between time zones',
    'email': 'Sending, reading, or managing emails and contacts',
    'general': 'Greetings, jokes, or other general conversation',
    'iot': 'Controlling smart home devices and appliances',
    'lists': 'Creating, checking, or removing list items',
    'music': 'Playing or controlling music and expressing music preferences',
    'news': 'Getting news or current events',
    'play': 'Playing audiobooks, games, podcasts, or radio',
    'qa': 'Asking factual questions or requesting information',
    'recommendation': 'Asking for recommendations for movies, events, or places',
    'social': 'Posting or checking social media',
    'takeaway': 'Ordering takeaway food or checking takeaway options',
    'transport': 'Checking transport, booking taxis or tickets, or traffic',
    'weather': 'Checking the weather',
}


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rows = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet').to_pylist()[:n]
    scenarios = common.massive_scenarios()
    from run_julia import build_engine
    engine = build_engine(4)

    options = [FULL[s] for s in scenarios]
    for label, sf, q in [
        ('dict+jevq', lambda u: {'text': u}, common.JEV_QUESTION),
        ('raw+jevq', lambda u: u, common.JEV_QUESTION),
        ('dict+scenario', lambda u: {'text': u},
         'Which scenario best describes this request?'),
        ('raw+scenario', lambda u: u,
         'Which scenario best describes this request?'),
        ('dict+intent', lambda u: {'text': u},
         'Which intent best describes this request?'),
    ]:
        correct = 0
        for r in rows:
            req = dict(state=sf(r['utt']), question=q, type='choice',
                       options=options)
            try:
                a = engine.predict([req])[0]
                if a['index'] == r['scenario']:
                    correct += 1
            except Exception:
                pass
        print(f'{label}: {correct}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
