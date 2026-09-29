"""MASSIVE format probe, round 3: annot_utt (slot-annotated) state, locale in
state, and descriptive option texts. en-US TRAIN split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common

DESCRIPTIONS = {
    'alarm': 'Alarms, timers and wake-up requests',
    'audio': 'Audio playback and volume control',
    'calendar': 'Calendar events and scheduling',
    'cooking': 'Recipes and cooking',
    'datetime': 'Dates, times and time conversion',
    'email': 'Email and contacts',
    'general': 'Greetings, jokes and general conversation',
    'iot': 'Smart home and connected device control',
    'lists': 'Lists, notes and reminders',
    'music': 'Music and songs',
    'news': 'News and headlines',
    'play': 'Playing media, games, audiobooks and podcasts',
    'qa': 'Questions and knowledge queries',
    'recommendation': 'Recommendations for places, movies and events',
    'social': 'Social media and posting',
    'takeaway': 'Ordering food and takeaway',
    'transport': 'Transport, traffic and taxis',
    'weather': 'Weather forecasts',
}
GLOSS = {s: f'This request is about {d.lower()}.' for s, d in DESCRIPTIONS.items()}


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    table = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet')
    rows = table.to_pylist()[:n]
    scenarios = common.massive_scenarios()
    from run_julia import build_engine
    engine = build_engine(4)

    variants = [
        # (label, state_fn, question, options_fn)
        ('annot-dict+jevq+names',
         lambda r: {'text': r['annot_utt']}, common.JEV_QUESTION,
         lambda: list(scenarios)),
        ('annot-raw+scenario-req+names',
         lambda r: r['annot_utt'], 'Which scenario best describes this request?',
         lambda: list(scenarios)),
        ('locale-dict+jevq+names',
         lambda r: {'locale': r['locale'], 'text': r['utt']}, common.JEV_QUESTION,
         lambda: list(scenarios)),
        ('dict+scenario-req+desc',
         lambda r: {'text': r['utt']}, 'Which scenario best describes this request?',
         lambda: [DESCRIPTIONS[s] for s in scenarios]),
        ('dict+jevq+desc',
         lambda r: {'text': r['utt']}, common.JEV_QUESTION,
         lambda: [DESCRIPTIONS[s] for s in scenarios]),
        ('dict+jevq+gloss',
         lambda r: {'text': r['utt']}, common.JEV_QUESTION,
         lambda: [GLOSS[s] for s in scenarios]),
        ('dict+scenario-req+gloss',
         lambda r: {'text': r['utt']}, 'Which scenario best describes this request?',
         lambda: [GLOSS[s] for s in scenarios]),
        ('raw+jevq+desc',
         lambda r: r['utt'], common.JEV_QUESTION,
         lambda: [DESCRIPTIONS[s] for s in scenarios]),
    ]
    for label, sf, q, of in variants:
        options = of()
        correct = 0
        for r in rows:
            gold = scenarios[r['scenario']]
            req = dict(state=sf(r), question=q, type='choice', options=options)
            try:
                a = engine.predict([req])[0]
                if a['index'] == r['scenario']:
                    correct += 1
            except Exception:
                pass
        print(f'{label}: {correct}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
