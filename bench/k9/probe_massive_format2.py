"""MASSIVE format probe, round 2: more question/state/option shapes plus a
noul-ranking hypothesis, all on the en-US TRAIN split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    table = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet')
    rows = table.to_pylist()[:n]
    scenarios = common.massive_scenarios()
    from run_julia import build_engine
    engine = build_engine(4)

    def run(reqs, gold_idx):
        try:
            a = engine.predict(reqs)
            return a
        except Exception:
            return None

    questions = [
        'Which scenario best describes the input text?',
        'What is the scenario of this request?',
        'What kind of request is this?',
        'Which of the following scenarios does this request belong to?',
        'What does the user want?',
    ]
    states = {
        'dict': lambda u: {'text': u},
        'raw': lambda u: u,
        'utt-dict': lambda u: {'utterance': u},
        'req-dict': lambda u: {'request': u},
    }
    for q in questions:
        for sname, sf in states.items():
            correct = 0
            for r in rows:
                gold = scenarios[r['scenario']]
                req = dict(state=sf(r['utt']), question=q, type='choice',
                           options=list(scenarios))
                a = run([req], r['scenario'])
                if a and scenarios[a[0]['index']] == gold:
                    correct += 1
            print(f'{sname} + {q[:50]!r}: {correct}/{len(rows)}', flush=True)

    # noul ranking over the 18 scenarios, hypothesis statement as question
    correct = 0
    for r in rows:
        gold = scenarios[r['scenario']]
        reqs = [dict(state={'text': r['utt']},
                     question=f'This request is about {s}.', type='noul',
                     options=['false', 'true']) for s in scenarios]
        a = run(reqs, r['scenario'])
        if a:
            best = max(range(len(a)), key=lambda i: a[i]['probabilities'][1])
            if scenarios[best] == gold:
                correct += 1
    print(f'noul-rank hypothesis: {correct}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
