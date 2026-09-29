"""Probe MASSIVE request-format candidates on the TRAIN split (never the
reported test slice). The published eval code is not shipped, so the format
is underdetermined; train-split accuracy identifies which reconstruction the
checkpoint was trained for."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'julia-1' / 'repo'))

import common


def rows_for(utt, scenarios, state_mode, question, option_mode):
    if option_mode == 'names':
        options = list(scenarios)
    elif option_mode == 'hypothesis':
        options = [f'This request is about {s}.' for s in scenarios]
    elif option_mode == 'belongs':
        options = [f'This request belongs to the {s} scenario.' for s in scenarios]
    state = {'text': utt} if state_mode == 'dict' else utt
    return dict(state=state, question=question, type='choice', options=options)


def main():
    import pyarrow.parquet as pq
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    table = pq.read_table(common.MASSIVE_DIR / 'en-US-train.parquet')
    rows = table.to_pylist()[:n]
    scenarios = common.massive_scenarios()

    sys.path.insert(0, str(common.ROOT / 'models' / 'julia-1' / 'repo'))
    from run_julia import build_engine
    engine = build_engine(4)

    variants = [
        ('jev-dict+jevq+names', 'dict', common.JEV_QUESTION, 'names'),
        ('raw+jevq+names', 'raw', common.JEV_QUESTION, 'names'),
        ('dict+scenario-req+names', 'dict',
         'Which scenario best describes this request?', 'names'),
        ('raw+scenario-req+names', 'raw',
         'Which scenario best describes this request?', 'names'),
        ('dict+scenario-text+names', 'dict',
         'Which scenario best describes this text?', 'names'),
        ('dict+jevq+hypothesis', 'dict', common.JEV_QUESTION, 'hypothesis'),
        ('dict+scenario-req+belongs', 'dict',
         'Which scenario best describes this request?', 'belongs'),
    ]
    for label, state_mode, question, option_mode in variants:
        correct = 0
        for r in rows:
            gold = scenarios[r['scenario']]
            req = rows_for(r['utt'], scenarios, state_mode, question, option_mode)
            try:
                answer = engine.predict([req])[0]
                if req['options'][answer['index']] == (
                        gold if option_mode == 'names'
                        else req['options'][scenarios.index(gold)]):
                    correct += 1
            except Exception:
                pass
        print(f'{label}: {correct}/{len(rows)} = {correct / len(rows):.3f}',
              flush=True)


if __name__ == '__main__':
    main()
