"""K17 judge parity golden: torch logits for the TypeScript/numpy gates.

Every reachable state of the 30 hard fixture problems plus 1,000 states
sampled from fresh 5- and 6-number problems (10 each, random.Random(42);
sampling random.Random(7)). For each item the raw model logit (pre-sigmoid)
is stored; the numpy spec and the WGSL graph are gated against it.

Also dumps num_feats vectors for every distinct (x, target) pair so the TS
feature function can be compared bit for bit.

Usage:
  .venv/bin/python convert/judge_golden.py <upstream-checkout>

Writes tests/golden/search/judge-parity.json.
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, sys.argv[1])
from interference_search.countdown import gen_problem, reachable_states  # noqa: E402
from interference_search.judge import encode, load_judge, num_feats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/golden/search/countdown-fixture.json"
OUT = ROOT / "tests/golden/search/judge-parity.json"
WEIGHTS = ROOT / "models/countdown-judge/countdown_judge.pt"


def main() -> None:
    fx = json.loads(FIXTURE.read_text())
    items: list[dict] = []
    for p in fx["hard_problems"]:
        items += [{"s": st["s"], "t": p["target"]} for st in p["states"]]

    rng = random.Random(42)
    sample_rng = random.Random(7)
    for n in (5, 6):
        pool: list[dict] = []
        for _ in range(10):
            p = gen_problem(rng, n_numbers=n)
            pool += [{"s": list(s), "t": p["target"]}
                     for s in reachable_states(p["numbers"])]
        items += sample_rng.sample(pool, min(500, len(pool)))

    model = load_judge(str(WEIGHTS)).model
    model64 = load_judge(str(WEIGHTS)).model.double().eval()  # exact ref
    feats_seen: dict[tuple, dict] = {}
    with torch.no_grad():
        for start in range(0, len(items), 256):
            chunk = items[start:start + 256]
            states = [c["s"] for c in chunk]
            X, M = encode(states, [c["t"] for c in chunk])
            logits = model(X, M).tolist()
            logits64 = model64(X.double(), M).tolist()
            for c, lg, lg64 in zip(chunk, logits, logits64):
                c["logit"] = lg
                c["logit64"] = lg64
                for x in c["s"]:
                    pair = (x, c["t"])
                    if pair not in feats_seen:
                        feats_seen[pair] = {"x": x, "t": c["t"],
                                            "f": num_feats(x, c["t"])}

    OUT.write_text(json.dumps({
        "source": "countdown_judge.pt @ Badtheorylabs/interference-search",
        "generator": "convert/judge_golden.py",
        "items": items,
        "feats": sorted(feats_seen.values(), key=lambda d: (d["t"], d["x"])),
    }))
    print(f"wrote {OUT}: {len(items)} state logits, "
          f"{len(feats_seen)} feature vectors")


if __name__ == "__main__":
    main()
