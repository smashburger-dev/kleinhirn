"""Close decisions of a K28 parity run (K28_DESIGN section 10; rules of 30.09. and, replacing the f16 bound, 01.10.).

f16 (decision 3 of docs/K28_S3_WORKORDER.md): a decision (argmax, zero-shot choice, best passage, token argmax)
is close when the golden gap between the best and the second best value is below twice the largest f16 logit
deviation of the run, delta_run. Only a close decision can turn over under f16 rounding. The f16 gate counts
the other decisions, and only when the run is evaluable: delta_run <= 3 * delta_sim, where delta_sim comes
from the independent simulation (convert/k28_f16_sim.py, bench/results/k28-f16-sim/<slug>.json). The output
holds deltaSim and evaluable; without a simulation file evaluable is null.

f32 (decision 4): only for a model with a float64 reference (bench/results/k28-f64-reference-<slug>.json with
`gaps`). The f32 decisions that differ from the fp32 golden are allowed when their torch64 gap is below twice
golden32_vs_torch64, the error of the fp32 golden; the others are not. Output `decisions32` per decision:
n, deviations, allowed, notAllowed. The newest f32 parity file of the model is used.

Reads bench/results/k28-parity-<slug>-f16-<commit>.json and the golden of the model, prints one JSON line
and appends it to bench/results/k28-close-calls-<tag>.json when --out is given. Embeddings have no decision.
Zero-shot, best passage and token argmax: result files written since K28.S hold the indices of the
differing decisions (zeroShotDisagreements, bestPassageDisagreements, argmaxPerTokenDisagreements) and
failuresNotClose is exact. Older files hold only the rate; then failures are counted from the rate and
failuresNotCloseAtLeast is a lower bound.

Usage: .venv-k28/bin/python convert/k28_close_calls.py <result-f16.json>... [--out bench/results/k28-close-calls-k28.6.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import ROOT  # noqa: E402
import k28_forward as F  # noqa: E402


def gap(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    return float(v[-1] - v[-2])


def latest_f32(slug: str):
    """Newest f32 parity result of a model (the run id ends with the epoch milliseconds)."""
    best = None
    for p in (ROOT / "bench/results").glob(f"k28-parity-{slug}-f32-*.json"):
        if p.name.endswith("-text.json") or p.name.endswith("-layers.json"):
            continue
        doc = json.loads(p.read_text())
        t = int(str(doc.get("run_id", "0")).split("-")[-1])
        if best is None or t > best[0]:
            best = (t, doc)
    return best[1] if best else None


# decision name -> (key of the disagreement indices in the parity metrics, rate key, count key)
DECISIONS32 = {
    "argmax": ("disagreements", "argmaxAgreement", None),
    "zeroShot": ("zeroShotDisagreements", "zeroShotAgreement", "zeroShotTexts"),
    "bestPassage": ("bestPassageDisagreements", "bestPassageAgreement", "queries"),
    "tokenArgmax": ("argmaxPerTokenDisagreements", "argmaxPerTokenAgreement", "tokens"),
}


def analyse32(slug: str):
    """f32 decisions against torch64: which deviations from the golden are inside the golden's own error."""
    ref_path = ROOT / f"bench/results/k28-f64-reference-{slug}.json"
    res = latest_f32(slug)
    if not ref_path.exists() or res is None:
        return None
    return decisions32(json.loads(ref_path.read_text()), res["metrics"])


def decisions32(ref: dict, m: dict) -> dict:
    """ref: the float64 reference with gaps; m: metrics of the f32 parity run."""
    if "gaps" not in ref:
        return {"error": "float64 reference without gaps; rerun convert/k28_f64_reference.py"}
    limit = 2 * ref["golden32_vs_torch64"]
    out = {"limit": limit, "decisions": {}}
    for name, gaps in ref["gaps"].items():
        key = DECISIONS32[name][0]
        idx = m.get(key)
        if idx is None:
            out["decisions"][name] = {"n": len(gaps), "error": f"parity result lacks {key}"}
            continue
        allowed = [i for i in idx if gaps[i] < limit]
        out["decisions"][name] = {"n": len(gaps), "deviations": len(idx), "allowed": len(allowed),
                                  "notAllowed": len(idx) - len(allowed)}
    return out


def analyse(path: Path) -> dict:
    res = json.loads(path.read_text())
    slug, task, m = res["model"], res["task"], res["metrics"]
    doc = json.loads((ROOT / f"tests/golden/k28/{slug}/{task}.json").read_text())
    dev = m.get("maxAbsLogitDiff", m.get("maxAbsDiffFinal", 0.0))
    limit = 2 * dev
    out = {"model": slug, "task": task, "maxAbsLogitDiff": dev, "closeLimit": limit, "decisions": {}}
    sim_path = ROOT / f"bench/results/k28-f16-sim/{slug}.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())["delta_sim"]
        out.update(deltaSim=sim, evaluable=bool(dev <= 3 * sim))
    else:
        out.update(deltaSim=None, evaluable=None)
    d32 = analyse32(slug)
    if d32 is not None:
        out["decisions32"] = d32

    def rec(name, gaps, failures, indices=None):
        close = [i for i, g in enumerate(gaps) if g < limit]
        entry = {"n": len(gaps), "close": len(close), "notClose": len(gaps) - len(close), "failures": failures}
        if indices is not None:
            entry["failuresNotClose"] = len([i for i in indices if gaps[i] >= limit])
        else:
            entry["failuresNotCloseAtLeast"] = max(0, failures - len(close))
        out["decisions"][name] = entry

    if task in ("sequence-classification", "nli"):
        items = doc["items"]
        rec("argmax", [gap(i["logits"]) if len(i["logits"]) > 1 else float("inf") for i in items],
            len(m["disagreements"]), m["disagreements"])
    if task == "nli":
        rate, n = m["zeroShotAgreement"], m["zeroShotTexts"]
        idx = m.get("zeroShotDisagreements")
        rec("zeroShot", [gap(t["entail_logits"]) for t in doc["texts"]],
            len(idx) if idx is not None else round((1 - rate) * n), idx)
    if task == "reranking":
        idx = m.get("bestPassageDisagreements")
        rec("bestPassage", [gap(q["scores"]) for q in doc["queries"]],
            len(idx) if idx is not None else round((1 - m["bestPassageAgreement"]) * m["queries"]), idx)
    if task == "token-classification":
        gold = F.Goldens(slug.replace("__", "/", 1), task)
        gaps = []
        for item in doc["items"]:
            lg = gold.array(item["logits"])
            s = np.sort(lg, axis=-1)
            gaps.extend((s[:, -1] - s[:, -2]).tolist())
        idx = m.get("argmaxPerTokenDisagreements")
        rec("tokenArgmax", gaps,
            len(idx) if idx is not None else round((1 - m["argmaxPerTokenAgreement"]) * m["tokens"]), idx)
    return out


def main():
    args = sys.argv[1:]
    out = args.pop(args.index("--out") + 1) if "--out" in args else None
    paths = [Path(a) for a in args if a != "--out"]
    rows = [analyse(p) for p in paths]
    for r in rows:
        print(json.dumps(r))
    if out:
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
