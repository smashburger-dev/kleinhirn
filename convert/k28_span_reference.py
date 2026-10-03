"""Token-classification spans of the HF pipeline on the tokenizer.json of a model (K28.3b G3).

The goldens of K28.2 and K28.5 come from AutoTokenizer (transformers 5.0.0), whose XLM-R class puts
the start of a word after the leading space; tokenizer.json through the tokenizers library puts it on
the space. This runs the same pipeline (aggregation_strategy "simple", the 200 texts of the golden)
with PreTrainedTokenizerFast built from tokenizer.json, writes the spans to
bench/results/k28-span-reference-<slug>.json and, with --compare FILE, compares them with the spans
the browser text path wrote into a k28-parity ...-text.json result (code point offsets, group, score).

Usage: .venv-k28/bin/python convert/k28_span_reference.py <model-id> [--compare bench/results/k28-parity-<slug>-f32-<commit>-text.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import ROOT, model_dir, repo_dir, slug  # noqa: E402

N_TOKEN_TEXTS = 200


def main():
    model_id = sys.argv[1]
    compare = sys.argv[sys.argv.index("--compare") + 1] if "--compare" in sys.argv else None
    import torch
    from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast, pipeline

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repo = repo_dir(model_id)
    own = model_dir(model_id) / "tokenizer.json"
    tok = PreTrainedTokenizerFast.from_pretrained(str(repo)) if (repo / "tokenizer.json").exists() \
        else PreTrainedTokenizerFast(tokenizer_file=str(own))
    doc = json.loads((ROOT / f"tests/golden/k28/{slug(model_id)}/token-classification.json").read_text())
    corpus = json.loads((ROOT / "tests/corpus/texts1000.json").read_text())
    corpus = corpus["texts"] if isinstance(corpus, dict) else corpus
    model = AutoModelForTokenClassification.from_pretrained(
        str(repo), attn_implementation="eager", dtype=torch.float32, trust_remote_code=False).float().eval()
    tok.model_max_length = doc["lengths"]["model"]
    pipe = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple", device="cpu")
    spans = []
    for item in doc["items"][:N_TOKEN_TEXTS]:
        spans.append([{"entity_group": s["entity_group"], "score": float(s["score"]), "start": int(s["start"]),
                       "end": int(s["end"])} for s in pipe(corpus[item["text_index"]])])
    out = ROOT / f"bench/results/k28-span-reference-{slug(model_id)}.json"
    out.write_text(json.dumps({"model": model_id, "reference": "PreTrainedTokenizerFast from tokenizer.json",
                               "spans": spans}) + "\n")
    print(out, sum(len(s) for s in spans), "spans")
    if compare:
        got = json.loads(Path(compare).read_text())["metrics"]["spansGot"]
        equal = texts_bad = 0
        max_score = 0.0
        first = None
        for i, (g, w) in enumerate(zip(got, spans)):
            ok = len(g) == len(w)
            for a, b in zip(g, w):
                max_score = max(max_score, abs(a["score"] - b["score"]))
                if (a["entity_group"], a["start"], a["end"]) != (b["entity_group"], b["start"], b["end"]) \
                        or abs(a["score"] - b["score"]) > 1e-4:
                    ok = False
            if ok:
                equal += 1
            elif first is None:
                first = {"text_index": i, "got": g, "want": w}
        res = {"texts": len(spans), "equal": equal, "maxAbsScoreDiff": max_score, "firstDiffering": first}
        print(json.dumps(res)[:600])
        (ROOT / f"bench/results/k28-span-compare-{slug(model_id)}.json").write_text(json.dumps(res, indent=1) + "\n")
        if equal != len(spans):
            sys.exit(1)


if __name__ == "__main__":
    main()
