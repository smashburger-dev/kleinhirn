"""Decisions of a K28 pilot with the tokenization of tokenizer.json (K28.3b G3).

The goldens of K28.5 tokenize with AutoTokenizer (transformers 5.0.0), which differs from
tokenizer.json for XLM-R: pairs get one </s> between the texts instead of two, Precompiled is not
applied. The browser text path follows tokenizer.json. This runs PyTorch fp32 on the ids that
tokenizer.json gives (tokenizers library, truncation and padding set by the call) and writes the
decision per case (argmax, zero-shot choice, best passage) to
bench/results/k28-text-reference-<slug>.json; with --compare FILE it compares them with the
decisions the text path wrote into a k28-parity ...-text.json result.

Usage: .venv-k28/bin/python convert/k28_text_reference.py <model-id> [--compare bench/results/k28-parity-<slug>-f32-<commit>-text.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import ROOT, model_dir, repo_dir, slug  # noqa: E402
from k28_golden import HYPOTHESIS, NLI_LABELS, corpus_texts, load_classifier  # noqa: E402


def main():
    model_id = sys.argv[1]
    compare = sys.argv[sys.argv.index("--compare") + 1] if "--compare" in sys.argv else None
    from tokenizers import Tokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    golden_dir = ROOT / f"tests/golden/k28/{slug(model_id)}"
    task = next(t for t in ("sequence-classification", "nli", "reranking") if (golden_dir / f"{t}.json").exists())
    doc = json.loads((golden_dir / f"{task}.json").read_text())
    L = doc["lengths"]["model"]
    repo_tj = repo_dir(model_id) / "tokenizer.json"
    tk = Tokenizer.from_file(str(repo_tj if repo_tj.exists() else model_dir(model_id) / "tokenizer.json"))
    tk.no_padding()
    model = load_classifier(model_id, "sequence")
    texts = corpus_texts()

    def logits(first, second=None, strategy="longest_first"):
        tk.enable_truncation(max_length=L, strategy=strategy)
        e = tk.encode(first, second)
        ids = torch.tensor([e.ids])
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if model.config.model_type != "distilbert":
            kwargs["token_type_ids"] = torch.tensor([e.type_ids])
        with torch.no_grad():
            return model(**kwargs).logits[0].numpy()

    choices = []
    if task == "sequence-classification":
        for item in doc["items"]:
            text = item["text"] if "text" in item else texts[item["text_index"]]
            choices.append(int(np.argmax(logits(text))))
    elif task == "nli":
        ent = doc["entail_index"]
        for t in doc["texts"]:
            ent_logits = [logits(texts[t["text_index"]], HYPOTHESIS.format(lab), "only_first")[ent] for lab in NLI_LABELS]
            choices.append(int(np.argmax(ent_logits)))
    else:
        for q in doc["queries"]:
            rows = [it for it in doc["items"] if it["query"] == q["query"]]
            scores = [logits(texts[r["text_index"]], texts[r["passage_text_index"]])[0] for r in rows]
            choices.append(int(np.argmax(scores)))
    out = ROOT / f"bench/results/k28-text-reference-{slug(model_id)}.json"
    out.write_text(json.dumps({"model": model_id, "task": task, "reference": "tokenizer.json through tokenizers",
                               "choices": choices}) + "\n")
    print(out, task, len(choices), "decisions")
    if compare:
        got = json.loads(Path(compare).read_text())["metrics"]["choices"]
        equal = sum(1 for a, b in zip(got, choices) if a == b)
        golden_equal = None
        res = {"task": task, "cases": len(choices), "equal": equal}
        print(json.dumps(res))
        (ROOT / f"bench/results/k28-text-compare-{slug(model_id)}.json").write_text(json.dumps(res, indent=1) + "\n")
        if equal != len(choices):
            sys.exit(1)


if __name__ == "__main__":
    main()
