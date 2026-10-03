"""float64 PyTorch check of one K28 model against the numpy forward and the fp32 goldens (K28.6 step 4).

Why: G2 compares the numpy forward (float64, from the manifest) with the PyTorch fp32 golden. A model with
very large activations (OpenMed ChemicalDetect, state magnitude up to 8.5e3) makes the fp32 golden itself
deviate from the exact value by more than the gate (1e-4). This runs the same checkpoint in PyTorch float64
and reports three maxima over the model's golden items: numpy vs golden, numpy vs torch64, golden vs torch64.
Token classification, sequence classification, NLI and reranking models; writes
bench/results/k28-f64-reference-<slug>.json.

K28.S3 (decision 4): the result also holds the golden gap of every decision in torch64 (`gaps`: argmax,
zeroShot, bestPassage, tokenArgmax; 1e30 for a single class). An f32 engine decision that differs from the
fp32 golden is allowed only when its torch64 gap is below twice golden32_vs_torch64 (convert/k28_close_calls.py).

Usage: .venv-k28/bin/python convert/k28_f64_reference.py <model-id>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k28_forward as F  # noqa: E402
from k28_common import ROOT, model_dir, repo_dir, slug  # noqa: E402


def make_exact_float64(model):
    """ModernBERT in transformers 5 is not float64 end to end: the rotary embedding forces float32 for
    inv_freq, positions and cos/sin, and the eager attention upcasts the softmax to float32. That leaves a
    1e-5 to 1e-4 error in the "float64" reference itself. This redoes both in float64 so the reference is exact.
    Returns a context manager that undoes the softmax patch; the rotary patch lives on the model instances."""
    import contextlib

    import torch.nn.functional as Fn
    from transformers.models.modernbert import modeling_modernbert as mm

    for mod in model.modules():
        if isinstance(mod, mm.ModernBertRotaryEmbedding):
            for lt in mod.layer_types:
                base = mod.config.rope_parameters[lt]["rope_theta"]
                dim = getattr(mod.config, "head_dim", None) or mod.config.hidden_size // mod.config.num_attention_heads
                inv = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
                setattr(mod, f"{lt}_inv_freq", inv)

            def fwd(x, position_ids, layer_type=None, mod=mod):
                inv = getattr(mod, f"{layer_type}_inv_freq").to(torch.float64)
                freqs = (inv[None, :, None] @ position_ids[:, None, :].to(torch.float64)).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

            mod.forward = fwd

    orig = Fn.softmax

    @contextlib.contextmanager
    def exact_softmax():
        Fn.softmax = lambda x, dim=None, _stacklevel=3, dtype=None: orig(x, dim=dim, dtype=None)
        try:
            yield
        finally:
            Fn.softmax = orig

    return exact_softmax()


BIG = 1e30


def top2_gap(values) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    return float(v[-1] - v[-2]) if len(v) > 1 else BIG


def decision_gaps(task: str, doc: dict, logits: list) -> dict:
    """Gaps of the decisions of one task in the float64 logits (list of arrays, one per golden item)."""
    gaps: dict[str, list[float]] = {}
    if task in ("sequence-classification", "nli"):
        gaps["argmax"] = [top2_gap(l) for l in logits]
    if task == "nli":
        ent = doc["entail_index"]
        gaps["zeroShot"] = [top2_gap([logits[t["text_index"] * 5 + j][ent] for j in range(5)]) for t in doc["texts"]]
    if task == "reranking":
        scores: dict[int, dict[int, float]] = {}
        for item, l in zip(doc["items"], logits):
            scores.setdefault(item["query"], {})[item["passage"]] = float(np.asarray(l).ravel()[0])
        gaps["bestPassage"] = [top2_gap([scores[q["query"]][k] for k in sorted(scores[q["query"]])]) for q in doc["queries"]]
    if task == "token-classification":
        gaps["tokenArgmax"] = [g for l in logits for g in np.sort(np.asarray(l), axis=-1)[:, -1] - np.sort(np.asarray(l), axis=-1)[:, -2]]
        gaps["tokenArgmax"] = [float(g) for g in gaps["tokenArgmax"]]
    return gaps


def main():
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

    model_id = sys.argv[1]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest, T = F.load_tensors(model_dir(model_id) / "f32")
    spec, head, task = manifest["spec"], manifest["head"], manifest["task"]
    gold = F.Goldens(model_id, task)
    cls = AutoModelForTokenClassification if task == "token-classification" else AutoModelForSequenceClassification
    model = cls.from_pretrained(str(repo_dir(model_id)), attn_implementation="eager", dtype=torch.float64).eval()
    exact = make_exact_float64(model)
    exact.__enter__()
    worst = {"numpy_vs_golden32": 0.0, "numpy_vs_torch64": 0.0, "golden32_vs_torch64": 0.0}
    logit_max = 0.0
    n = 0
    all64 = []
    for item in gold.doc["items"]:
        ids = np.array(item["input_ids"])
        got = F.head_forward(T, head, F.encoder(T, spec, ids, None))
        with torch.no_grad():
            t64 = model(input_ids=torch.tensor([ids]),
                        attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits[0].numpy()
        want = gold.array(item["logits"]) if task == "token-classification" else np.array(item["logits"])
        want = want.reshape(t64.shape)
        got = np.asarray(got).reshape(t64.shape)
        all64.append(t64)
        worst["numpy_vs_golden32"] = max(worst["numpy_vs_golden32"], float(np.abs(got - want).max()))
        worst["numpy_vs_torch64"] = max(worst["numpy_vs_torch64"], float(np.abs(got - t64).max()))
        worst["golden32_vs_torch64"] = max(worst["golden32_vs_torch64"], float(np.abs(want - t64).max()))
        logit_max = max(logit_max, float(np.abs(t64).max()))
        n += 1
    out = {"model": model_id, "items": n, "max_abs_logit": logit_max, **worst,
           "gaps": decision_gaps(task, gold.doc, all64)}
    (ROOT / f"bench/results/k28-f64-reference-{slug(model_id)}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
