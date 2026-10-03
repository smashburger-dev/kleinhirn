"""Count the switches of the 57 non-WordPiece tokenizers (K28.3b): BPE and Unigram model flags,
merge formats, added-token flags, normalizer, pre-tokenizer and post-processor types.

Counts are per model and per distinct tokenizer.json (sha256). Result:
bench/results/k28-tokenizer-switches.json

Usage: .venv-k28/bin/python convert/k28_tokenizer_switches.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import K28_DIR, ROOT, slug  # noqa: E402


def walk(node):
    if node is None:
        return
    if node["type"] == "Sequence":
        for child in node.get("normalizers") or node.get("pretokenizers") or node.get("processors") or []:
            yield from walk(child)
    else:
        yield node


def features(tj):
    out = Counter()
    m = tj["model"]
    out[f"model {m['type']}"] += 1
    if m["type"] == "BPE":
        for key in ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk",
                    "byte_fallback", "ignore_merges"):
            out[f"BPE {key}={json.dumps(m.get(key))}"] += 1
        out[f"BPE merges as {'pairs' if isinstance(m['merges'][0], list) else 'strings'}"] += 1
    if m["type"] == "Unigram":
        out[f"Unigram byte_fallback={json.dumps(m.get('byte_fallback'))}"] += 1
        out["Unigram unk_id=" + ("null" if m.get("unk_id") is None else "set")] += 1
    for n in walk(tj["normalizer"]):
        out[f"normalizer {n['type']}"] += 1
        if n["type"] == "Replace":
            kind, pattern = next(iter(n["pattern"].items()))
            out[f"Replace {kind} {pattern!r} -> {n['content']!r}"] += 1
        if n["type"] == "Strip":
            out[f"Strip left={n['strip_left']} right={n['strip_right']}"] += 1
    for p in walk(tj["pre_tokenizer"]):
        desc = {k: v for k, v in p.items() if k != "type"}
        out[f"pre_tokenizer {p['type']} {json.dumps(desc, sort_keys=True)}"] += 1
    post = tj["post_processor"]
    out[f"post_processor {post['type'] if post else None}"] += 1
    if post and post["type"] in ("RobertaProcessing", "ByteLevel"):
        out[f"post_processor {post['type']} trim_offsets={post['trim_offsets']} add_prefix_space={post['add_prefix_space']}"] += 1
    for a in tj["added_tokens"]:
        for flag in ("lstrip", "rstrip", "single_word", "normalized", "special"):
            if a[flag]:
                out[f"added token {flag}"] += 1
        out["added token"] += 1
    out["models with an added token with lstrip"] += int(any(a["lstrip"] for a in tj["added_tokens"]))
    out["models with an added token with rstrip"] += int(any(a["rstrip"] for a in tj["added_tokens"]))
    out["models with an added token with single_word"] += int(any(a["single_word"] for a in tj["added_tokens"]))
    out["models with a normalized added token that is not special"] += int(
        any(a["normalized"] and not a["special"] for a in tj["added_tokens"]))
    return out


def main():
    lst = json.loads((ROOT / "bench/results/k28-tokenizers-nonwordpiece.json").read_text())
    perModel, perSha = Counter(), Counter()
    for sha, models in lst["bySha256"].items():
        d = K28_DIR / slug(models[0])
        path = d / "tokenizer.json" if (d / "tokenizer.json").exists() else d / "repo/tokenizer.json"
        f = features(json.loads(path.read_text()))
        for k, v in f.items():
            perModel[k] += v * len(models) if not k.startswith("added token") else v * len(models)
            perSha[k] += v
    doc = {"models": lst["models"], "distinctTokenizers": lst["tokenizers"],
           "perModel": dict(sorted(perModel.items())), "perDistinctTokenizer": dict(sorted(perSha.items()))}
    (ROOT / "bench/results/k28-tokenizer-switches.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for k in sorted(perModel):
        if not k.startswith("added token ") or True:
            print(f"{perModel[k]:5d} {perSha[k]:4d}  {k}")


if __name__ == "__main__":
    main()
