"""Download K28 models at the frozen revision (K28.2 step 1).

Reads revision, weight file and sha256 from data/k28/models.json, downloads
only config, tokenizer and sentence-transformers files plus the one weight
file into models/k28/<slug>/repo, and checks the weight sha256. The HF cache
lives under models/k28/_hf. Aborts when free disk space is below 15 GB.

With --tokenizer-only no weight file and no Sentence-Transformers modules are
fetched (K28.3 step 1); --wordpiece selects the 32 WordPiece models of the list,
--all-nonwordpiece the other 57 (K28.3b; also vocab.json, merges.txt, spm.model).

K28.5 (--pilot-k28.5): the 16 RoBERTa, XLM-R and DistilBERT models of
docs/K28_5_WORKORDER.md, result in bench/results/k28-fetch-k28.5.json.

K28.6 (--pilot-k28.6): the 11 DeBERTa and ModernBERT models of docs/K28_6_WORKORDER.md, result in
bench/results/k28-fetch-k28.6.json.

Usage: .venv-k28/bin/python convert/k28_fetch.py <model-id>... | --pilot | --wordpiece
       | --pilot-k28.5 | --pilot-k28.6 [--tokenizer-only]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import (  # noqa: E402
    K28_DIR, ROOT, load_entry, repo_dir, resolve_ids, slug, weight_file,
)

os.environ["HF_HOME"] = str(K28_DIR / "_hf")
from huggingface_hub import snapshot_download  # noqa: E402

MIN_FREE_BYTES = 15 * 10**9
RESULT = ROOT / "bench/results/k28-fetch-pilot.json"
RESULT_K285 = ROOT / "bench/results/k28-fetch-k28.5.json"
RESULT_K286 = ROOT / "bench/results/k28-fetch-k28.6.json"
PATTERNS = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt",
    "spm.model", "sentencepiece.bpe.model", "added_tokens.json", "modules.json",
    "config_sentence_transformers.json", "sentence_bert_config.json",
    "*_Pooling/config.json", "*_Dense/config.json", "*_Dense/model.safetensors",
    "*_Dense/pytorch_model.bin",
]


def free_bytes() -> int:
    return shutil.disk_usage(ROOT).free


def dir_bytes(path: Path, skip_cache: bool = True) -> tuple[int, int]:
    total = 0
    count = 0
    for p in path.rglob("*"):
        if skip_cache and ".cache" in p.parts:
            continue
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
            count += 1
    return total, count


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


TOKENIZER_PATTERNS = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "added_tokens.json",
    "vocab.json", "merges.txt", "spm.model", "sentencepiece.bpe.model",
]
MAX_TOKENIZER_BYTES = 100 * 10**6
MAX_TOKENIZER_BYTES_NONWP = 500 * 10**6


def fetch_tokenizer(model_id: str) -> dict:
    entry = load_entry(model_id)
    free_before = free_bytes()
    if free_before < MIN_FREE_BYTES:
        raise SystemExit(f"stop: {free_before / 1e9:.1f} GB free, below 15 GB")
    target = repo_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id, revision=entry["revision"], local_dir=str(target),
        allow_patterns=TOKENIZER_PATTERNS,
    )
    names = sorted(p.name for p in target.iterdir() if p.is_file() and p.name in TOKENIZER_PATTERNS)
    size = sum((target / n).stat().st_size for n in names)
    return {"id": model_id, "slug": slug(model_id), "revision": entry["revision"],
            "files": names, "tokenizerBytes": size, "freeBytesAfter": free_bytes()}


def fetch(model_id: str) -> dict:
    entry = load_entry(model_id)
    weight = weight_file(entry)
    free_before = free_bytes()
    if free_before < MIN_FREE_BYTES:
        raise SystemExit(f"stop: {free_before / 1e9:.1f} GB free, below 15 GB")
    target = repo_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)
    t = time.time()
    snapshot_download(
        repo_id=model_id, revision=entry["revision"], local_dir=str(target),
        allow_patterns=PATTERNS + [weight],
    )
    seconds = time.time() - t
    path = target / weight
    digest = sha256_file(path)
    if digest not in entry["weightSha256"]:
        raise SystemExit(f"stop: sha256 mismatch for {model_id}/{weight}: {digest}")
    downloaded, files = dir_bytes(target)
    return {
        "id": model_id, "slug": slug(model_id), "revision": entry["revision"],
        "weightFile": weight, "weightSha256": digest, "sha256Ok": True,
        "listedWeightBytes": entry["weightBytes"], "weightBytes": path.stat().st_size,
        "downloadBytes": downloaded, "files": files,
        "diskBytes": dir_bytes(target, skip_cache=False)[0],
        "seconds": round(seconds, 1),
        "freeBytesBefore": free_before, "freeBytesAfter": free_bytes(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-k28.5", dest="k285", action="store_true")
    parser.add_argument("--pilot-k28.6", dest="k286", action="store_true")
    parser.add_argument("--wordpiece", action="store_true")
    parser.add_argument("--all-nonwordpiece", action="store_true")
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--result", type=Path, help="result file instead of the default of the group (K28.S sweep)")
    args = parser.parse_args()
    ids = resolve_ids(args.models, args.pilot, args.wordpiece, args.k285, args.all_nonwordpiece, args.k286)
    if args.tokenizer_only:
        rows = []
        for model_id in ids:
            row = fetch_tokenizer(model_id)
            rows.append(row)
            print(json.dumps(row), flush=True)
        total = sum(r["tokenizerBytes"] for r in rows)
        suffix = "nonwordpiece" if args.all_nonwordpiece else "wordpiece"
        limit = MAX_TOKENIZER_BYTES_NONWP if args.all_nonwordpiece else MAX_TOKENIZER_BYTES
        (ROOT / f"bench/results/k28-fetch-tokenizers-{suffix}.json").write_text(json.dumps({
            "rows": rows, "totalTokenizerBytes": total, "freeBytesEnd": free_bytes(),
        }, indent=1) + "\n")
        if total > limit:
            raise SystemExit(f"stop: {total} tokenizer bytes exceed the {limit // 10**6} MB limit")
        return
    rows = []
    for model_id in ids:
        row = fetch(model_id)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = args.result or (RESULT_K286 if args.k286 else RESULT_K285 if args.k285 else RESULT)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        "rows": rows, "totalDownloadBytes": sum(r["downloadBytes"] for r in rows),
        "freeBytesEnd": free_bytes(),
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
