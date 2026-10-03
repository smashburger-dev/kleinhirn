"""Compare two tools/k28_convert.ts reports (K28.5 step 2).

Shard and tokenizer hashes must be equal, the manifest hash may differ only
because of the new field `maxLength`. Checked against the manifests on disk:
the manifest without `maxLength` must hash like the old report.

Usage: .venv-k28/bin/python convert/k28_compare_convert.py OLD.json NEW.json OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    old = {r["id"]: r for r in json.loads(Path(sys.argv[1]).read_text())["rows"]}
    new = {r["id"]: r for r in json.loads(Path(sys.argv[2]).read_text())["rows"]}
    rows, shards, differing, manifests_only_max_length = [], 0, 0, 0
    for mid, nrow in new.items():
        orow = old[mid]
        for dtype in ("f32", "f16"):
            oh, nh = orow["hashes"][dtype], nrow["hashes"][dtype]
            assert sorted(oh) == sorted(nh), (mid, dtype, "file lists differ")
            bad = [f for f in oh if f != "manifest.json" and oh[f] != nh[f]]
            shards += len(oh) - 1
            differing += len(bad)
            path = ROOT / "models/k28" / mid.replace("/", "__") / dtype / "manifest.json"
            manifest = json.loads(path.read_text())
            maxlen = manifest.pop("maxLength")
            text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
            same = hashlib.sha256(text.encode()).hexdigest() == oh["manifest.json"]
            manifests_only_max_length += int(same)
            rows.append({"id": mid, "dtype": dtype, "maxLength": maxlen, "differingFiles": bad,
                         "manifestEqualWithoutMaxLength": same})
    out = {"models": len(new), "files_shards_and_tokenizer": shards, "files_differing": differing,
           "manifests": len(rows), "manifests_equal_without_maxLength": manifests_only_max_length,
           "rows": rows}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})
    if differing or manifests_only_max_length != len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
