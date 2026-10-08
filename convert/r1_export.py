"""R1 ORT graphs (docs/R1_WORKORDER.md, Festlegung 9), built on convert/k28_8_export.py.

Per model, for the graph type of the K28.8 winner (std or opt, data/k28/k28.8-best.json):
  L128-<type>-f32.onnx, L512-<type>-f32.onnx   fixed shape [1, L], int32 inputs
  Ldyn-<type>-f32.onnx, Ldyn-<type>-f16.onnx   dynamic sequence length, batch 1
The f16 graphs at fixed L already exist from K28.8 (L<L>-<type>-f16.onnx).
Weights: the original Hugging Face checkpoint in repo/ (convert/k28_fetch.py, frozen revisions of data/k28/models.json).
Each graph gets the CPU parity of K28.8 Festlegung 9 (goldens up to L tokens; the
dynamic graph runs every golden case at its own length, without padding).
A dynamic graph that fails the parity or the export is recorded with the reason and
no file is kept (the cell then runs with the fixed graphs).

Report: bench/results/r1-export-<slug>.json
Usage: PYTHONDONTWRITEBYTECODE=1 ../kleinhirn/.venv/bin/python convert/r1_export.py <model|slug> [128] [512] [dyn]
"""
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k28_8_export as ex  # noqa: E402
from k28_8_common import K28_DIR, MODELS, ROOT, model_id_of, sha256_file, slug, work_dir  # noqa: E402

MIN_FREE = 8 * 10**9
BEST = json.loads((ROOT / "data/k28/k28.8-best.json").read_text())


def graph_type(model_id: str) -> str:
    return BEST["cells"][f"{slug(model_id)}|L128"]["winner"]["graph"]


def free() -> int:
    return shutil.disk_usage(ROOT).free


def need_space(what: str) -> None:
    if free() < MIN_FREE:
        raise SystemExit(f"stop: {free() / 1e9:.1f} GB free, below 8 GB ({what})")


class DynWrapper(ex.Wrapper):
    """As Wrapper, with position ids computed from the input length (RoBERTa, XLM-R)."""

    def __init__(self, model, family, task, pooling, pad_id):
        super().__init__(model, family, task, pooling, 1, pad_id)
        self.pad_id = pad_id
        self.dyn_positions = family in ("roberta", "xlm-roberta")

    def forward(self, input_ids, attention_mask):
        ids = input_ids.long()
        mask = attention_mask.long()
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if self.dyn_positions:
            kwargs["position_ids"] = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0) + self.pad_id + 1
        out = self.model(**kwargs)
        if self.task != "embeddings":
            return out.logits
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)


def export_dynamic(model_id: str, out: Path) -> dict:
    family, task = MODELS[model_id]
    repo = K28_DIR / slug(model_id) / "repo"
    model, config = ex.load_model(repo, family, task)
    wrapper = DynWrapper(model, family, task, ex.pooling_of(repo), config.pad_token_id).eval()
    length = 96  # trace length; the axis is dynamic
    ids = torch.from_numpy(ex.load_inputs(model_id, 128)[0:1, :length].copy())
    mask = torch.ones_like(ids)
    t = time.time()
    with torch.no_grad():
        torch.onnx.export(wrapper, (ids.int(), mask.int()), str(out),
                          input_names=["input_ids", "attention_mask"], output_names=["output"],
                          dynamic_axes={"input_ids": {1: "seq"}, "attention_mask": {1: "seq"}},
                          opset_version=ex.OPSET, dynamo=False)
    onnx.checker.check_model(str(out))
    return {"seconds": round(time.time() - t, 1), "traceLength": length, "dynamicAxes": "sequence"}


def run_exact(sess, ids, length, pad):
    x = np.asarray([ids], np.int32)
    m = np.ones_like(x)
    return sess.run(["output"], {"input_ids": x, "attention_mask": m})[0][0]


def parity_dynamic(path: Path, model_id: str, pad: int) -> dict:
    saved = ex.run_padded
    ex.run_padded = run_exact
    try:
        return ex.parity(path, model_id, 10**9, pad)
    finally:
        ex.run_padded = saved


def build_fixed(model_id: str, length: int, gtype: str, report: dict, pad: int) -> None:
    out_dir = work_dir(model_id) / "onnx"
    final = out_dir / f"L{length}-{gtype}-f32.onnx"
    entry = report["graphs"].setdefault(f"L{length}-{gtype}-f32", {})
    std32 = out_dir / f"L{length}-std-f32.onnx"
    need_space(f"export L{length}")
    entry["export"] = ex.export_std(model_id, length, std32)
    if gtype == "opt":
        entry["optimizer"] = ex.optimize(model_id, std32, final, "opt")
        std32.unlink()
    entry.update(ex.census(final))
    entry["parityCpu"] = ex.parity(final, model_id, length, pad)
    entry["admissibleCpu"] = entry["parityCpu"]["pass"]
    entry["file"] = str(final.relative_to(ROOT))
    entry["sha256"] = sha256_file(final)
    entry["date"] = time.strftime("%Y-%m-%d %H:%M")
    if not entry["admissibleCpu"]:
        raise SystemExit(f"stop: {final.name} fails the CPU parity: {entry['parityCpu']}")


def build_dynamic(model_id: str, gtype: str, report: dict, pad: int) -> None:
    out_dir = work_dir(model_id) / "onnx"
    f32 = out_dir / f"Ldyn-{gtype}-f32.onnx"
    f16 = out_dir / f"Ldyn-{gtype}-f16.onnx"
    std32 = out_dir / "Ldyn-std-f32.onnx"
    entry = report["graphs"].setdefault(f"Ldyn-{gtype}", {})
    try:
        need_space("export dyn")
        entry["export"] = export_dynamic(model_id, std32)
        if gtype == "opt":
            entry["optimizer"] = ex.optimize(model_id, std32, f32, "opt")
            std32.unlink()
        else:
            std32.rename(f32)
        entry["f32"] = {**ex.census(f32), "parityCpu": parity_dynamic(f32, model_id, pad)}
        ex.to_f16(f32, f16)
        entry["f16"] = {**ex.census(f16), "parityCpu": parity_dynamic(f16, model_id, pad),
                        "file": str(f16.relative_to(ROOT)), "sha256": sha256_file(f16)}
        entry["f32"]["file"] = str(f32.relative_to(ROOT))
        entry["f32"]["sha256"] = sha256_file(f32)
        entry["admissibleCpu"] = bool(entry["f32"]["parityCpu"]["pass"])
        if not entry["admissibleCpu"]:
            entry["failed"] = f"CPU parity: {entry['f32']['parityCpu']}"
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - recorded, the cell falls back to the fixed graphs
        entry["admissibleCpu"] = False
        entry["failed"] = f"{type(e).__name__}: {str(e)[:400]}"
        traceback.print_exc()
    if not entry.get("admissibleCpu"):
        for p in (f32, f16, std32):
            p.unlink(missing_ok=True)
    entry["date"] = time.strftime("%Y-%m-%d %H:%M")


def main():
    args = sys.argv[1:]
    model_id = model_id_of(args[0])
    todo = args[1:] or ["128", "512", "dyn"]
    torch.set_num_threads(4)
    gtype = graph_type(model_id)
    report_file = ROOT / f"bench/results/r1-export-{slug(model_id)}.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {
        "model": model_id, "graphType": gtype, "graphs": {}}
    report["versions"] = {"onnx": onnx.__version__, "torch": torch.__version__, "opset": ex.OPSET,
                          "weights": "original Hugging Face checkpoints at the frozen revisions (bench/results/r1-fetch.json)"}
    pad = json.loads((K28_DIR / slug(model_id) / "repo/config.json").read_text()).get("pad_token_id") or 0
    for t in todo:
        if t == "dyn":
            build_dynamic(model_id, gtype, report, pad)
        else:
            build_fixed(model_id, int(t), gtype, report, pad)
        report_file.write_text(json.dumps(report, indent=1) + "\n")
        print(f"{slug(model_id)} {t}: free {free() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
