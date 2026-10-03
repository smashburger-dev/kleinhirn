"""K28.8 ORT graphs (workorder, Festlegung 6 to 9, phase 1 step 4).

Per model and L one graph with the fixed shape [1, L] and int32 inputs
input_ids and attention_mask (K20 pattern, convert/export_onnx.py). The
wrapper puts head and pooling into the graph so that it returns exactly the
output of the workorder table: logits, or the pooled vector before the L2 norm
(pooling from 1_Pooling/config.json). TorchScript exporter (dynamo=False),
opset 17, attn_implementation "eager", ModernBERT without the torch.compile
path. DeBERTa: relative positions frozen per L as in export_onnx.py. RoBERTa
and XLM-R: position ids as a constant i + pad_token_id + 1 (the HF rule for
inputs without pad) instead of the ne/cumsum/mul/add chain.

Graphs (Festlegung 7): std (export), opt (optimizer, model_type bert, default
fusions); for BERT, RoBERTa, XLM-R, DistilBERT also opt-noemb (no
EmbedLayerNormalization) and opt-mha (use_multi_head_attention). f16 of every
graph through the optimizer's conversion (keep_io_types=True).

Per graph: node count, fused operators, file size, CPU parity (onnxruntime
CPU) of the f32 twin and of the f16 graph against the goldens on the golden
cases up to L tokens (padded to L, mask 0 on padding). Gate (Festlegung 9):
classification argmax 100 %, reranking best passage of every query 100 % (plus
the largest logit deviation), embeddings cosine against `pooled` >= 0.9999.
The std f32 graph also runs the 320 measurement inputs; its outputs are the
reference of Festlegung 10 (cpu-f32-L<L>.bin). The f32 twins are deleted after
the check; only the f16 graphs stay.

Report: bench/results/k28.8-export-<slug>.json (merged per L and graph).
Graphs: models/k28/<slug>/k28.8/onnx/L<L>-<graph>-f16.onnx.

Usage: PYTHONDONTWRITEBYTECODE=1 ../kleinhirn/.venv/bin/python convert/k28_8_export.py
       <model|slug> <L> [graph ...] [--keep-f32]
Without graphs: all graphs of the family.
"""
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model import OnnxModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_8_common import (  # noqa: E402
    K28_DIR, MODELS, N_INPUTS, ROOT, inputs_meta, model_id_of, sha256_file, slug, work_dir,
)

OPSET = 17
FUSED = ["Attention", "MultiHeadAttention", "SkipLayerNormalization", "EmbedLayerNormalization",
         "BiasGelu", "FastGelu"]
MIN_FREE = 10 * 10**9


def graphs_of(family: str) -> list[str]:
    if family in ("bert", "roberta", "xlm-roberta", "distilbert"):
        return ["std", "opt", "opt-noemb", "opt-mha"]
    return ["std", "opt"]


# ------------------------------------------------------------------ wrapper


class Wrapper(torch.nn.Module):
    """int32 [1, L] ids and mask in, the output of the workorder table out."""

    def __init__(self, model, family: str, task: str, pooling: str | None, length: int, pad_id: int | None):
        super().__init__()
        self.model, self.family, self.task, self.pooling = model, family, task, pooling
        if family in ("roberta", "xlm-roberta"):
            # HF: cumsum of (ids != pad) + pad; for inputs without pad that is i + pad + 1.
            self.register_buffer("positions", torch.arange(length, dtype=torch.long).unsqueeze(0) + pad_id + 1)
        else:
            self.positions = None

    def forward(self, input_ids, attention_mask):
        ids = input_ids.long()
        mask = attention_mask.long()
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if self.positions is not None:
            kwargs["position_ids"] = self.positions
        out = self.model(**kwargs)
        if self.task != "embeddings":
            return out.logits
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)


def pooling_of(repo: Path) -> str | None:
    cfg = repo / "1_Pooling/config.json"
    if not cfg.exists():
        return None
    p = json.loads(cfg.read_text())
    modes = [k for k in ("pooling_mode_cls_token", "pooling_mode_mean_tokens", "pooling_mode_max_tokens",
                         "pooling_mode_mean_sqrt_len_tokens") if p.get(k)]
    if modes == ["pooling_mode_cls_token"]:
        return "cls"
    if modes == ["pooling_mode_mean_tokens"]:
        return "mean"
    raise SystemExit(f"stop: pooling {modes} not covered")


def load_model(repo: Path, family: str, task: str):
    kwargs = {"attn_implementation": "eager", "torch_dtype": torch.float32}
    config = transformers.AutoConfig.from_pretrained(str(repo))
    if family == "modernbert":
        config.reference_compile = False
    cls = transformers.AutoModel if task == "embeddings" else transformers.AutoModelForSequenceClassification
    return cls.from_pretrained(str(repo), config=config, **kwargs).eval(), config


def export_std(model_id: str, length: int, out: Path) -> dict:
    family, task = MODELS[model_id]
    repo = K28_DIR / slug(model_id) / "repo"
    model, config = load_model(repo, family, task)
    wrapper = Wrapper(model, family, task, pooling_of(repo), length, config.pad_token_id).eval()
    ids = torch.from_numpy(load_inputs(model_id, length)[0:1].copy())
    mask = torch.ones_like(ids)
    patched = {}
    if family == "deberta-v2":
        from transformers.models.deberta_v2.modeling_deberta_v2 import (
            DebertaV2Encoder, build_relative_position)
        enc = model.deberta.encoder
        dummy = torch.zeros(1, 1, length, 1)
        rel_pos = build_relative_position(dummy, dummy, bucket_size=enc.position_buckets,
                                          max_position=enc.max_relative_positions)
        with torch.no_grad():
            reference = wrapper(ids, mask)
        original = DebertaV2Encoder.get_rel_pos
        DebertaV2Encoder.get_rel_pos = lambda self, h, query_states=None, relative_pos=None: rel_pos
        patched["restore"] = lambda: setattr(DebertaV2Encoder, "get_rel_pos", original)
        with torch.no_grad():
            frozen = wrapper(ids, mask)
        err = float((frozen - reference).abs().max())
        if err > 1e-5:
            patched["restore"]()
            raise SystemExit(f"stop: frozen relative positions changed the output by {err}")
        patched["frozenRelPosCheck"] = err
    try:
        t = time.time()
        with torch.no_grad():
            torch.onnx.export(wrapper, (ids.int(), mask.int()), str(out),
                              input_names=["input_ids", "attention_mask"], output_names=["output"],
                              opset_version=OPSET, dynamo=False)
        seconds = time.time() - t
    finally:
        if "restore" in patched:
            patched["restore"]()
    onnx.checker.check_model(str(out))
    info = {"seconds": round(seconds, 1), "torch": torch.__version__, "transformers": transformers.__version__,
            "pooling": pooling_of(repo), "padId": config.pad_token_id}
    if "frozenRelPosCheck" in patched:
        info["frozenRelPosMaxAbs"] = patched["frozenRelPosCheck"]
    if family in ("roberta", "xlm-roberta"):
        info["positions"] = f"constant i + {config.pad_token_id} + 1"
    return info


# ------------------------------------------------------------------ variants


def heads_hidden(model_id: str) -> tuple[int, int]:
    cfg = json.loads((K28_DIR / slug(model_id) / "repo/config.json").read_text())
    heads = cfg.get("num_attention_heads", cfg.get("n_heads"))
    hidden = cfg.get("hidden_size", cfg.get("dim"))
    return int(heads), int(hidden)


def optimize(model_id: str, src: Path, dst: Path, graph: str) -> dict:
    heads, hidden = heads_hidden(model_id)
    options = FusionOptions("bert")
    if graph == "opt-noemb":
        options.enable_embed_layer_norm = False
    if graph == "opt-mha":
        options.use_multi_head_attention = True
    fused = optimizer.optimize_model(str(src), model_type="bert", num_heads=heads, hidden_size=hidden,
                                     optimization_options=options)
    stats = fused.get_fused_operator_statistics()
    fused.save_model_to_file(str(dst))
    return {"model_type": "bert", "num_heads": heads, "hidden_size": hidden,
            "enable_embed_layer_norm": options.enable_embed_layer_norm,
            "use_multi_head_attention": options.use_multi_head_attention,
            "fused_operator_statistics": stats}


def to_f16(src: Path, dst: Path) -> None:
    half = OnnxModel(onnx.load(str(src)))
    half.convert_float_to_float16(keep_io_types=True)
    half.save_model_to_file(str(dst))


def census(path: Path) -> dict:
    model = onnx.load(str(path))
    ops = Counter(n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}:{n.op_type}"
                  for n in model.graph.node)
    plain = Counter(n.op_type for n in model.graph.node)
    return {"nodes": len(model.graph.node), "fused": {k: plain.get(k, 0) for k in FUSED},
            "ops": dict(ops.most_common()), "bytes": path.stat().st_size}


# ------------------------------------------------------------------ parity


def load_inputs(model_id: str, length: int) -> np.ndarray:
    meta = inputs_meta(model_id, length)
    path = work_dir(model_id) / meta["ids"]
    if sha256_file(path) != meta["idsSha256"]:
        raise SystemExit(f"stop: {path} differs from its sha256")
    return np.fromfile(path, np.int32).reshape(meta["shape"])


def golden_of(model_id: str) -> tuple[dict, np.ndarray | None]:
    family, task = MODELS[model_id]
    g = json.loads((ROOT / "tests/golden/k28" / slug(model_id) / f"{task}.json").read_text())
    pooled = None
    if task == "embeddings":
        gdir = K28_DIR / slug(model_id) / "golden"
        index = json.loads((gdir / "index.json").read_text())
        e = next(x for x in index["entries"] if x["name"] == g["pooled"]["name"])
        raw = (gdir / e["file"]).read_bytes()[e["offset"]: e["offset"] + e["bytes"]]
        pooled = np.frombuffer(raw, np.float32).reshape(e["shape"])
    return g, pooled


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_padded(sess, ids: list[int], length: int, pad: int) -> np.ndarray:
    x = np.full((1, length), pad, np.int32)
    m = np.zeros((1, length), np.int32)
    x[0, : len(ids)] = ids
    m[0, : len(ids)] = 1
    return sess.run(["output"], {"input_ids": x, "attention_mask": m})[0][0]


def parity(path: Path, model_id: str, length: int, pad: int) -> dict:
    family, task = MODELS[model_id]
    g, pooled = golden_of(model_id)
    sess = session(path)
    items = g["items"]
    use = [i for i, it in enumerate(items) if len(it["input_ids"]) <= length]
    outs = {i: run_padded(sess, items[i]["input_ids"], length, pad) for i in use}
    finite = all(np.isfinite(o).all() for o in outs.values())
    rep = {"cases": len(use), "of": len(items), "finite": bool(finite)}
    if task == "sequence-classification":
        agree = sum(int(np.argmax(outs[i]) == items[i]["argmax"]) for i in use)
        rep["argmaxAgreement"] = agree / len(use)
        rep["maxAbsLogitDiff"] = float(max(np.abs(outs[i] - np.asarray(items[i]["logits"])).max() for i in use))
        rep["pass"] = agree == len(use) and finite
    elif task == "reranking":
        queries = [q for q in g["queries"]
                   if all(len(it["input_ids"]) <= length for it in items if it["query"] == q["query"])]
        ok = 0
        for q in queries:
            rows = [i for i, it in enumerate(items) if it["query"] == q["query"]]
            scores = [float(outs[i][0]) for i in rows]
            ok += int(int(np.argmax(scores)) == q["best"])
        rep["queries"] = len(queries)
        rep["queriesOf"] = len(g["queries"])
        rep["bestPassageAgreement"] = ok / len(queries)
        rep["maxAbsLogitDiff"] = float(max(abs(float(outs[i][0]) - items[i]["logit"]) for i in use))
        rep["pass"] = ok == len(queries) and finite
    else:
        cos = []
        diff = 0.0
        for i in use:
            a, b = outs[i].astype(np.float64), pooled[i].astype(np.float64)
            cos.append(float(a @ b / np.sqrt((a @ a) * (b @ b))))
            diff = max(diff, float(np.abs(a - b).max()))
        rep["minCosine"] = min(cos)
        rep["meanCosine"] = float(np.mean(cos))
        rep["maxAbsDiff"] = diff
        rep["pass"] = min(cos) >= 0.9999 and finite
    return rep


def measurement_outputs(path: Path, model_id: str, length: int) -> np.ndarray:
    sess = session(path)
    ids = load_inputs(model_id, length)
    mask = np.ones((1, length), np.int32)
    return np.stack([sess.run(["output"], {"input_ids": ids[i: i + 1], "attention_mask": mask})[0][0]
                     for i in range(N_INPUTS)])


# ------------------------------------------------------------------ main


def free_bytes() -> int:
    return shutil.disk_usage(ROOT).free


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_f32 = "--keep-f32" in sys.argv
    model_id = model_id_of(args[0])
    length = int(args[1])
    family, task = MODELS[model_id]
    graphs = args[2:] or graphs_of(family)
    if any(g not in graphs_of(family) for g in graphs):
        raise SystemExit(f"stop: graphs {graphs}, allowed for {family}: {graphs_of(family)}")
    if free_bytes() < MIN_FREE:
        raise SystemExit(f"stop: {free_bytes() / 1e9:.1f} GB free, below 10 GB")
    torch.set_num_threads(4)
    out_dir = work_dir(model_id) / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = ROOT / f"bench/results/k28.8-export-{slug(model_id)}.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {
        "model": model_id, "family": family, "task": task, "lengths": {}}
    report["versions"] = {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "torch": torch.__version__,
                          "transformers": transformers.__version__, "opset": OPSET}
    rep_l = report["lengths"].setdefault(str(length), {"graphs": {}})
    pad = json.loads((K28_DIR / slug(model_id) / "repo/config.json").read_text()).get("pad_token_id") or 0

    std32 = out_dir / f"L{length}-std-f32.onnx"
    if not std32.exists():
        rep_l["export"] = export_std(model_id, length, std32)
        ref = measurement_outputs(std32, model_id, length)
        ref_file = work_dir(model_id) / f"cpu-f32-L{length}.bin"
        ref.astype(np.float32).tofile(ref_file)
        rep_l["cpuReference"] = {"file": str(ref_file.relative_to(ROOT)), "shape": list(ref.shape),
                                 "sha256": sha256_file(ref_file), "finite": bool(np.isfinite(ref).all()),
                                 "graph": std32.name, "provider": "CPUExecutionProvider"}
    for graph in graphs:
        if free_bytes() < MIN_FREE:
            raise SystemExit(f"stop: {free_bytes() / 1e9:.1f} GB free, below 10 GB")
        f32 = out_dir / f"L{length}-{graph}-f32.onnx"
        f16 = out_dir / f"L{length}-{graph}-f16.onnx"
        entry = {}
        if graph != "std":
            entry["optimizer"] = optimize(model_id, std32, f32, graph)
        to_f16(f32, f16)
        entry["f32"] = {**census(f32), "parityCpu": parity(f32, model_id, length, pad)}
        entry["f16"] = {**census(f16), "parityCpu": parity(f16, model_id, length, pad),
                        "file": str(f16.relative_to(ROOT)), "sha256": sha256_file(f16)}
        entry["admissibleCpu"] = entry["f32"]["parityCpu"]["pass"]
        entry["date"] = time.strftime("%Y-%m-%d %H:%M")
        rep_l["graphs"][graph] = entry
        if graph != "std" and not keep_f32:
            f32.unlink()
        report_file.write_text(json.dumps(report, indent=1) + "\n")
        p32, p16 = entry["f32"]["parityCpu"], entry["f16"]["parityCpu"]
        print(json.dumps({"model": model_id, "L": length, "graph": graph, "nodes": entry["f16"]["nodes"],
                          "fused": entry["f16"]["fused"], "f16MB": round(entry["f16"]["bytes"] / 1e6, 1),
                          "f32": {k: v for k, v in p32.items() if k not in ("of",)},
                          "f16": {k: v for k, v in p16.items() if k not in ("of",)}}), flush=True)
    if not keep_f32:
        std32.unlink()
    report_file.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {report_file.relative_to(ROOT)}, free {free_bytes() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
