"""K20gc: rewrite the fused small-upstream graphs so ORT Web can run with
enableGraphCapture. Capture needs every compute node on the WebGPU EP
(only shape-only CPU subgraphs are tolerated); the DeBERTa mask chain
(int64 Unsqueeze/Squeeze/Mul on the [1,128] attention mask) forced
MemcpyFromHost/MemcpyToHost around /encoder/encoder/Squeeze. The input
shapes are fixed ([1,128], [1,16]), so the chain is rebuilt with constant
Reshapes. Everything else is left untouched.

Rounds (--round N), each starting from model_<p>_opt.onnx:
  1  Squeeze + Unsqueeze_2 -> one Reshape [1,1,128,1] (still int64)
  2  the whole mask chain in float32: Cast(attention_mask, float) ->
     Reshape [1,1,1,128] and [1,1,128,1] -> Mul -> Cast(bool)
  3  as 2, with the two Reshapes fed from a Cast to float16 (fp16 graph only)
Output: models/small-upstream/onnx/model_<p>_opt_gc.onnx and the report
bench/results/k20gc-surgery-small-upstream.json (CPU parity, 968 goldens,
same routine as convert/optimize_onnx.py).

Usage: .venv/bin/python convert/capture_surgery.py [--round N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_onnx import CONFIGS, ROOT, census, file_bytes, parity  # noqa: E402

CHAIN = ["/encoder/encoder/Unsqueeze", "/encoder/encoder/Unsqueeze_1",
         "/encoder/encoder/Squeeze", "/encoder/encoder/Unsqueeze_2",
         "/encoder/encoder/Mul"]
MASK_MUL = "/encoder/encoder/Mul"


def const(name, values):
    return numpy_helper.from_array(np.asarray(values, np.int64), name)


def reshape(graph, name, src, dst, shape):
    graph.initializer.append(const(f"{name}_shape", shape))
    return helper.make_node("Reshape", [src, f"{name}_shape"], [dst], name=name)


def surgery(model, round_):
    g = model.graph
    nodes = {n.name: n for n in g.node}
    replaced = []
    if round_ == 1:
        new = [reshape(g, "/gc/mask_col", "/encoder/encoder/Unsqueeze_1_output_0",
                       "/encoder/encoder/Unsqueeze_2_output_0", [1, 1, 128, 1])]
        drop = ["/encoder/encoder/Squeeze", "/encoder/encoder/Unsqueeze_2"]
    else:
        mul = nodes[MASK_MUL]
        mul_out = mul.output[0]
        cast_bool = next(n for n in g.node if n.input and n.input[0] == mul_out)
        new = [
            helper.make_node("Cast", ["attention_mask"], ["/gc/mask_f"],
                             name="/gc/mask_cast", to=TensorProto.FLOAT),
            reshape(g, "/gc/mask_row", "/gc/mask_f", "/gc/mask_row_out", [1, 1, 1, 128]),
            reshape(g, "/gc/mask_col", "/gc/mask_f", "/gc/mask_col_out", [1, 1, 128, 1]),
            helper.make_node("Mul", ["/gc/mask_row_out", "/gc/mask_col_out"],
                             ["/gc/mask_mul"], name="/gc/mask_mul"),
            helper.make_node("Cast", ["/gc/mask_mul"], [cast_bool.output[0]],
                             name="/gc/mask_bool", to=TensorProto.BOOL),
        ]
        drop = CHAIN + [cast_bool.name]
    keep = [n for n in g.node if n.name not in drop]
    replaced = [n.name for n in g.node if n.name in drop]
    first = min(i for i, n in enumerate(g.node) if n.name in drop)
    insert_at = sum(1 for n in list(g.node)[:first] if n.name not in drop)
    ordered = keep[:insert_at] + new + keep[insert_at:]
    del g.node[:]
    g.node.extend(ordered)
    return replaced, [n.name for n in new]


def main():
    round_ = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 1
    cfg = CONFIGS["small-upstream"]
    items = json.loads((ROOT / cfg["golden"]).read_text())["items"]
    report = {"round": round_, "precisions": {}}
    for prec in ("f32", "f16"):
        src = ROOT / cfg[prec]
        dst = src.with_name(src.stem + "_gc.onnx")
        model = onnx.load(str(src))
        before = len(model.graph.node)
        replaced, added = surgery(model, round_)
        onnx.checker.check_model(model)
        onnx.save(model, str(dst))
        report["precisions"][prec] = {
            "replaced_nodes": replaced, "added_nodes": added,
            "nodes_before": before, "nodes_after": len(model.graph.node),
            "bytes_before": file_bytes(src), "bytes_after": file_bytes(dst),
            "parity_cpu": parity(dst, items, cfg["feeds"]),
            "census_after": census(model)["nodes"],
            "file": str(dst.relative_to(ROOT)),
        }
    out = ROOT / "bench" / "results" / "k20gc-surgery-small-upstream.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({p: {k: v for k, v in r.items() if k != "census_after"}
                      for p, r in report["precisions"].items()}, indent=1))


if __name__ == "__main__":
    main()
