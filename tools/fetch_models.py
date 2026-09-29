"""Fetch the pinned model checkpoints kleinhirn runs against.

Downloads each Hugging Face snapshot at a pinned commit and verifies the
sha256 of every file the pipeline consumes. Files land under models/,
which is gitignored — model weights never live in this repository.

Usage (inside a venv with huggingface_hub, see convert/requirements.txt):
    python3 tools/fetch_models.py [name ...]          # default: all
    python3 tools/fetch_models.py --list

After fetching, export kleinhirn weights with:
    .venv/bin/python convert/export_weights.py <name>
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Per model: HF repo + pinned commit + the files the pipeline reads,
# each with the sha256 recorded at download time. Verification compares
# against these hashes; a mismatch aborts before anything is trusted.
MODELS = {
    "small-upstream": {
        "repo": "fastino/gliner2.5-small-v1",
        "revision": "7e6f537f10337497069276892a5ef435028252ce",
        "target": MODELS_DIR / "small-upstream/checkpoint",
        "files": {
            "config.json": "0b7d9e1401ceeb83e992ec66d2f93bff7e5646428f1b4706ec527cf88f53578a",
            "encoder_config/config.json": "db837d0dc587f5858687ef860c1f400de10f3c3e44f88daef8cbda80d74e4c9c",
            "model.safetensors": "4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b",
            "tokenizer.json": "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
        },
    },
    "base-upstream": {
        "repo": "fastino/gliner2.5-base-v1",
        "revision": "b0c10b23313ec3ff028821dff298dd743e010706",
        "target": MODELS_DIR / "base-upstream/checkpoint",
        "files": {
            "config.json": "0eb92d00584d613aab32b2178f84a85176b62c87ae3689ce9084e83f6eba64d1",
            "encoder_config/config.json": "d36a845b9f25dcaf1ec45a1c4bdf65ea4ac20596537e14530ec9f660a63aeca4",
            "model.safetensors": "7274094de2e0c2a37a386f55fc4e23061a954da5bd7a335e7dfe56f2743c277a",
            "tokenizer.json": "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
        },
    },
    "multi-upstream": {
        "repo": "fastino/gliner2.5-multi-v1",
        "revision": "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        "target": MODELS_DIR / "multi-upstream/checkpoint",
        "files": {
            "config.json": "8b59a0f426a65859c89cd1ea850c3529c09aa3be3a6fafd8eddfdd17b1bf0146",
            "encoder_config/config.json": "fa4f9ef2903b5369ab172333aae4574e6a476511d7465845cf59f8360ee18716",
            "model.safetensors": "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395",
            "tokenizer.json": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
        },
    },
    "julia-1": {
        "repo": "SupersonicLabs/Julia-1",
        "revision": "a85b127321d580d65176c89ced8273f305745d85",
        "target": MODELS_DIR / "julia-1/repo",
        "files": {
            "config.json": "da8dc5e01fa7c23035ac5b9f02979ba114aac6fc039d7635f56e8043ef8f9672",
            "encoder/config.json": "c79cda42d42ddf777218254845ce92fb25a10dc35874be40d06abd707894523d",
            "inference-policy.json": "415468ae1c229186e63a277b2fd2e5ee3bbf4c00655125c0e43e01d5a8406e8a",
            "julia/__init__.py": "02485eb4dd7dadd13907380d1a9c3398709bd8af54b3e0e12b3061a3ebb5c1da",
            "julia/cuda.py": "138ab63182f25473ce3886296c47ac44e94fc50ae5270927ae4c373f7bb49137",
            "julia/data.py": "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
            "julia/inference.py": "79b4e716365a6e07da4580ead725d5b76d06ef3a824a3ec23738feba64704b36",
            "julia/model.py": "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81",
            "julia/probabilities.py": "2a0197d2e0fa5a3c4b06b93599a85706724df7b0d2cc821a13ed293aef59f206",
            "julia/router/README.md": "b590e128d869da6392f42145a7a235a08a49d4c94126fe77f83a96aede0cb547",
            "julia/router/__init__.py": "832cf0a44569941efba66dfa7099b645fdb1084082e40c6115bb0ac5edd78a96",
            "julia/router/build.py": "ea059c3a97425cadb1068fe41626b6d3bcd77e1efccfc154dbde29352c432ed9",
            "julia/router/encoder.py": "df25efee2ed916d52af9c4e9c0d98e80854ca70bbfc8140f1de871f290fd3238",
            "julia/router/engine.py": "91bb30987ff8626ed1610955c6fd796939a5d271e2405b73f8674ccb947eb931",
            "julia/router/native.py": "42bdba5ea7d874e1f53b2f17e99e731789611b5bd6734f687eaa2b0262c5d7ae",
            "julia/router/native/LAWS.bend": "1aacac9a06d800208e4440108b98387bf276446f94092e1ae5617e970903b9da",
            "julia/router/native/PROOF.bend": "360305f19b1172637e2013e44dea22b5ab2b4a9b27d1482d88198021e5524192",
            "julia/router/native/bridge.c": "d479a67f4a8b6f47638b0a395162a90bf967b9b3035b2a828977d83a2fa8dbe5",
            "julia/router/native/router.bend": "2445f1e982ac2ec707442aaf144b9b5315c7fef0fe4598f0f25f39b84afbc556",
            "julia/router/router.py": "624b0b82fd34e4c6874a37595fa91a1d79553d0ca9d8111ef8330ed805f6be7c",
            "julia/router/specialize.py": "4ba766a45785da0f48898143eb04e13bdf736019a8ba76b71ad84a4bfec8b2d9",
            "julia/router/tests/synthetic.py": "f2b37bd6ceaf184449f1037b6aa8fda3bcb6263c76b3d11453c11c18d23d0196",
            "julia/router/tests/test_engine.py": "eb91ee1963400cb732350dadb7f5e39cb68917c6314ca8039e24974310364d52",
            "julia/router/tests/test_router.py": "e4df6d1f7d0962d0b5b119572d2d3897f1f4cadaa72600f26f3e72090a49172b",
            "julia/router/transformer.py": "0ce04fab361ca4e8bed636c615e5a6e178fa1a52332db34457e5235c667e1786",
            "julia/typed.py": "ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d",
            "julia_config.json": "b9b881646beeac5414eaa78b39d3fe89ff8a2c062731ad718432620f9aee6bf7",
            "model.safetensors": "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72",
            "provenance.json": "d3322e07820ad33b38e4bf5b4c4833c6b7e858ca553754fb8cac682ee802fd25",
            "pyproject.toml": "863dcbbc79ab1b903dcc3b69d3bb6e1443ffcd025050174d33b1ef20c30ef84d",
            "scripts/reproduce_typed.py": "07ed6a7d68d1e367b4417ab61c1f7e7bba4858a4e6ba22b01797f244abb3b160",
            "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer/tokenizer_config.json": "6b069e57db0ce0794c22547725275f22618809c4ef4249307337e651a0dfef8c",
        },
    },
    "julia-1-onnx": {
        "repo": "SupersonicLabs/Julia-1-ONNX",
        "revision": "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8",
        "target": MODELS_DIR / "julia-1/onnx",
        "files": {
            "model.onnx": "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9",
            "model.onnx.data": "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1",
            "parity-cases.json": "534670f25823f3f9abf9e812500918fabc606682dfd4e2456d827889d9bdca20",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "6b069e57db0ce0794c22547725275f22618809c4ef4249307337e651a0dfef8c",
            "index.js": "0453faaef728aa8891d439bb720e938d43880e4df10e8dd294a413af656699ea",
            "benchmark-webgpu.html": "43bf2fd65225e209647a9814998c22cee9dc86f014f9bcc243bdfd30f262a8f6",
            "export.py": "1717aa8248ded2b8ebfc47e4b028c1608775bef07b8f31e75e02e46366fcbf88",
            "package.json": "dbd2d305327f4bb1ec08b39454be4f22e7128bfb61c7846ca977b34dc725278d",
        },
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, spec: dict) -> None:
    from huggingface_hub import hf_hub_download
    target = spec["target"]
    target.mkdir(parents=True, exist_ok=True)
    ok = 0
    for rel, want in spec["files"].items():
        src = Path(hf_hub_download(
            spec["repo"], rel, revision=spec["revision"], repo_type="model"))
        got = sha256(src)
        if got != want:
            raise SystemExit(
                f"{name}: {rel} sha256 mismatch\n  want {want}\n  got  {got}")
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != got:
            dst.write_bytes(src.read_bytes())
        ok += 1
    print(f"{name}: {ok} files verified @ {spec['revision'][:8]} -> {target}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        for name, spec in MODELS.items():
            print(f"{name}: {spec['repo']} @ {spec['revision']}")
        return
    names = [a for a in args if not a.startswith("-")] or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; known: {list(MODELS)}")
    for name in names:
        fetch(name, MODELS[name])


if __name__ == "__main__":
    main()
