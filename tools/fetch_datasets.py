"""Fetch the pinned benchmark inputs under models/k9-data and jev-benchmarks.

Everything lands in gitignored directories; dataset texts never enter the
repository. Each source is pinned to an immutable revision and verified by
sha256 where the pipeline recorded one.

Usage (inside .venv-julia, needs huggingface_hub + datasets + pyarrow):
    python3 tools/fetch_datasets.py            # everything
    python3 tools/fetch_datasets.py typed      # one group

After fetching, rebuild the frozen task manifests:
    .venv-julia/bin/python bench/k9/export_tasks.py
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K9 = ROOT / "models/k9-data"
MASSIVE = K9 / "massive"
JEV = ROOT / "models/jev-benchmarks"
JEV_COMMIT = "0d610cc53e79bcbec691312b0c4adb4a0e371642"

TYPED_REPO = "LocalLLaMA/typed-decisions"
TYPED_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
TYPED_FILE = "all/test-00000-of-00001.parquet"
TYPED_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"

MASSIVE_REPO = "AmazonScience/massive"
# Pinned commit of refs/convert/parquet (auto-converted parquet branch).
MASSIVE_REVISION = "ed58ac423a2f4121720918bf5301577edce4ffd3"
MASSIVE_SHA256 = {
    "af-ZA-test.parquet": "17d2160c76324291ac9ade669058f3d9b8dcb2d15a700c0c70e7ab5f52b1438f",
    "am-ET-test.parquet": "78f3c0f216ff554e7002aac6e164551f5af574594da28afaa9eab278984350d5",
    "ar-SA-test.parquet": "fabc7afec26a29ec31c2deb016d47f9a4e38c33fa63c4b92b9f069e2b9f1054c",
    "az-AZ-test.parquet": "a9367a44a3a1f2abb5b40cc6b2e41e7a0b10fe831bd91cfba0e283e8ad175e0a",
    "bn-BD-test.parquet": "71996ba55c7c133ef516b7b032d74f2ba40ca04ff5e1796cc87b4b05ed01c019",
    "ca-ES-test.parquet": "be25de215c46b418939f37a44b2c739a0b21202245d1eb234885fbf0cbfaa155",
    "cy-GB-test.parquet": "a0e72aee00032ee350369de00423cc12f8d7095313d08aa0d01a63dbc55bf776",
    "da-DK-test.parquet": "10f55ded56a212c8d305f93651aa26269bde7316028f0c032ae8d0e1b71bf229",
    "de-DE-test.parquet": "3f15d6dcac758a56b4528a06b002252846208ebb8101f6159bc16ed836c55982",
    "el-GR-test.parquet": "54e76c55ff0af230643d704d6bf36954c654ecf68f0feaa0e43347f991d8ed3c",
    "en-US-test.parquet": "c418da9a5f3a7425b5cf44941bcea032a31040dbda2da85e0dd9323785c7731e",
    "en-US-train.parquet": "9c8b3ce8a96ec3b0d4aee104d8778d9f6bacdbd18f2cea232f6d8333d24401b1",
    "es-ES-test.parquet": "ca94eebc836e61e8bc530966c6565bb2c2b9d6184cc910c767c7fa6a9563de19",
    "fa-IR-test.parquet": "157aa8e07db085a559523a52c31bbd2cf4620fe37fa8b0728b946507e6b1e09d",
    "fi-FI-test.parquet": "3e41ad98f5d0cfa7250ed4f77742f7b26405dc4430a5d385a48e43e870ec7339",
    "fr-FR-test.parquet": "aacd57aa2bcf6e1d037a350a41b70cb756bbf51199d2023f40a3ebb9d38a1981",
    "he-IL-test.parquet": "c26f881c0944a774e86c9cbf39372f3b0d9204b9ce10257699bc9f88fae46071",
    "hi-IN-test.parquet": "f09ebfb9519169e5920785a1d6225d5d352c7de4a002235af99765147d219875",
    "hu-HU-test.parquet": "e21e751ba634b8d0367834a336d077d5e750d4c216d73a9b0190762a94d1433e",
    "hy-AM-test.parquet": "bf09a81950fbb6af80ced2c90b7789ac89ba9bff6ec5e7087d1dab3dafa0adff",
    "id-ID-test.parquet": "f79f35050f8307bc1e256d27bb2bf377a1f011c1883ffa41f1a258eb24edfebe",
    "is-IS-test.parquet": "fd024308a7e1ebb48bbb9158fd49b8504dd079d563d43d14ff4de7a15d4113ba",
    "it-IT-test.parquet": "db30bd1b876d3357cca2f3205a9b03ae53b49395722f25bffb7c3dcc796a4610",
    "ja-JP-test.parquet": "017d55af0c9d90faa93363798976570e0f58c3c1f90bbdc4be45da43c1310d01",
    "jv-ID-test.parquet": "e22d2638f428bd260c5ffcc53eec6612bc493d7226851c73c8e78344a0fbd1c8",
    "ka-GE-test.parquet": "9359f2cae83faaf843f421e21daa9dbe00cc6a85ebf3753b190671bd6da9e771",
    "km-KH-test.parquet": "80e1310492f4802b58df4507f21d32a1234a60809ca0808d0aabf9b439c39dcf",
    "kn-IN-test.parquet": "cc31c39e6c421d423ba1bb858e2943997e6170b28051925c5ca81ad7d261723d",
    "ko-KR-test.parquet": "3e16361b8d1ecdf2de25684883f9b70a84b5397b965b393be957b9341c15bcfa",
    "lv-LV-test.parquet": "71aad929c35ee9c3e830968490946f4f239826f33101fd68f4a6a3685ac44fe1",
    "ml-IN-test.parquet": "c8f0f8f818a82951403f9d489c8516d3feb76bcf5ac7f4175a062ddd75512af4",
    "mn-MN-test.parquet": "524dc525c446ea858e0f53f6f21e9671b03773a3eedf84f134411ff64c323ec6",
    "ms-MY-test.parquet": "cd6f3610a1f59fd427c35a4a877d95533965ff9377fa0bd7cde376f79cf12cf6",
    "my-MM-test.parquet": "3f0c657af65cc9bde61060a258ece108911833814ce4edcd30c86ec8195b7e71",
    "nb-NO-test.parquet": "2abc9a5f3b98aa4802aeb6e7f858cc95642f134780ad5a457decb7b642b8eb76",
    "nl-NL-test.parquet": "4cfaf06796320aba9e29866caf34594a988bf6c15139e7d4c4d41a2cffc2c186",
    "pl-PL-test.parquet": "0a4cddb8eb96d1ced67fbf2bcc27b57d3fb4b80bb1a4e28c46f257b799725bfb",
    "pt-PT-test.parquet": "99f5b531928d1cd837e4427def29f5827af49dbd9742af458d71d3378b284ec4",
    "ro-RO-test.parquet": "38c2971641f85825f753eef79fcec47ce2aad317ec3ebd1aecc174224851de5a",
    "ru-RU-test.parquet": "d53f1c58d753e64b7cf7f10face6c4a57ac93e7f9db82826b43058eacaf2e46a",
    "sl-SL-test.parquet": "a3025a598c76753e14534bd5d411b304a90a62f4af635f02e4a41307628deb2c",
    "sq-AL-test.parquet": "c5e893a3accb21e771c84416444bbfbe6382a7591d35d696985d664d2a8aa839",
    "sv-SE-test.parquet": "1a62db57d5d6fd03fc32515ebe90952bea55dae8c154df2ae18b3a67c37d781a",
    "sw-KE-test.parquet": "edef2ab3dda2fc48eb8b4f1ebd35d0d74706f999703952b5805694b243314cb1",
    "ta-IN-test.parquet": "8b308250f0a45b25beebb5ad10371171fe955af9f1ece75d224808481f6e0516",
    "te-IN-test.parquet": "4fb52accf9dac9ade3369a50cc7fbaf464bbe0e11c49390fd2bf899192e3e2a1",
    "th-TH-test.parquet": "c4e4e4c77814961e15a95926135b22ce4dacfb2156e34a791ec13039a7b88bc6",
    "tl-PH-test.parquet": "be23839ccf32bbce1fd7999937ce9833b2e2f8f8fb222ee379b953337fbf7062",
    "tr-TR-test.parquet": "f4eed8e1230df594c8aabf26d2c32b2309b2acce3de46cba7b6a33e8c05f1601",
    "ur-PK-test.parquet": "b65ce087e7ff225f137dd6a2af217e5ce91806b5c69f67299720d49a4a0b0ce4",
    "vi-VN-test.parquet": "af38f1da3143d8ed44604158992d92daa0235201301da856b58ae38c7a1cb757",
    "zh-CN-test.parquet": "7eca92a47e6bffeec3fc7de751948b82904680353a42eef7e8c6fdbe89adb2db",
    "zh-TW-test.parquet": "b8f2d32bbfee535cbae6118533b2033c7dd64abd56d7e740b12ec53bee06c0a4",
}

BTZSC_REPO = "btzsc/btzsc"
BTZSC_REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
BTZSC_SPLITS = ["agnews", "emotiondair", "banking77"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src: Path, dst: Path, want: str | None) -> None:
    got = sha256(src)
    if want and got != want:
        raise SystemExit(
            f"{dst.name}: sha256 mismatch\n  want {want}\n  got  {got}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or sha256(dst) != got:
        dst.write_bytes(src.read_bytes())


def fetch_typed() -> None:
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(
        TYPED_REPO, TYPED_FILE, repo_type="dataset", revision=TYPED_REVISION))
    place(src, K9 / "typed-test.parquet", TYPED_SHA256)
    print(f"typed: verified @ {TYPED_REVISION[:8]}")


def fetch_massive() -> None:
    from huggingface_hub import hf_hub_download
    for name, want in MASSIVE_SHA256.items():
        locale, split = name.replace(".parquet", "").rsplit("-", 1)
        rel = f"{locale}/{split}/0000.parquet"
        src = Path(hf_hub_download(
            MASSIVE_REPO, rel, repo_type="dataset",
            revision=MASSIVE_REVISION))
        place(src, MASSIVE / name, want)
    print(f"massive: {len(MASSIVE_SHA256)} parquets verified "
          f"@ {MASSIVE_REVISION[:8]}")


def fetch_btzsc() -> None:
    from datasets import load_dataset
    cache = K9 / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for split in BTZSC_SPLITS:
        load_dataset(BTZSC_REPO, split, split="test",
                     revision=BTZSC_REVISION, cache_dir=str(cache))
    print(f"btzsc: {BTZSC_SPLITS} cached @ {BTZSC_REVISION[:8]}")


def fetch_jev() -> None:
    if not JEV.exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none",
             "https://github.com/AbdelStark/jev-benchmarks", str(JEV)],
            check=True)
    subprocess.run(["git", "-C", str(JEV), "checkout", "--quiet", JEV_COMMIT],
                   check=True)
    head = subprocess.run(
        ["git", "-C", str(JEV), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if head != JEV_COMMIT:
        raise SystemExit(f"jev-benchmarks: HEAD {head} != {JEV_COMMIT}")
    print(f"jev-benchmarks: checked out @ {JEV_COMMIT[:8]}")


def main() -> None:
    steps = {
        "typed": fetch_typed,
        "massive": fetch_massive,
        "btzsc": fetch_btzsc,
        "jev": fetch_jev,
    }
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(steps)
    for name in names:
        if name not in steps:
            raise SystemExit(f"unknown group: {name}; known: {list(steps)}")
        steps[name]()


if __name__ == "__main__":
    main()
