"""tokenizer.json from a DeBERTa-v2 spm.model without the sentencepiece and protobuf packages.

MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c has only spm.model. AutoTokenizer would convert
it with DebertaV2Converter (transformers/convert_slow_tokenizer.py), which needs sentencepiece and
protobuf; .venv-k28 has neither, and nothing else is installed. This reads the four fields the converter
uses from the ModelProto wire format (pieces with score and type, trainer_spec.unk_id,
normalizer_spec.precompiled_charsmap) and builds the same tokenizer with the tokenizers library.
check() converts the spm.model of a model that has a tokenizer.json from the same file
(MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) and compares normalizer, pre-tokenizer, model and
post-processor with it.

Usage: .venv-k28/bin/python convert/k28_spm.py --check
"""
import json
import struct
import sys
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import model_dir, repo_dir  # noqa: E402


def _varint(buf, i):
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def _fields(buf):
    """(field number, wire type, value) of one message; value is an int, bytes or the raw 4 bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"wire type {wire}")
        yield number, wire, value


def parse_spm(path):
    """pieces [(piece, score, type)], unk_id, precompiled charsmap bytes of a SentencePiece model."""
    pieces, unk_id, charsmap, model_type = [], 0, b"", 1
    for number, _, value in _fields(Path(path).read_bytes()):
        if number == 1:
            piece, score, kind = "", 0.0, 1
            for n, _, v in _fields(value):
                if n == 1:
                    piece = v.decode("utf-8")
                elif n == 2:
                    score = struct.unpack("<f", v)[0]
                elif n == 3:
                    kind = v
            pieces.append((piece, score, kind))
        elif number == 2:  # trainer_spec
            for n, _, v in _fields(value):
                if n == 3:
                    model_type = v
                elif n == 40:
                    unk_id = v
        elif number == 3:  # normalizer_spec
            for n, _, v in _fields(value):
                if n == 2:
                    charsmap = bytes(v)
    if model_type != 1:
        raise ValueError(f"model_type {model_type} is not Unigram")
    return pieces, unk_id, charsmap


def deberta_v2_tokenizer(spm_path, mask_id):
    """DebertaV2Converter with split_by_punct false, do_lower_case false, add_prefix_space true."""
    pieces, unk_id, charsmap = parse_spm(spm_path)
    tk = Tokenizer(models.Unigram([(p, s) for p, s, _ in pieces], unk_id=unk_id, byte_fallback=False))
    norms = [normalizers.Strip()]
    if charsmap:
        norms.append(normalizers.Precompiled(charsmap))
    norms.append(normalizers.Replace(Regex(" {2,}"), " "))
    tk.normalizer = normalizers.Sequence(norms)
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")])
    tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    # control (3) and user defined (4) pieces are added tokens, control ones are special
    tk.add_tokens([AddedToken(p, normalized=False, special=k == 3)
                   for _, (p, _, k) in sorted(enumerate(pieces)) if k in (3, 4)])
    ids = {p: i for i, (p, _, _) in enumerate(pieces)}
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0", pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])])
    # [UNK] is a piece of type unknown, not control, and [MASK] comes from added_tokens.json behind
    # the vocabulary: both are special tokens the slow tokenizer adds, flags as in the converted files
    assert mask_id == len(pieces)
    tk.add_special_tokens([AddedToken("[UNK]", normalized=True, special=True),
                           AddedToken("[MASK]", normalized=False, special=True)])
    return tk


def convert_model(model_id):
    repo = repo_dir(model_id)
    added = json.loads((repo / "added_tokens.json").read_text())
    tk = deberta_v2_tokenizer(repo / "spm.model", added["[MASK]"])
    out = model_dir(model_id) / "tokenizer.json"
    out.write_text(tk.to_str())
    return out


def check():
    sibling = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    repo = repo_dir(sibling)
    added = json.loads((repo / "added_tokens.json").read_text())
    mine = json.loads(deberta_v2_tokenizer(repo / "spm.model", added["[MASK]"]).to_str())
    theirs = json.loads((repo / "tokenizer.json").read_text())

    def norm(o):
        s = json.dumps(o, sort_keys=True)
        return json.loads(s.replace('"add_prefix_space": true, ', ""))
    bad = []
    for key in ("normalizer", "pre_tokenizer", "post_processor", "model", "added_tokens"):
        a, b = mine[key], theirs[key]
        if key == "pre_tokenizer":  # the file also carries the legacy add_prefix_space key
            for t in (a, b):
                for p in t["pretokenizers"]:
                    p.pop("add_prefix_space", None)
                    p.setdefault("split", True)
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            bad.append(key)
    print("differing parts:", bad or "none")
    return not bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
