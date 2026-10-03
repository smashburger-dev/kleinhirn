"""Tiny tokenizer.json fixtures for tests/hf_tokenizer_b.test.ts, expectations from HF tokenizers.

Builds four small tokenizers with the tokenizers library (ByteLevel BPE with RobertaProcessing,
NFC plus normalized added tokens, Metaspace Unigram with Strip and Replace, Metaspace BPE with
byte fallback), encodes a fixed list of texts and pairs, and writes tokenizer and expected
results (ids, type ids, offsets in code points, word ids, sequence ids) to tests/fixtures/hf-tokenizers/.
Offsets here count code points; the test converts them to UTF-16 units.

Usage: .venv-k28/bin/python convert/k28_tokenizer_fixtures.py
"""
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, AddedToken, models, normalizers, pre_tokenizers, processors
from tokenizers.pre_tokenizers import ByteLevel

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/hf-tokenizers"

TEXTS = [
    "hello world", "  hello   world  ", "hello, world!", "héllo wörld", "áb é", "日本語 text",
    "emoji 🚀 here", "tab\there\nnewline", "", " ", "<mask> here", "a <mask> b", "a  <mask>  b",
    "<mask><mask>", "x<s>y", "</s> end", "ab" * 40, " nbsp　ideo", "hello hello hello",
    "unknown ☃ char", "ﬁne ² x",
]
PAIRS = [("hello world", "hello"), ("a <mask> b", "é x"), ("hello hello hello hello", "world world world")]


def byte_chars():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def bpe_bytelevel(post, normalizer=None, add_prefix_space=False, mask_normalized=False):
    bc = byte_chars()
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    vocab = {t: i for i, t in enumerate(specials)}
    for b in range(256):
        vocab.setdefault(bc[b], len(vocab))
    merges = [("Ġ", "h"), ("e", "l"), ("el", "l"), ("ell", "o"), ("Ġh", "ello"), ("o", "r"),
              ("w", "or"), ("wor", "l"), ("worl", "d"), ("Ġ", "w"), ("Ġw", "or"), ("Ġwor", "l"), ("Ġworl", "d"),
              ("Ã", "©"), ("Ġ", "Ġ")]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    tk = Tokenizer(models.BPE(vocab, merges))
    if normalizer is not None:
        tk.normalizer = normalizer
    tk.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space, trim_offsets=True)
    tk.post_processor = post
    tk.add_special_tokens([AddedToken(t, special=True) for t in specials if t != "<mask>"])
    tk.add_special_tokens([AddedToken("<mask>", special=True, lstrip=True, normalized=mask_normalized)])
    return tk


def unigram_metaspace():
    pieces = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁hello", -3.0), ("▁world", -3.2),
              ("▁", -2.0), ("h", -4.0), ("e", -4.0), ("l", -4.0), ("o", -4.0), ("w", -4.0),
              ("r", -4.0), ("d", -4.0), ("▁h", -3.5), ("ello", -3.1), ("wor", -3.6), ("ld", -3.7),
              ("!", -5.0), (",", -5.0), ("a", -4.5), ("b", -4.5), ("<0xE2>", -9.0), ("<0x98>", -9.0),
              ("<0x83>", -9.0)]
    tk = Tokenizer(models.Unigram(pieces, unk_id=2, byte_fallback=True))
    tk.normalizer = normalizers.Sequence([
        normalizers.Strip(True, True),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Replace("hashtag", " hashtag "),
    ])
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tk.post_processor = processors.TemplateProcessing(
        single="</s> $A </s>", pair="</s> $A </s> </s> $B </s>", special_tokens=[("</s>", 1)])
    tk.add_special_tokens([AddedToken("<pad>", special=True), AddedToken("</s>", special=True)])
    tk.add_special_tokens([AddedToken("<mask>", special=True, lstrip=True)])
    return tk


def bpe_metaspace():
    specials = ["<s>", "</s>", "<unk>"]
    base = ["▁", "h", "e", "l", "o", "w", "r", "d", "a", "b"]
    vocab = {t: i for i, t in enumerate(specials + base)}
    merges = [("▁", "h"), ("e", "l"), ("el", "l"), ("ell", "o"), ("▁h", "ello")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    for b in (0xE2, 0x98, 0x83):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    tk = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tk.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(" ", "▁")])
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=False)
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1", special_tokens=[("<s>", 0), ("</s>", 1)])
    tk.add_special_tokens([AddedToken(t, special=True) for t in specials])
    return tk


def expected(tk, first, second, length, truncation):
    tk.no_padding()
    tk.enable_truncation(max_length=length, strategy=truncation)
    try:
        e = tk.encode(first, second)
    except Exception as exc:  # truncation errors
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "ids": e.ids, "type_ids": e.type_ids, "offsets": [list(o) for o in e.offsets],
            "word_ids": [-1 if w is None else w for w in e.word_ids],
            "sequence_ids": [-1 if s is None else s for s in e.sequence_ids]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roberta = processors.RobertaProcessing(("</s>", 2), ("<s>", 0), trim_offsets=True, add_prefix_space=False)
    tokenizers = {
        "bpe-roberta": bpe_bytelevel(roberta),
        "bpe-roberta-prefix": bpe_bytelevel(
            processors.RobertaProcessing(("</s>", 2), ("<s>", 0), trim_offsets=True, add_prefix_space=True),
            add_prefix_space=True),
        "bpe-nfc-bytelevel-post": bpe_bytelevel(processors.ByteLevel(trim_offsets=True),
                                                normalizers.NFC(), mask_normalized=True),
        "unigram-metaspace": unigram_metaspace(),
        "bpe-metaspace": bpe_metaspace(),
    }
    for name, tk in tokenizers.items():
        tk.save(str(OUT / f"{name}.json"), pretty=True)
        cases = []
        for text in TEXTS:
            for length in (64, 6):
                cases.append({"first": text, "second": None, "max_length": length, "truncation": "longest_first",
                              **expected(tk, text, None, length, "longest_first")})
        for a, b in PAIRS:
            for length, strategy in ((64, "longest_first"), (9, "longest_first"), (9, "only_second")):
                cases.append({"first": a, "second": b, "max_length": length, "truncation": strategy,
                              **expected(tk, a, b, length, strategy)})
        (OUT / f"{name}.expected.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")
        print(name, len(cases), sum(1 for c in cases if not c["ok"]), "errors")


if __name__ == "__main__":
    main()
