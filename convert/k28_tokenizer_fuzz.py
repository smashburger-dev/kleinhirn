"""Fuzz goldens for the K28.3b tokenizer: tokenizer.json through the tokenizers library.

For every distinct tokenizer.json of the 56 non-WordPiece models (bench/results/k28-tokenizers-nonwordpiece.json)
and of the 32 WordPiece models (k28-tokenizers-wordpiece.json)
this encodes a fixed seed's worth of hand-made and random strings (special tokens with every spacing,
all White_Space chars, combining marks, Hangul, emoji with ZWJ, compatibility forms, controls, long
words), single and as pairs, without truncation, and writes the results to
models/k28/_tokenizer-fuzz/<sha256>.json (not in git, models/ is ignored). tools/k28_tokenizer_check.ts
--fuzz compares the TypeScript tokenizer with them. A second file per tokenizer with a Precompiled or
NFC/NFKC/BertNormalizer normalizer holds normalize_str of every Unicode scalar value that the normalizer changes
(models/k28/_tokenizer-fuzz/<sha256>.normalize.json). compose-probe.json holds NFC and NFKC of random strings with the alignment of every char. pretok.json holds the words of
BertPreTokenizer, ByteLevel (add_prefix_space false) and WhitespaceSplit on "a" + scalar value + "b" for every
scalar value that splits the string.

Usage: .venv-k28/bin/python convert/k28_tokenizer_fuzz.py
"""
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k28_common import K28_DIR, ROOT, slug  # noqa: E402

OUT = K28_DIR / "_tokenizer-fuzz"
N_RANDOM = 1500
SPACES = [" ", " ", " ", "　", "\t", "\n", "\r", " ", "​", "﻿", "\u0085",
          " ", " ", " ", "\x0b", "\x0c"]
POOLS = {
    "ascii": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?()-_/'\"@#$%&*+=<>[]{}|\\^~`"),
    "latin": list("äöüßéèêëàâçñøåæœÿÄÖÜÉİıŞşĞğ"),
    "marks": [chr(c) for c in (0x300, 0x301, 0x308, 0x323, 0x327, 0x483, 0x64b, 0xe31, 0x20d0, 0x93c, 0x94d)],
    "cjk": list("日本語中文字漢語한국어テスト한글") + [chr(c) for c in (0x1100, 0x1161, 0x11a8, 0xac00, 0xd7a3, 0x3131)],
    "emoji": ["🚀", "😀", "👨‍👩‍👧", "👍🏽", "🇩🇪", "🇺🇸", "❤️", "‍", "️", "\U0001f3fb", "☃", "✓"],
    "compat": ["ﬁ", "ﬂ", "ﬀ", "²", "³", "①", "™", "…", "１２３", "ＡＢＣ", "㎏", "ǆ", "Ǆ", "½", "Å", "Å", "ẛ", "ʼn"],
    "scripts": list("абвгдежзийклмнопрстуфхцчшщыьэюяΑΒΓΔαβγδ") + list("أبجدهوزحطي") + list("कखगघचछजझ") + list("กขคงจฉช"),
    "control": ["\x00", "\x01", "\x07", "\x1f", "\x7f", "\x80", "\x9f", "­", "‎", "‏", "�", ""],
    "marker": ["▁", "Ġ", "Ċ", "▁▁", "Ġ"],
    "space": SPACES,
}
KEYS = list(POOLS)
WEIGHTS = [10, 3, 2, 2, 2, 2, 2, 1, 1, 6]


def special_tokens(tj):
    return [t["content"] for t in tj["added_tokens"]][:12]


def crafted(tj):
    out = ["", " ", "  ", "a", "a b", " a b ", "a  b", "\n", "a\nb", " a ", "x" * 300, "ab " * 100]
    for t in special_tokens(tj):
        out += [t, " " + t, t + " ", "a " + t + " b", "a  " + t + "  b", t + t, "x" + t + "y", "\n" + t + "\n",
                "a" + t, t + "a", t + " " + t, "　" + t, t + " b"]
    for sp in SPACES:
        out += ["a" + sp + "b", sp + "a", "a" + sp, sp * 3 + "ab" + sp * 2]
    out += ["é", "ẹ́", "각", "한국어 텍스트", "ﬁne ﬂow", "m²/s", "x™", "①②③",
            "Hello, World! 🚀 It's 10:30...", "http://url", "hashtag #url @usuario", "hola @usuario!", "x url y",
            "á@usuarió", "_@usuario_"]
    return out


def random_text(rng):
    words = []
    for _ in range(rng.randint(1, 8)):
        w = "".join(rng.choices([c for k in KEYS for c in POOLS[k]][:0] or [rng.choice(POOLS[rng.choices(KEYS, WEIGHTS)[0]])
                                                                               for _ in range(rng.randint(1, 6))], k=1))
        for _ in range(rng.randint(0, 5)):
            w += rng.choice(POOLS[rng.choices(KEYS, WEIGHTS)[0]])
        words.append(w)
    sep = rng.choice([" ", " ", " ", "  ", "\t", "\n", " ", ""])
    text = sep.join(words)
    if rng.random() < 0.15:
        text = " " + text
    if rng.random() < 0.15:
        text += " "
    return text


def result(tk, first, second):
    e = tk.encode(first, second)
    return {"first": first, "second": second, "ids": e.ids, "type_ids": e.type_ids,
            "attention_mask": e.attention_mask, "offsets": [list(o) for o in e.offsets],
            "word_ids": [-1 if w is None else w for w in e.word_ids],
            "sequence_ids": [-1 if s is None else s for s in e.sequence_ids]}


BOUNDARY = list("aZ09_ .,-/@#\t\n") + ["\u00e9", "e\u0301", "\u0301", "\u00b2", "\u2581", "\u3000", "\u00a0", "\u0085", "\u200b",
                                      "\u65e5", "\U0001f680", "\u0e31", "\u094d", "\u2013", "\u00ad", "\ufeff"]
REPLACE_WORDS = ["url", "@usuario", "hashtag", "http://url", "  ", "   "]


def replace_probe(tk):
    """normalize_str of every word between every pair of boundary chars (Replace patterns with \\W)."""
    out = {}
    for w in REPLACE_WORDS:
        for left in BOUNDARY + [""]:
            for right in BOUNDARY + [""]:
                text = left + w + right
                out[text] = tk.normalizer.normalize_str(text)
    return out


def compose_probe():
    """NFC and NFKC with alignment: 40000 random strings of chars with a decomposition or a combining class.

    A WordLevel model with a Split pre-tokenizer on every char turns the offsets of each token into the
    alignment of the normalized char; normalized text and offsets (code points) are stored.
    """
    import unicodedata
    from tokenizers import Regex, models, normalizers, pre_tokenizers
    rel = [cp for cp in range(0x110000) if not 0xD800 <= cp < 0xE000 and (
        unicodedata.combining(chr(cp)) or unicodedata.normalize("NFKD", chr(cp)) != chr(cp)
        or unicodedata.normalize("NFC", chr(cp)) != chr(cp))]
    basic = [ord(c) for c in "abcdeo AEOuUiIxyzNn.,"]
    rng = random.Random(11)
    texts = ["".join(chr(rng.choice(rel) if rng.random() < 0.6 else rng.choice(basic))
                     for _ in range(rng.randint(1, 7))) for _ in range(40000)]
    out = {"texts": texts}
    for name, norm in (("NFC", normalizers.NFC()), ("NFKC", normalizers.NFKC())):
        tk = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tk.normalizer = norm
        tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
        tk.no_truncation()
        res = []
        for t in texts:
            e = tk.encode(t)
            res.append([norm.normalize_str(t), [list(o) for o in e.offsets]])
        out[name] = res
    (OUT / "compose-probe.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print("compose probe", len(texts), "strings", flush=True)


def sweep_pretok():
    from tokenizers import pre_tokenizers
    pts = {"bert": pre_tokenizers.BertPreTokenizer(),
           "bytelevel": pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
           "whitespace": pre_tokenizers.WhitespaceSplit()}
    out = {}
    for name, pt in pts.items():
        changed = {}
        for cp in range(0x110000):
            if 0xD800 <= cp < 0xE000:
                continue
            text = "a" + chr(cp) + "b"
            words = [w for w, _ in pt.pre_tokenize_str(text)]
            if len(words) != 1:
                changed[cp] = words
        out[name] = changed
    (OUT / "pretok.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print("pretok sweep", {k: len(v) for k, v in out.items()}, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_sha = {}
    for kind in ("nonwordpiece", "wordpiece"):
        by_sha.update(json.loads((ROOT / f"bench/results/k28-tokenizers-{kind}.json").read_text())["bySha256"])
    sweep_pretok()
    compose_probe()
    for sha, models in by_sha.items():
        d = K28_DIR / slug(models[0])
        path = d / "tokenizer.json" if (d / "tokenizer.json").exists() else d / "repo/tokenizer.json"
        tj = json.loads(path.read_text())
        tk = Tokenizer.from_file(str(path))
        tk.no_truncation()
        tk.no_padding()
        rng = random.Random(int(sha[:8], 16))
        texts = crafted(tj) + [random_text(rng) for _ in range(N_RANDOM)]
        cases = [result(tk, t, None) for t in texts]
        for i in range(0, len(texts) - 1, 7):
            cases.append(result(tk, texts[i], texts[i + 1]))
        (OUT / f"{sha}.json").write_text(json.dumps({"models": models, "cases": cases}, ensure_ascii=False) + "\n")
        chain = json.dumps(tj["normalizer"])
        if any(t in chain for t in ("Precompiled", '"NFC"', '"NFKC"', "BertNormalizer")):
            changed = {}
            for cp in range(0x110000):
                if 0xD800 <= cp < 0xE000:
                    continue
                c = chr(cp)
                n = tk.normalizer.normalize_str(c)
                if n != c:
                    changed[cp] = n
            (OUT / f"{sha}.normalize.json").write_text(json.dumps(changed, ensure_ascii=False) + "\n")
        if '"Replace"' in chain and "Regex" in chain:
            (OUT / f"{sha}.replace.json").write_text(json.dumps(replace_probe(tk), ensure_ascii=False) + "\n")
        print(sha[:8], models[0], len(cases), "cases", flush=True)


if __name__ == "__main__":
    main()
