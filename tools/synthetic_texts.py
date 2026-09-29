"""Deterministic synthetic corpus: strings we own outright.

Pools: German and English sentences, code lines, numbers and formulas,
emoji and CJK. The output is stable — same code, same lines, no external
input and no randomness beyond a fixed local shuffle.

Usage: python3 tools/synthetic_texts.py          # prints the lines
       python3 tools/synthetic_texts.py --json   # prints a JSON list
"""
import json
import sys

DE_SUBJ = ["Das Modell", "Der Tokenizer", "Die Engine", "Ein Vektor",
           "Die Matrix", "Der Parser", "Das Notebook", "Der Thread",
           "Die Schleife", "Ein Renderer"]
DE_VERB = ["berechnet", "vergleicht", "verwirft", "liefert", "prüft",
           "verkürzt", "sortiert", "bündelt", "liest", "markiert"]
DE_OBJ = ["den Mittelwert der Stichprobe", "die Labels der Anfrage",
          "den letzten Token der Zeile", "den Rang der Matrix",
          "die Spitze des Speichers", "die Differenz zweier Tensoren",
          "die Maske der Aufmerksamkeit", "den Fehler der Vorhersage",
          "die Liste der Kandidaten", "den Ausgang des Vergleichs"]

EN_SUBJ = ["The encoder", "A worker", "The tokenizer", "The scheduler",
           "A cursor", "The decoder", "The cache", "A shader",
           "The sampler", "The pipeline"]
EN_VERB = ["returns", "discards", "measures", "compares", "clamps",
           "tokenizes", "sorts", "reads", "marks", "joins"]
EN_OBJ = ["the smallest positive margin", "every padded position",
          "the label with the highest logit", "a window of 128 tokens",
          "the first non-empty bucket", "the mean of two runs",
          "all rows beyond the threshold", "the last pending request",
          "the shared staging buffer", "a null-terminated span"]

CODE_T = [
    "def f{n}(xs):\n    return sum(x * {k} for x in xs) / max(len(xs), 1)",
    "const s{n} = xs.filter((x) => x > {k}).map((x) => x * x);",
    "for i in range({k}):\n    acc += weights[i] * inputs[i]",
    "mask = (scores > {k}) & (counts < {m})",
    "assert len(folds) == {k}, folds",
    "return {{'label': labels[argmax(logits)], 'score': float(probs[{k}])}}",
    "rng = np.random.default_rng({k})\nperm = rng.permutation(n)",
    "while queue and steps < {k}:\n    node = queue.popleft()",
    "dict(zip(keys[:{k}], values))",
    "if common == 0:\n    return 0.0",
    "yield (chunk_start, text[chunk_start:chunk_start + {k}])",
    "raise ValueError(f'k must be between 1 and {m}')",
    "sorted(items, key=lambda p: (-p[{k}], p[0]))",
    "np.clip(grad, -{k}.0, {k}.0)",
    "result = a_{n} @ b_{n} + c_{n}",
]

NUM_T = [
    "Der Wert liegt bei {v} Prozent, die Schranke bei {w} Prozent.",
    "The ratio is {v} out of {w}, rounded to {r}.",
    "Version {v}.{w}.{r} bricht den Vertrag der Schnittstelle.",
    "Seed {v} liefert {w} Zeilen, davon {r} dupliziert.",
    "Ein Batch von {v} Anfragen dauerte {w}.{r} ms.",
    "Der Fehler beträgt {v}e-{w} bei einer Schranke von {r}e-{v}.",
    "{v} von {w} Token wurden als [UNK] markiert ({r} %).",
    "p = 0.{v}, n = {w}, Erwartungswert {r}.",
]

EMOJI_T = [
    "Das Ergebnis {a} wurde mit {b} bestätigt — alles {c}.",
    "Status: {a} ok, {b} wartet, {c} fehlgeschlagen.",
    "{a} Testlauf {n}: {b} Parität, {c} Speicher.",
]

EMOJI = ["✓", "✗", "🎯", "🚀", "👍", "⚠️", "🔥", "📊", "🧪", "🛠️"]

CJK = [
    "她在考试中得了最高分，因此获得了奖励。",
    "这个模型在浏览器中运行，不需要服务器。",
    "向量 u 和 v 的点积是 -47。",
    "彼女は試験で最高点を取ったので、賞をもらった。",
    "トークナイザーはテキストを ID に変換する。",
    "行列のランクは2であり、4ではない。",
    "그녀는 시험에서 가장 높은 점수를 받았기 때문에 상을 받았다.",
    "한국어 텍스트도 토크나이저가 올바르게 처리해야 한다.",
    "모델은 브라우저에서 실행되며 서버가 필요 없다.",
    "矩阵的秩是2，不是4。",
]

MATH_T = [
    "$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$ mit $a = {v}$",
    "Es gilt $u^\\top v = {v}$ und $\\|u\\|_2 = \\sqrt{{{w}}}$.",
    "Die Softmax-Funktion hebt das Maximum $e^{{x_i - m}}$ hervor.",
    "RMSE $= \\sqrt{{\\frac{{1}}{{n}} \\sum_i (\\hat{{y}}_i - y_i)^2}}$",
    "$\\alpha = {v}$, $\\beta = \\pi^{{e}}$, $\\gamma \\geq {w}$",
]


def synthetic_texts() -> list[str]:
    lines: list[str] = []
    # Pools first: fixed language samples, then combinatorial fills.
    lines.extend(CJK)
    for i in range(90):
        lines.append(f"{DE_SUBJ[i % 10]} {DE_VERB[i // 10 % 10]} "
                     f"{DE_OBJ[i // 10]} in Schritt {i % 7 + 1}.")
    for i in range(90):
        lines.append(f"{EN_SUBJ[i % 10]} {EN_VERB[i // 10 % 10]} "
                     f"{EN_OBJ[i // 10]} during pass {i % 7 + 1}.")
    for i in range(120):
        lines.append(CODE_T[i % len(CODE_T)].format(
            n=i, k=i % 9 + 1, m=i % 13 + 2))
    for i in range(80):
        lines.append(NUM_T[i % len(NUM_T)].format(
            v=(i * 37) % 97 + 3, w=(i * 53) % 89 + 11, r=(i * 29) % 79 + 5))
    for i in range(40):
        lines.append(EMOJI_T[i % len(EMOJI_T)].format(
            a=EMOJI[i % 10], b=EMOJI[(i + 3) % 10], c=EMOJI[(i + 7) % 10],
            n=i + 1))
    for i in range(40):
        lines.append(MATH_T[i % len(MATH_T)].format(v=i % 9 + 2, w=i % 12 + 4))
    for i in range(30):
        lines.append(f"{CJK[i % len(CJK)]} (Beispiel {i + 1})")
    # Dedup preserving order, then pad with indexed variants if needed.
    seen: set[str] = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    extra = 0
    while len(out) < 500:
        line = f"Synthetischer Satz {extra}: der Zähler steht auf {extra * 3}."
        extra += 1
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:500]


if __name__ == "__main__":
    texts = synthetic_texts()
    if "--json" in sys.argv:
        print(json.dumps(texts, ensure_ascii=False, indent=1))
    else:
        print("\n".join(texts))
