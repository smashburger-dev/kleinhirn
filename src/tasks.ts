// Small pure helpers on encoder outputs, no GPU (softmax, L2 norm, span
// grouping); the text methods of EncoderModel use them.

// A NaN or Inf in a readback is a failed or overflowed run, not a score:
// throw before softmax, argmax or a cache can turn it into a plausible answer.
export function assertFinite(values: ArrayLike<number>, what: string): void {
  let bad = 0;
  for (let i = 0; i < values.length; i += 1) if (!Number.isFinite(values[i])) bad += 1;
  if (bad) throw new Error(`${what}: ${bad} of ${values.length} values are not finite`);
}

// Start of the word row of a token id in a table of `length` values and rows of `width`: the id
// must be an integer inside the vocabulary (a fraction would gather halves of two rows, a
// too large id an empty row).
export function tokenRowOffset(id: number, width: number, length: number): number {
  if (!Number.isInteger(id) || id < 0 || (id + 1) * width > length) {
    throw new Error(`token id ${id} is outside the vocabulary of ${length / width} rows (ids are integers from 0)`);
  }
  return id * width;
}

export function softmax(logits: ArrayLike<number>): Float32Array<ArrayBuffer> {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) max = Math.max(max, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= sum;
  return out;
}

export function l2normalize(v: ArrayLike<number>): Float32Array<ArrayBuffer> {
  let sq = 0;
  for (let i = 0; i < v.length; i += 1) sq += v[i] * v[i];
  const norm = Math.max(Math.sqrt(sq), 1e-12);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i += 1) out[i] = v[i] / norm;
  return out;
}

// Zero-shot over NLI: one entailment logit per label (the pair of text and
// hypothesis), softmax over the labels. Returns the index of the chosen
// label and the probabilities.
export function zeroShot(
  entailmentLogits: ArrayLike<number>,
): { choice: number; probabilities: Float32Array<ArrayBuffer> } {
  const probabilities = softmax(entailmentLogits);
  return { choice: argmaxRows(probabilities, probabilities.length)[0], probabilities };
}

// Label names in class order from the manifest's id -> name map. Ids must be integers from 0 to
// classes - 1; missing ones become LABEL_i (the default names of transformers). More labels than
// classes, or an id outside, throws.
export function completeLabels(labels: Record<string, string> | undefined, classes: number): string[] {
  const out = Array.from({ length: classes }, (_, i) => `LABEL_${i}`);
  for (const [key, name] of Object.entries(labels ?? {})) {
    const id = Number(key);
    if (!Number.isInteger(id) || String(id) !== key || id < 0 || id >= classes) {
      throw new Error(`label id '${key}' is not an integer from 0 to ${classes - 1}`);
    }
    out[id] = name;
  }
  return out;
}

// Argmax per row of a row-major [rows, cols] array. Ties go to the lower index.
export function argmaxRows(data: ArrayLike<number>, cols: number): number[] {
  if (!Number.isInteger(cols) || cols <= 0) throw new Error(`argmaxRows needs a positive integer column count, got ${cols}`);
  const rows = Math.floor(data.length / cols);
  const out: number[] = [];
  for (let r = 0; r < rows; r += 1) {
    let best = 0;
    for (let c = 1; c < cols; c += 1) {
      if (data[r * cols + c] > data[r * cols + best]) best = c;
    }
    out.push(best);
  }
  return out;
}

export interface EntitySpan {
  // label without its B- or I- prefix
  group: string;
  // mean of the token probabilities
  score: number;
  // UTF-16 offsets into the input text
  start: number;
  end: number;
}

function splitTag(name: string): { begin: boolean; tag: string } {
  if (name.startsWith('B-')) return { begin: true, tag: name.slice(2) };
  if (name.startsWith('I-')) return { begin: false, tag: name.slice(2) };
  return { begin: false, tag: name };
}

// Token classification spans like transformers' TokenClassificationPipeline
// with aggregation_strategy="simple": special tokens drop out, every other
// token takes its argmax label and that label's probability, neighbours with
// the same tag join unless the next one starts with B-, and groups labelled
// "O" are removed. probs is row-major [rows, labels.length].
export function aggregateSimple(
  probs: ArrayLike<number>, labels: string[], offsets: ArrayLike<readonly [number, number]>,
  specialMask: ArrayLike<number>, ignore: string[] = ['O'],
): EntitySpan[] {
  const cols = labels.length;
  const tokens: { label: string; score: number; start: number; end: number }[] = [];
  const best = argmaxRows(probs, cols);
  for (let i = 0; i < best.length; i += 1) {
    if (specialMask[i]) continue;
    tokens.push({
      label: labels[best[i]], score: probs[i * cols + best[i]],
      start: offsets[i][0], end: offsets[i][1] });
  }
  const groups: EntitySpan[] = [];
  let run: typeof tokens = [];
  const flush = () => {
    if (!run.length) return;
    const name = run[0].label;
    const dash = name.indexOf('-');
    let sum = 0;
    for (const t of run) sum += t.score;
    groups.push({
      group: dash < 0 ? name : name.slice(dash + 1), score: sum / run.length,
      start: run[0].start, end: run[run.length - 1].end });
  };
  for (const t of tokens) {
    if (run.length) {
      const cur = splitTag(t.label);
      const last = splitTag(run[run.length - 1].label);
      if (!(cur.tag === last.tag && !cur.begin)) {
        flush();
        run = [];
      }
    }
    run.push(t);
  }
  flush();
  return groups.filter((g) => !ignore.includes(g.group));
}
