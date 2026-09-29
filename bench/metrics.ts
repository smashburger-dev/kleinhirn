// Measurement core for every kleinhirn benchmark. Pure functions: no DOM, no GPU.
//
// CoreML reference values:
//   median = statistics.median (mean of the two middle values for even n)
//   p95    = sorted[floor(0.95 * n)]
// Parity compares logits over the valid labels only; callers trim padding.

export interface LatencySummary {
  n: number;
  medianMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
  meanMs: number;
}

export interface ParitySummary {
  n: number;
  argmaxAgreement: number;
  maxAbsLogitDiff: number;
  meanAbsLogitDiff: number;
  maxAbsProbDiff: number;
  disagreements: number[];
}

const ascending = (values: readonly number[]): number[] => [...values].sort((a, b) => a - b);

export function median(sorted: readonly number[]): number {
  if (!sorted.length) throw new Error('median of empty sample');
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function quantileFloor(sorted: readonly number[], q: number): number {
  if (!sorted.length) throw new Error('quantile of empty sample');
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

export function summarizeLatency(samplesMs: readonly number[]): LatencySummary {
  const sorted = ascending(samplesMs);
  return {
    n: sorted.length,
    medianMs: median(sorted),
    p95Ms: quantileFloor(sorted, 0.95),
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    meanMs: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  };
}

// First index of the maximum, like numpy.argmax.
export function argmax(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}

export function softmax(logits: readonly number[]): number[] {
  const top = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - top));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

// reference[i] and candidate[i] hold the logits of item i over its valid labels.
export function compareLogits(reference: readonly number[][], candidate: readonly number[][]): ParitySummary {
  if (reference.length !== candidate.length) {
    throw new Error(`item count differs: reference ${reference.length}, candidate ${candidate.length}`);
  }
  if (!reference.length) throw new Error('parity of empty sample');
  let agree = 0;
  let maxLogit = 0;
  let sumLogit = 0;
  let countLogit = 0;
  let maxProb = 0;
  const disagreements: number[] = [];
  reference.forEach((ref, i) => {
    const cand = candidate[i];
    if (ref.length !== cand.length) throw new Error(`label count differs at item ${i}: ${ref.length} vs ${cand.length}`);
    if (argmax(ref) === argmax(cand)) agree += 1;
    else disagreements.push(i);
    const refProb = softmax(ref);
    const candProb = softmax(cand);
    ref.forEach((value, j) => {
      const diff = Math.abs(value - cand[j]);
      maxLogit = Math.max(maxLogit, diff);
      sumLogit += diff;
      countLogit += 1;
      maxProb = Math.max(maxProb, Math.abs(refProb[j] - candProb[j]));
    });
  });
  return {
    n: reference.length,
    argmaxAgreement: agree / reference.length,
    maxAbsLogitDiff: maxLogit,
    meanAbsLogitDiff: sumLogit / countLogit,
    maxAbsProbDiff: maxProb,
    disagreements,
  };
}
