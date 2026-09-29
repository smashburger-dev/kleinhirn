// K27 dispatch-profile aggregation shared by bench/kleinhirn.ts and
// bench/julia-bench.ts. A row is one profiled forward: one time (ms) per
// dispatch, same order in every row (names).

import { summarizeLatency } from './metrics.ts';

export interface DispatchRow {
  seqLen: number;
  times: Record<string, number>;
}

export interface DispatchProfile {
  granularity: 'dispatch';
  warmup: number;
  items: number;
  names: string[];
  seqLens: number[];
  rowsMs: number[][];
  byOp: Record<string, { medianMs: number; p95Ms: number; count: number }>;
  gpuSumMedianMs: number;
  gpuSumMs: number[];
  wallMedianMs: number;
  wallMs: number[];
  paritySample: { items: number; bitIdentical: number };
  quantization: {
    valuesTotal: number;
    distinctValues: number;
    quantumMs: number;
    allMultiplesOfQuantum: boolean;
    quantized: boolean;
  };
}

const LAYER_KEY = /^L\d+\./;

const round5 = (v: number) => Number(v.toFixed(5));

export function bitIdentical(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return ua.every((v, i) => v === ub[i]);
}

export function buildDispatchProfile(
  rows: DispatchRow[], wallMs: number[], warmup: number,
  paritySample: { items: number; bitIdentical: number },
): DispatchProfile {
  const names = Object.keys(rows[0].times);
  const rowsMs = rows.map((r) => names.map((n) => round5(r.times[n])));
  const summed = rows.map((r) => {
    const byOp = new Map<string, number>();
    for (const n of names) {
      const key = n.replace(LAYER_KEY, '');
      byOp.set(key, (byOp.get(key) ?? 0) + r.times[n]);
    }
    return byOp;
  });
  const ops = [...summed[0].keys()];
  const byOp: DispatchProfile['byOp'] = {};
  for (const op of ops) {
    const s = summarizeLatency(summed.map((m) => m.get(op) as number));
    byOp[op] = {
      medianMs: round5(s.medianMs), p95Ms: round5(s.p95Ms),
      count: names.filter((n) => n.replace(LAYER_KEY, '') === op).length,
    };
  }
  const gpuSumMs = rows.map((r) => round5(names.reduce((a, n) => a + r.times[n], 0)));
  const all = rows.flatMap((r) => names.map((n) => r.times[n]));
  const positive = all.filter((v) => v > 0);
  const quantum = positive.length ? Math.min(...positive) : 0;
  const onGrid = quantum > 0
    && all.every((v) => Math.abs(v / quantum - Math.round(v / quantum)) < 0.01);
  return {
    granularity: 'dispatch', warmup, items: rows.length, names,
    seqLens: rows.map((r) => r.seqLen), rowsMs, byOp,
    gpuSumMedianMs: round5(summarizeLatency(gpuSumMs).medianMs), gpuSumMs,
    wallMedianMs: round5(summarizeLatency(wallMs).medianMs),
    wallMs: wallMs.map(round5), paritySample,
    quantization: {
      valuesTotal: all.length,
      distinctValues: new Set(all.map(round5)).size,
      quantumMs: round5(quantum),
      allMultiplesOfQuantum: onGrid,
      // Unquantized timestamps have nanosecond granularity: the smallest
      // nonzero value is then far below 10 us and not a common divisor.
      quantized: onGrid && quantum >= 0.01,
    },
  };
}
