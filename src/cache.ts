// Merge-keyed result cache and batch helpers (K16). The merge key covers
// the token ids plus every schema field that influences logits; identical
// keys always produce identical logits, so a hit returns the stored arrays.
// The cache is a fixed-size LRU and can be disabled with size 0.

import type { SchemaInput } from './tokenizer/schema.ts';
import type { JuliaPreparedInput } from './julia.ts';

export const BATCH_SIZES = [1, 4, 8, 16] as const;
export const MAX_BATCH = BATCH_SIZES[BATCH_SIZES.length - 1];

// Smallest supported batch size covering n rows; > 16 must be chunked by
// the caller (batch plans exist only for these sizes).
export function nextBatchSize(n: number): number {
  for (const b of BATCH_SIZES) {
    if (n <= b) return b;
  }
  throw new Error(`batch of ${n} exceeds the maximum of ${MAX_BATCH}`);
}

export class LruCache<V> {
  private map = new Map<string, V>();

  private maxEntries: number;

  constructor(maxEntries: number) {
    this.maxEntries = maxEntries;
  }

  get size(): number {
    return this.map.size;
  }

  get enabled(): boolean {
    return this.maxEntries > 0;
  }

  // Hits refresh recency; misses return undefined.
  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v === undefined) return undefined;
    this.map.delete(key);
    this.map.set(key, v);
    return v;
  }

  set(key: string, value: V): void {
    if (this.maxEntries <= 0) return;
    if (this.map.has(key)) this.map.delete(key);
    else if (this.map.size >= this.maxEntries) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
    this.map.set(key, value);
  }

  clear(): void {
    this.map.clear();
  }
}

// Schema routing (indices/mask/groups) plus the real tokens determine the
// logits; the bucket enters separately because a call may force a larger
// bucket than the smallest fit (logits stay equal, the key stays strict).
export function schemaMergeKey(input: SchemaInput): string {
  return `${input.inputIds.subarray(0, input.seqLen).join(',')}`
    + `|${input.markerIndices.join(',')}`
    + `|${input.markerMask.join(',')}`
    + `|${input.markerGroups.join(',')}`;
}

export function juliaMergeKey(input: JuliaPreparedInput): string {
  return `${input.qtype}|${input.inputIds.subarray(0, input.seqLen).join(',')}`
    + `|${input.markers.join(',')}`;
}

// Row stride inside a batch plan: sequence b occupies rows
// [b*stride, b*stride + seqLen). Quantizing stride to a multiple of 64
// bounds the number of lazily built plans while keeping the padded tail
// per sequence under 64 rows (the single path dispatches seqLen rows).
export function batchStride(maxSeqLen: number): number {
  return Math.ceil(maxSeqLen / 64) * 64;
}

// Sort unique inputs by ascending seqLen before chunking so each batch
// chunk's maxSeqLen stays close to its members; returns the reordered
// unique array and the remapped slots (results still resolve by slot).
export function sortByKey<T>(
  unique: T[], slot: number[], keyOf: (input: T) => number,
): { unique: T[]; slot: number[] } {
  const order = unique.map((_, i) => i).sort(
    (a, b) => keyOf(unique[a]) - keyOf(unique[b]));
  const remap = new Array<number>(unique.length);
  order.forEach((oldIdx, newIdx) => { remap[oldIdx] = newIdx; });
  return {
    unique: order.map((i) => unique[i]),
    slot: slot.map((s) => remap[s]),
  };
}

// First-occurrence dedup by merge key. Returns the unique inputs in stable
// order plus, for every input index, the unique slot it resolves to
// (duplicates point at the earlier entry they match).
export function dedupMerge<T>(
  inputs: T[], keyOf: (input: T) => string,
): { unique: T[]; slot: number[] } {
  const seen = new Map<string, number>();
  const unique: T[] = [];
  const slot: number[] = [];
  for (const input of inputs) {
    const key = keyOf(input);
    let existing = seen.get(key);
    if (existing === undefined) {
      existing = unique.length;
      seen.set(key, existing);
      unique.push(input);
    }
    slot.push(existing);
  }
  return { unique, slot };
}
