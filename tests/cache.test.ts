// K16 cache unit tests: LRU hit/eviction/disabled behaviour, merge keys
// for both engines, and in-batch dedup. CPU-only, no model files needed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LruCache, batchStride, dedupMerge, juliaMergeKey, nextBatchSize,
  schemaMergeKey, sortByKey,
} from '../src/cache.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';

function schemaInput(ids: number[], markers: number[] = [1]): SchemaInput {
  const inputIds = new Int32Array(128);
  inputIds.set(ids);
  const markerIndices = new Int32Array(16);
  markerIndices.set(markers);
  const markerMask = new Float32Array(16);
  markerMask.fill(1, 0, markers.length);
  const markerGroups = new Int32Array(16);
  return {
    inputIds, attentionMask: new Int32Array(128),
    markerIndices, markerMask, markerGroups, seqLen: ids.length,
  };
}

function juliaInput(ids: number[], markers: number[], qtype = 0): JuliaPreparedInput {
  return {
    inputIds: Int32Array.from(ids), markers, qtype, seqLen: ids.length,
  };
}

test('LruCache: set/get hit', () => {
  const c = new LruCache<number>(4);
  c.set('a', 1);
  assert.equal(c.get('a'), 1);
  assert.equal(c.get('b'), undefined);
});

test('LruCache: evicts least recently used at capacity', () => {
  const c = new LruCache<number>(2);
  c.set('a', 1);
  c.set('b', 2);
  c.get('a'); // refresh a; b is now oldest
  c.set('c', 3);
  assert.equal(c.get('b'), undefined);
  assert.equal(c.get('a'), 1);
  assert.equal(c.get('c'), 3);
  c.set('d', 4); // capacity 2: a is now oldest
  assert.equal(c.get('a'), undefined);
  assert.equal(c.size, 2);
});

test('LruCache: size 0 disables', () => {
  const c = new LruCache<number>(0);
  assert.equal(c.enabled, false);
  c.set('a', 1);
  assert.equal(c.get('a'), undefined);
  assert.equal(c.size, 0);
});

test('schemaMergeKey: identical token ids plus schema collide', () => {
  const a = schemaInput([1, 2, 3], [1, 2]);
  const b = schemaInput([1, 2, 3], [1, 2]);
  assert.equal(schemaMergeKey(a), schemaMergeKey(b));
  const otherIds = schemaInput([1, 2, 4], [1, 2]);
  assert.notEqual(schemaMergeKey(a), schemaMergeKey(otherIds));
  const otherMarkers = schemaInput([1, 2, 3], [1, 3]);
  assert.notEqual(schemaMergeKey(a), schemaMergeKey(otherMarkers));
  const longer = schemaInput([1, 2, 3, 4], [1, 2]);
  assert.notEqual(schemaMergeKey(a), schemaMergeKey(longer));
});

test('juliaMergeKey: qtype and markers distinguish', () => {
  const a = juliaInput([1, 2, 3], [5, 9], 0);
  assert.equal(juliaMergeKey(a), juliaMergeKey(juliaInput([1, 2, 3], [5, 9], 0)));
  assert.notEqual(
    juliaMergeKey(a), juliaMergeKey(juliaInput([1, 2, 3], [5, 9], 1)));
  assert.notEqual(
    juliaMergeKey(a), juliaMergeKey(juliaInput([1, 2, 3], [5, 10], 0)));
});

test('dedupMerge: duplicates in one batch map to the first occurrence', () => {
  const a = schemaInput([1, 2, 3]);
  const b = schemaInput([9, 9]);
  const a2 = schemaInput([1, 2, 3]);
  const a3 = schemaInput([1, 2, 3]);
  const { unique, slot } = dedupMerge([a, b, a2, a3], schemaMergeKey);
  assert.equal(unique.length, 2);
  assert.equal(unique[0], a);
  assert.equal(unique[1], b);
  assert.deepEqual(slot, [0, 1, 0, 0]);
});

test('nextBatchSize: supported sizes, overflow throws', () => {
  assert.equal(nextBatchSize(1), 1);
  assert.equal(nextBatchSize(3), 4);
  assert.equal(nextBatchSize(4), 4);
  assert.equal(nextBatchSize(5), 8);
  assert.equal(nextBatchSize(9), 16);
  assert.equal(nextBatchSize(16), 16);
  assert.throws(() => nextBatchSize(17));
});

test('batchStride: 64-quantized, never below maxSeqLen', () => {
  assert.equal(batchStride(1), 64);
  assert.equal(batchStride(63), 64);
  assert.equal(batchStride(64), 64);
  assert.equal(batchStride(65), 128);
  assert.equal(batchStride(258), 320);
  assert.equal(batchStride(512), 512);
  assert.equal(batchStride(513), 576);
  assert.equal(batchStride(1024), 1024);
});

test('sortByKey: reorders uniques and remaps slots', () => {
  const inputs = [
    schemaInput([1, 2, 3, 4, 5]),   // seqLen 5
    schemaInput([7]),               // seqLen 1
    schemaInput([1, 2, 3, 4, 5]),   // dup of first
    schemaInput([9, 9, 9]),         // seqLen 3
  ];
  const { unique, slot } = dedupMerge(inputs, schemaMergeKey);
  const sorted = sortByKey(unique, slot, (i) => i.seqLen);
  assert.deepEqual(sorted.unique.map((i) => i.seqLen), [1, 3, 5]);
  assert.deepEqual(sorted.slot, [2, 0, 2, 1]);
  // Results resolve to the same input objects as before sorting.
  for (const [i, input] of inputs.entries()) {
    assert.equal(
      sorted.unique[sorted.slot[i]].inputIds.subarray(0, input.seqLen)
        .join(','),
      input.inputIds.subarray(0, input.seqLen).join(','));
  }
});
