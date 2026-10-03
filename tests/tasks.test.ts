import { test } from 'node:test';
import assert from 'node:assert/strict';
import { aggregateSimple, argmaxRows, l2normalize, softmax, zeroShot } from '../src/tasks.ts';

test('softmax sums to one and keeps order', () => {
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 1e-6);
  assert.ok(p[0] < p[1] && p[1] < p[2]);
  assert.ok(Math.abs(p[2] - 0.6652409557748219) < 1e-6);
});

test('softmax survives large logits', () => {
  const p = softmax([1000, 1001]);
  assert.ok(Number.isFinite(p[0]) && Math.abs(p[1] - 0.7310585786300049) < 1e-6);
});

test('l2normalize gives unit length', () => {
  const v = l2normalize([3, 4]);
  assert.deepEqual([...v], [0.6000000238418579, 0.800000011920929]);
});

test('argmaxRows per row, ties go to the lower index', () => {
  assert.deepEqual(argmaxRows([1, 3, 2, 5, 5, 0], 3), [1, 0]);
  assert.deepEqual(argmaxRows([1, 3, 2, 5], 2), [1, 1]);
});

test('zeroShot picks the largest entailment logit', () => {
  const r = zeroShot([-2.595508098602295, -0.0128010343760252, 0.7721060514450073,
    -2.5513229370117188, -2.819657802581787]);
  assert.equal(r.choice, 2);
  assert.ok(Math.abs(r.probabilities[2] - 0.6434125900268555) < 1e-6);
});

test('aggregateSimple groups like the HF pipeline: B- opens a group, O groups vanish, specials drop', () => {
  const labels = ['O', 'B-PER', 'I-PER', 'B-LOC'];
  const hot = (i: number, p = 0.9) => labels.map((_, k) => (k === i ? p : (1 - p) / 3));
  // [CLS] John Smith went to Paris and Rome [SEP]
  const rows = [hot(0), hot(1), hot(2), hot(0), hot(0), hot(3), hot(0), hot(3, 0.6), hot(0)];
  const probs = rows.flat();
  const offsets: [number, number][] = [[0, 0], [0, 4], [5, 10], [11, 15], [16, 18], [19, 24], [25, 28],
    [29, 33], [0, 0]];
  const mask = [1, 0, 0, 0, 0, 0, 0, 0, 1];
  const spans = aggregateSimple(probs, labels, offsets, mask);
  assert.deepEqual(spans.map((s) => [s.group, s.start, s.end]),
    [['PER', 0, 10], ['LOC', 19, 24], ['LOC', 29, 33]]);
  assert.ok(Math.abs(spans[0].score - 0.9) < 1e-12);
  assert.ok(Math.abs(spans[2].score - 0.6) < 1e-12);
  // I- after O with the same tag as the run before O does not join across the O
  const two = aggregateSimple([...hot(1), ...hot(0), ...hot(2)], labels,
    [[0, 1], [2, 3], [4, 5]], [0, 0, 0]);
  assert.deepEqual(two.map((s) => [s.group, s.start, s.end]), [['PER', 0, 1], ['PER', 4, 5]]);
  // two B- in a row are two groups; a bare I- with a new tag starts a group
  const b2 = aggregateSimple([...hot(1), ...hot(1)], labels, [[0, 1], [2, 3]], [0, 0]);
  assert.equal(b2.length, 2);
});
