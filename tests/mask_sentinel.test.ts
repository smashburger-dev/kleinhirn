// K28.S3 step 1: masked keys must sit below every real score. A -1e4 sentinel gives padding keys weight
// when real scores fall under -1e4 (large activations); -1e30 does not. The test reads the sentinel
// from the kernel text and replays the kernel's softmax in float32 (max starts at -1e30, exp(s - max)).
// Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const f32 = Math.fround;

function sentinelOf(file: string): number {
  const src = readFileSync(new URL(`../src/kernels/${file}`, import.meta.url), 'utf8');
  const m = /if \([^)]*(?:outside|<= 0\.5)\) \{\s*scores\[j\] = (-1e\d+);/.exec(src);
  assert.ok(m, `${file}: masked-key assignment not found`);
  return Number(m[1]);
}

// Weights the kernel assigns, given real scores and a mask (true = keep).
function weights(real: number[], keep: boolean[], sentinel: number): number[] {
  const s = real.map((v, j) => (keep[j] ? f32(v) : f32(sentinel)));
  let m = f32(-1e30);
  for (const v of s) m = Math.max(m, v);
  const e = s.map((v) => f32(Math.exp(f32(v - m))));
  const sum = e.reduce((a, b) => f32(a + b), 0);
  return e.map((v) => f32(v / sum));
}

for (const file of ['mbattention.wgsl', 'attention.wgsl']) {
  test(`${file}: padding keys get zero weight when real scores are below -1e4`, () => {
    const sentinel = sentinelOf(file);
    const real = [-31000, -30500, -24000, -29000, -2, -2, -2, -2];
    const keep = [true, true, true, true, false, false, false, false];
    const w = weights(real, keep, sentinel);
    assert.equal(w.slice(4).reduce((a, b) => a + b, 0), 0);
    assert.ok(Math.abs(w.slice(0, 4).reduce((a, b) => a + b, 0) - 1) < 1e-6);
  });

  test(`${file}: real scores in the usual range keep their weights`, () => {
    const sentinel = sentinelOf(file);
    const real = [-13.4, -1.6, -5, -2, 0, 0];
    const keep = [true, true, true, true, false, false];
    const w = weights(real, keep, sentinel);
    assert.equal(w[4], 0);
    assert.equal(w[5], 0);
    assert.ok(w[1] > w[3] && w[3] > w[2] && w[2] > w[0]);
  });

  test(`${file}: fully masked rows stay uniform and finite`, () => {
    const sentinel = sentinelOf(file);
    const w = weights([1, 2, 3, 4], [false, false, false, false], sentinel);
    for (const v of w) assert.equal(v, 0.25);
  });
}
