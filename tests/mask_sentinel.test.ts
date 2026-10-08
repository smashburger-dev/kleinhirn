// K28.S3 step 1 and review R21: masked keys must get zero weight whatever the real scores are. A -1e4
// sentinel gave padding keys weight when real scores fell under -1e4 (large activations); a -1e30
// sentinel still did under -1e30. The kernels now leave masked keys out of the max and the
// exponentials, and the max starts at the lowest finite f32. The test checks that structure in the
// kernel text and replays the kernel's softmax in float32.
// Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const f32 = Math.fround;
const LOWEST = -3.4028234663852886e38;

function checkStructure(file: string): void {
  const src = readFileSync(new URL(`../src/kernels/${file}`, import.meta.url), 'utf8');
  assert.match(src, /const LOWEST = -3\.40282346638528859812e38f;/, `${file}: max start`);
  assert.match(src, /var m = LOWEST;/, `${file}: max starts at LOWEST`);
  assert.match(src, /\{ m = max\(m, scores\[j\]\); \}/, `${file}: max over kept keys only`);
  assert.match(src, /var e = 0\.0;\s+if \([^{]*\) \{ e = exp\(scores\[j\] - mx\); \}/, `${file}: exp of kept keys only`);
  assert.match(src, /let invTotal = select\(0\.0, 1\.0 \/ red\[0\], red\[0\] > 0\.0\);/, `${file}: zero-sum guard`);
}

// Weights the kernel assigns, given real scores and a mask (true = keep).
function weights(real: number[], keep: boolean[]): number[] {
  let m = f32(LOWEST);
  real.forEach((v, j) => { if (keep[j]) m = Math.max(m, f32(v)); });
  const e = real.map((v, j) => (keep[j] ? f32(Math.exp(f32(f32(v) - m))) : 0));
  const sum = e.reduce((a, b) => f32(a + b), 0);
  return e.map((v) => (sum > 0 ? f32(v / sum) : 0));
}

for (const file of ['mbattention.wgsl', 'attention.wgsl']) {
  test(`${file}: padding keys get zero weight when real scores are below -1e4`, () => {
    checkStructure(file);
    const w = weights([-31000, -30500, -24000, -29000, -2, -2, -2, -2], [true, true, true, true, false, false, false, false]);
    assert.equal(w.slice(4).reduce((a, b) => a + b, 0), 0);
    assert.ok(Math.abs(w.slice(0, 4).reduce((a, b) => a + b, 0) - 1) < 1e-6);
  });

  test(`${file}: padding keys get zero weight when real scores are below -1e30`, () => {
    checkStructure(file);
    const w = weights([-1e31, -2e31, 9, 9], [true, true, false, false]);
    assert.equal(w[2] + w[3], 0);
    assert.equal(w[0], 1);
  });

  test(`${file}: real scores in the usual range keep their weights`, () => {
    checkStructure(file);
    const w = weights([-13.4, -1.6, -5, -2, 0, 0], [true, true, true, true, false, false]);
    assert.equal(w[4], 0);
    assert.equal(w[5], 0);
    assert.ok(w[1] > w[3] && w[3] > w[2] && w[2] > w[0]);
  });

  test(`${file}: a row without a kept key gives zeros, never NaN`, () => {
    checkStructure(file);
    for (const v of weights([1, 2, 3, 4], [false, false, false, false])) assert.equal(v, 0);
  });
}
