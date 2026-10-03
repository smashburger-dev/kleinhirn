// K28.S3 decision 4: f32 decisions against torch64. The Python rule lives in convert/k28_close_calls.py;
// the test runs it on a small reference, as the sweep driver does through the result files.
// Skipped when .venv-k28 is absent. Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
const py = `${root}.venv-k28/bin/python`;

function run(ref: object, metrics: object): any {
  const code = 'import sys, json; sys.path.insert(0, "convert"); import k28_close_calls as c;'
    + 'a = json.loads(sys.stdin.read()); print(json.dumps(c.decisions32(a[0], a[1])))';
  return JSON.parse(execFileSync(py, ['-c', code], { cwd: root, input: JSON.stringify([ref, metrics]), encoding: 'utf8' }));
}

// golden error 1e-3 -> limit 2e-3; token 2 has torch64 gap 1e-3 (inside), token 3 has 0.5 (outside)
const ref = { golden32_vs_torch64: 1e-3, gaps: { tokenArgmax: [5, 5, 1e-3, 0.5] } };

test('an f32 deviation with a torch64 gap below twice the golden error is allowed', {
  skip: !existsSync(py),
}, () => {
  const d = run(ref, { argmaxPerTokenDisagreements: [2] });
  assert.deepEqual(d.decisions.tokenArgmax, { n: 4, deviations: 1, allowed: 1, notAllowed: 0 });
});

test('an f32 deviation with a larger torch64 gap is not allowed', { skip: !existsSync(py) }, () => {
  const d = run(ref, { argmaxPerTokenDisagreements: [2, 3] });
  assert.deepEqual(d.decisions.tokenArgmax, { n: 4, deviations: 2, allowed: 1, notAllowed: 1 });
});

test('a reference without gaps, or a parity file without indices, cannot be evaluated', {
  skip: !existsSync(py),
}, () => {
  assert.match(run({ golden32_vs_torch64: 1e-3 }, {}).error, /without gaps/);
  assert.match(run(ref, {}).decisions.tokenArgmax.error, /lacks argmaxPerTokenDisagreements/);
});
