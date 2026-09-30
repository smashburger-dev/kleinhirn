// Device result checker: a result JSON produced by the page (committed
// fixture, one real Chromium run) passes; the same file with one f32 logit
// changed by +1.0 is rejected; a wrong goldens hash is rejected.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { checkResult } from '../tools/check_device_result.mjs';

const load = () => JSON.parse(readFileSync(new URL('./fixtures/device-result-smoke.json', import.meta.url), 'utf8'));

test('smoke result passes', () => {
  const { errors, verdicts } = checkResult(load());
  assert.deepEqual(errors, []);
  assert.ok(verdicts.some((v) => v.ran));
  for (const v of verdicts.filter((x) => x.ran)) assert.equal(v.pass, true, v.name);
});

test('one logit off by 1.0 in an f32 case fails', () => {
  const r = load();
  const f32 = r.stages.find((s) => s.name === 'f32');
  assert.ok(f32?.ok, 'fixture needs an ok f32 stage');
  f32.parity.L128.perCaseLogits[3][2] += 1.0;
  const { errors, verdicts } = checkResult(r);
  assert.equal(verdicts.find((v) => v.name === 'f32').pass, false);
  assert.ok(errors.length > 0);
});

test('goldens hash mismatch is rejected', () => {
  const r = load();
  r.goldens.sha256 = '0'.repeat(64);
  assert.match(checkResult(r).errors[0], /goldens sha256 mismatch/);
});

test('missing required field is rejected', () => {
  const r = load();
  delete r.device;
  assert.ok(checkResult(r).errors.some((e) => e.includes('device')));
});

const marker = (over = {}) => ({
  present: true, firstSeen: '2026-09-26T10:00:00.000Z', lastSeen: '2026-09-29T10:00:00.000Z',
  visits: 3, ageDays: 3, readError: null, writeError: null, ...over,
});

test('storage marker: consistent report passes, present without visits fails', () => {
  const r = load();
  const absent = marker({ present: false, firstSeen: null, lastSeen: null, visits: null, ageDays: null });
  r.environment.storageMarker = { checkedAt: '2026-09-29T10:00:00.000Z', opfs: marker(), cache: absent, localStorage: marker() };
  assert.deepEqual(checkResult(r).errors, []);
  r.environment.storageMarker.opfs = marker({ visits: null });
  assert.ok(checkResult(r).errors.some((e) => e.includes('storageMarker.opfs') || e.includes('visits')));
});

test('limit < 200 is checked on the first items and flagged as diagnostic', () => {
  const r = load();
  const limit = 20;
  for (const st of r.stages.filter((x) => x.ok)) {
    for (const k of ['L128', 'L256']) st.parity[k].perCaseLogits = st.parity[k].perCaseLogits.slice(0, limit);
  }
  r.protocol = { warmup: 5, limit, trace: false, diagnostic: true };
  const { errors, verdicts } = checkResult(r);
  // parity flags were computed on all items; only the case count and shape are under test
  assert.ok(!errors.some((e) => /expected \d+ cases/.test(e)), errors.join('; '));
  assert.ok(verdicts.filter((v) => v.ran).every((v) => v.diagnostic === true));
});
