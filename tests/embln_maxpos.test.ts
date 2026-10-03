// K28.S2 finding 1: a bucket only sizes buffers. embln clamps the position row to the table, and the
// encoder rejects inputs longer than maxPositions - offset instead of rejecting the bucket at load.
// Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildPlan } from '../src/plan/build.ts';
import { EncoderModel } from '../src/encoder.ts';
import { PILOT_IDS, pilotSpec } from './helpers/pilot.ts';

const wgsl = readFileSync(new URL('../src/kernels/embln.wgsl', import.meta.url), 'utf8');

test('embln clamps the position index to the last table row', () => {
  assert.match(wgsl, /override MAXPOS: u32/);
  assert.match(wgsl, /min\(\(row % L\) \+ OFFSET, MAXPOS - 1u\)/);
});

for (const id of PILOT_IDS.slice(0, 3)) {
  test(`plan binds MAXPOS for a bucket longer than the position table, ${id}`, () => {
    const { spec, head } = pilotSpec(id);
    const small = { ...spec, embed: { ...spec.embed, maxPositions: spec.embed.positionOffset + 130 } };
    const plan = buildPlan(small, head, { length: 512, batch: 1, markers: 0, f16: false });
    const embed = plan.segments[0].ops[0];
    assert.equal(embed.kernel, 'embln');
    assert.equal(embed.constants.MAXPOS, small.embed.maxPositions);
    assert.equal(embed.constants.L, 512);
  });
}

test('runIds and runIdsBatch reject inputs beyond maxPositions - offset, with both numbers', async () => {
  const { spec } = pilotSpec(PILOT_IDS[0]);
  const model = Object.create(EncoderModel.prototype) as EncoderModel;
  Object.assign(model, { spec: { ...spec, embed: { ...spec.embed, maxPositions: 130, positionOffset: 2 } } });
  const long = { inputIds: new Array(129).fill(1) };
  await assert.rejects(model.runIds(long), /129 tokens.*130 position rows.*offset 2.*128 tokens/);
  await assert.rejects(model.runIdsBatch([long]), /129 tokens.*130 position rows/);
});
