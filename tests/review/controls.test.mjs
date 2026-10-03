// GREEN controls, static sweep, and explicitly non-reference characterization.
// These are not GPU parity measurements.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildPlan } from '../../src/plan/build.ts';
import { specFromHfConfig } from '../../src/plan/hf.ts';
import { specFromGlinerManifest, specFromJuliaManifest } from '../../src/plan/spec.ts';
import { DISTILUSE_DENSE } from '../helpers/pilot.ts';
import { GLINER_SPECS, GLINER_HEAD_HIDDEN, JULIA_SPEC } from '../helpers/plan-trace.ts';
import { declaredBindings } from '../helpers/bindings.ts';
import { JsonTokenizer } from '../../src/tokenizer/hf/index.ts';
import { normVariance, valueEpilogue, fixture, source, MIN } from '../helpers/review.mjs';

export function shapes() {
  const rows = JSON.parse(source('data/k28/models.json')).models;
  const accepted = [], rejected = [];
  for (const row of rows) {
    const st = row.sentenceTransformers;
    try {
      const extras = { task: row.task,
        template: [row.config.bos_token_id, null, row.config.eos_token_id],
        sentenceTransformers: st?.dense === true ? { ...st, dense: DISTILUSE_DENSE } : st };
      accepted.push({ id: row.id, ...specFromHfConfig(row.config, extras), markers: 0 });
    } catch (e) { rejected.push({ id: row.id, error: e.message }); }
  }
  for (const name of ['small', 'base']) {
    accepted.push({ id: `gliner-${name}`, ...specFromGlinerManifest(GLINER_SPECS[name],
      { temperature: 1, hiddenSize: GLINER_HEAD_HIDDEN[name] }), markers: 16 });
  }
  accepted.push({ id: 'julia-1', ...specFromJuliaManifest(JULIA_SPEC), markers: 20 });
  return { accepted, rejected };
}

export function sweep() {
  const { accepted, rejected } = shapes();
  let plans = 0, oversizedPlans = 0, oversizedAdmittedBatchPlans = 0;
  const over = [];
  for (const row of accepted) for (const length of [64, 128, 256, 512, 1024, 1280, 2048]) {
    for (const batch of [1, 4, 8, 16]) for (const f16 of [false, true]) {
      const p = buildPlan(row.spec, row.head, { length, batch, markers: row.markers, f16 });
      plans += 1;
      assert.ok(p.buffers.every((b) => Number.isSafeInteger(b.bytes) && b.bytes % 4 === 0));
      assert.ok(p.output.bytes % 4 === 0);
      for (const seg of p.segments) for (const op of [...(seg.captureOps ?? []), ...seg.ops]) {
        assert.equal(op.bind.length, declaredBindings(op.kernel));
        assert.ok(op.bind.length <= 8);
        for (const d of op.dispatch) {
          const n = d === 'rows' ? batch * length : d === 'rows8' ? Math.ceil(batch * length / 8) : d === 'rows16' ? Math.ceil(batch * length / 16) : d === 'rows32' ? Math.ceil(batch * length / 32) : d;
          assert.ok(Number.isInteger(n) && n >= 0 && n <= MIN.groups);
        }
      }
      const bad = p.buffers.filter((b) => b.bytes > MIN.binding);
      if (!bad.length) continue;
      oversizedPlans += 1;
      if (row.markers !== 0 || batch === 1) continue;
      // EncoderModel's actual guard, deliberately not corrected here.
      const { hidden: H, intermediate: I, embeddingSize: E } = row.spec;
      const guard = batch * length * Math.max(3 * H, I, E) * (f16 ? 2 : 4);
      if (guard > MIN.binding) continue;
      oversizedAdmittedBatchPlans += 1;
      over.push({ id: row.id, length, batch, f16, guard, bad: bad.map((b) => [b.id, b.bytes]) });
    }
  }
  return { shapes: accepted.length, rejected, plans, oversizedPlans, oversizedAdmittedBatchPlans, over };
}

test('C01 sweep all frozen accepted shapes plus GLiNER/Julia: bindings, sizes, dispatch', () => {
  const r = sweep();
  assert.equal(r.shapes, 92); // 89 accepted HF rows + three legacy shapes (dfc4b24)
  assert.equal(r.rejected.length, 0);
  assert.ok(r.oversizedAdmittedBatchPlans > 0); // records existing R02, not a correctness assertion
  console.log(JSON.stringify({ ...r, over: r.over.slice(0, 3) }));
});

test('C02 L%4=0 attention ignores stale masked NaN values', () => {
  assert.equal(valueEpilogue([1, 1, 0, 0], [2, 4, NaN, NaN]), 3);
});

test('C03 variance zero with positive epsilon is safe for exactly representable constants', () => {
  assert.equal(normVariance(new Float32Array(384).fill(1)), 0);
  assert.equal((1 - 1) / Math.sqrt(0 + 1e-7), 0);
});

test('C04 stored HF 0.22.2 tokenizer fixtures still match IDs, types and offsets', () => {
  let count = 0;
  for (const name of ['bpe-roberta', 'bpe-roberta-prefix', 'bpe-nfc-bytelevel-post', 'unigram-metaspace', 'bpe-metaspace']) {
    const tok = JsonTokenizer.fromJson(fixture(name));
    const cases = JSON.parse(readFileSync(`tests/fixtures/hf-tokenizers/${name}.expected.json`, 'utf8'));
    for (const c of cases) {
      const opt = { maxLength: c.max_length, truncation: c.truncation };
      if (!c.ok) { assert.throws(() => tok.encode(c.first, c.second, opt)); continue; }
      const e = tok.encode(c.first, c.second, opt);
      const offsets = c.offsets.map(([a, b], i) => {
        if (c.sequence_ids[i] < 0) return [a, b];
        const text = c.sequence_ids[i] === 0 ? c.first : c.second;
        return [[...text].slice(0, a).join('').length, [...text].slice(0, b).join('').length];
      });
      assert.deepEqual(e.ids, c.ids);
      assert.deepEqual(e.typeIds, c.type_ids);
      assert.deepEqual(e.offsets, offsets);
      count += 1;
    }
  }
  console.log(`stored HF fixture successes=${count}`);
});

test('T04 characterization only: rstrip plus added whitespace, HF parity UNVERIFIED', () => {
  const j = fixture('bpe-nfc-bytelevel-post');
  j.added_tokens = [
    { content: 'A', id: 900, special: false, normalized: false, single_word: false, lstrip: false, rstrip: true },
    { content: ' ', id: 901, special: false, normalized: false, single_word: false, lstrip: false, rstrip: false },
  ];
  const out = JsonTokenizer.fromJson(j).encode('A x');
  // Records current behavior; NOT an assertion of what HF 0.22.2 ought to emit.
  assert.deepEqual(out.ids, [900, 901, 125]);
  assert.deepEqual(out.offsets, [[0, 1], [2, 2], [2, 3]]);
  console.log(`T04 no new HF reference: ${JSON.stringify(out)}`);
});
