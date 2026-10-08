// Deliberately RED regression contracts for the unmodified engine at dfc4b24.
// Includes the user-requested ff-only update from bf91b9a (K28.6c convolution).
// Run: node --import ./tests/helpers/node-hooks.mjs --test tests/review/engine.test.mjs
// The mock records validation-relevant descriptors, it does NOT execute WGSL.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { specFromHfConfig } from '../../src/plan/hf.ts';
import { buildPlan } from '../../src/plan/build.ts';
import { model, normVariance, valueEpilogue, source } from '../helpers/review.mjs';
import { buildManifest, F16RangeError } from '../../src/convert/manifest.ts';
import { assertFinite } from '../../src/tasks.ts';

// Only constructs plan metadata: the old fixture's CPU table is never read.
function bertPlanOwner(overrides = {}) {
  const { spec, head } = specFromHfConfig({
    model_type: 'bert', hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
    intermediate_size: 128, vocab_size: 8, max_position_embeddings: 8192,
    ...overrides,
  }, { task: 'token-classification' });
  const { engine } = model('sentence-transformers/all-MiniLM-L6-v2', 'f32', []);
  engine.spec = spec;
  engine.head = head;
  return engine;
}

// Deferred (K28.R, after the merge): the kernel contract stays red. K28.R group A rejects
// bucket lengths that are not a multiple of 4 at load instead (tests/k28r_a.test.mjs, R01).
// A remainder loop in attention.wgsl and mbattention.wgsl would change shared kernels.
for (const kernel of ['attention', 'mbattention']) {
  test(`R01 ${kernel}: L=5 includes the fifth value`, () => {
    assert.match(source(`src/kernels/${kernel}.wgsl`), /j \+ 3u < L/);
    assert.match(source(`src/kernels/${kernel}.wgsl`), /for \(var j = L - L % 4u; j < L; j \+= 1u\)/);
    // Q=K=0, all five keys valid, V=[0,0,0,0,5]. Correct attention is 1.
    assert.equal(valueEpilogue([1, 1, 1, 1, 1], [0, 0, 0, 0, 5]), 1);
  });
}

test('R05 constant row N384 must not have negative LayerNorm variance', () => {
  for (const kernel of ['layernorm', 'embln']) {
    assert.match(source(`src/kernels/${kernel}.wgsl`), /let d = xs\[i\] - mean;\s+sq \+= d \* d;/);
  }
  const variance = normVariance(new Float32Array(384).fill(100.1));
  assert.equal(variance, 0, `variance=${variance}, sqrt(var+eps)=${Math.sqrt(variance + 1e-7)}`);
});

// Rewritten 06.10. (Noa, docs/R2_WORKORDER.md): binary16 cannot hold these values, so no engine
// change can store them. The contract is that the overflow is reported, never returned as logits.
test('R06 f16 overflow is reported: the converter rejects weights beyond f16, the engine non-finite outputs', async () => {
  // the review's cases: residual 40000 + 40000 and GeGLU 256 * 256 become Inf in f16 storage
  assert.equal(new Float16Array([40000 + 40000])[0], Infinity);
  assert.equal(new Float16Array([256 * 256])[0], Infinity);
  await assert.rejects(() => buildManifest([
    { name: 'head.classifier.bias', shape: [1], data: Float32Array.of(70000) },
  ], 'f16', {}), (e) => e instanceof F16RangeError);
  assert.throws(() => assertFinite(Float32Array.of(0.5, Infinity), 'engine output'), /1 of 2 values are not finite/);
  assert.throws(() => assertFinite(Float32Array.of(NaN, 1), 'engine output'), /1 of 2 values are not finite/);
});

// Deferred: bind type ids (and the type table) in the plan for DeBERTa without absolute positions.
// K28.R group F rejects the combination at load instead (tests/k28r_f.test.mjs, R07); this original stays red.
test('R07 relative DeBERTa with type_vocab_size=2 must bind type ids and table', () => {
  const { spec, head } = specFromHfConfig({
    model_type: 'deberta-v2', hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
    intermediate_size: 128, vocab_size: 8, max_position_embeddings: 512,
    relative_attention: true, share_att_key: true, norm_rel_ebd: 'layer_norm',
    position_buckets: 256, pos_att_type: ['p2c', 'c2p'], position_biased_input: false,
    type_vocab_size: 2,
  }, { task: 'token-classification' });
  const p = buildPlan(spec, head, { length: 128, batch: 1, markers: 0, f16: false });
  assert.ok(p.inputs.typeIds && p.segments[0].ops[0].bind.includes('w:embeddings.type.weight'));
});

// Deferred (K28.R, after the merge): the kernel contract stays red. K28.R group A rejects
// relative attention with a head width outside groups of 4 at load (tests/k28r_a.test.mjs, R08).
test('R08 relative DeBERTa D50 must not sum dimensions from the next head', () => {
  const src = source('src/kernels/attention.wgsl');
  assert.match(src, /s \+= k0 \* f32\(posQuery/);
  assert.match(src, /for \(var d = 0u; d \+ 3u < D; d \+= 4u\)/);
  assert.match(src, /for \(var d = D - D % 4u; d < D; d \+= 1u\)/);
  // D=50, h=0: the next head's first dimension holds K=2, posQuery=3; the kernel's p2c loop
  // (groups of 4 while d+3<D, then the scalar remainder) must not reach it.
  const D = 50, k = new Float32Array(2 * D), pq = new Float32Array(2 * D);
  k[D] = 2; pq[D] = 3;
  let score = 0;
  for (let d = 0; d + 3 < D; d += 4) for (let j = 0; j < 4; j += 1) score += k[d + j] * pq[d + j];
  for (let d = D - (D % 4); d < D; d += 1) score += k[d] * pq[d];
  assert.equal(score, 0, `p2c cross-head contribution=${score}`);
});

// Deferred to K27 (real HF positions need a position-id input, a ninth embln binding). K28.R group D
// rejects such ids instead (tests/k28r_d.test.mjs, R09); this original contract stays red.
test('R09 RoBERTa literal pad in IDs must not shift later positions', () => {
  const { engine, spec } = model('cardiffnlp/twitter-roberta-base-sentiment-latest');
  const ids = [0, 7, 1, 8, 2]; // <s> word <pad> word </s>
  const { mask, typeIds } = engine.maskAndTypes([{ inputIds: ids }], 128);
  assert.equal(mask[2], 1); // real token, not a mask hole; positions still use the pad ID in HF
  let count = 0;
  const hf = ids.map((id) => id === 1 ? 1 : ++count + 1);
  // embln with POSIDS 1 reads the position row from the high 16 bits of the type id
  assert.equal(engine.plans.get(128).plan.segments[0].ops[0].constants.POSIDS, 1);
  const actual = ids.map((_, i) => typeIds[i] >>> 16);
  assert.deepEqual(actual, hf);
});

test('R21 finite f32 scores below the sentinel must not read masked stale rows', () => {
  const src = source('src/kernels/mbattention.wgsl');
  assert.match(src, /if \(keep\(il, j, kbase\)\) \{ m = max\(m, scores\[j\]\); \}/);
  assert.match(src, /if \(keep\(il, j, kbase\)\) \{ e = exp\(scores\[j\] - mx\); \}/);
  // Real score can be finite and still below -1e30, e.g. Q=1e16, K=-1e16. Key 0 is kept, keys 1-3
  // are masked; the kernel leaves masked keys out of the max and the exponentials.
  const scores = [-1e31, 0, 0, 0];
  const keep = [true, false, false, false];
  const mx = Math.max(-3.4028234663852886e38, ...scores.filter((_, j) => keep[j]));
  const exp = scores.map((s, j) => (keep[j] ? Math.exp(s - mx) : 0));
  assert.equal(valueEpilogue(exp, [2, 9, 9, 9]), 2);
});
