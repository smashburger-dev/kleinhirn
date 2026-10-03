// Deliberately RED regression contracts for the unmodified engine at dfc4b24.
// Includes the user-requested ff-only update from bf91b9a (K28.6c convolution).
// Run: node --import ./tests/helpers/node-hooks.mjs --test tests/review/engine.test.mjs
// The mock records validation-relevant descriptors, it does NOT execute WGSL.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { specFromHfConfig } from '../../src/plan/hf.ts';
import { buildPlan } from '../../src/plan/build.ts';
import { model, normVariance, valueEpilogue, source } from '../helpers/review.mjs';

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
    // Q=K=0, all five keys valid, V=[0,0,0,0,5]. Correct attention is 1.
    assert.equal(valueEpilogue([1, 1, 1, 1, 1], [0, 0, 0, 0, 5]), 1);
  });
}

test('R05 constant row N384 must not have negative LayerNorm variance', () => {
  for (const kernel of ['layernorm', 'embln']) {
    assert.match(source(`src/kernels/${kernel}.wgsl`), /red\[0\] \/ f32\(N\) - mean \* mean/);
  }
  const variance = normVariance(new Float32Array(384).fill(100.1));
  assert.equal(variance, 0, `variance=${variance}, sqrt(var+eps)=${Math.sqrt(variance + 1e-7)}`);
});

test('R06 finite residual and GeGLU operands must fit f16 storage', () => {
  const residual = new Float16Array([40000 + 40000])[0];
  const gate = new Float16Array([256 * 256])[0]; // gelu(256) rounds to 256 in f32
  assert.ok(Number.isFinite(residual) && Number.isFinite(gate), `residual=${residual}, gate=${gate}`);
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
  assert.match(source('src/kernels/attention.wgsl'), /s \+= k0 \* f32\(posQuery/);
  // D=50, h=0: qreg[50..51] is zero, but the extra p2c term is NOT multiplied by qreg.
  const D = 50, k = new Float32Array(2 * D), pq = new Float32Array(2 * D);
  k[D] = 2; pq[D] = 3;
  let score = 0;
  for (let d = 0; d < D; d += 4) for (let j = 0; j < 4; j += 1) score += k[d + j] * pq[d + j];
  assert.equal(score, 0, `p2c cross-head contribution=${score}`);
});

// Deferred to K27 (real HF positions need a position-id input, a ninth embln binding). K28.R group D
// rejects such ids instead (tests/k28r_d.test.mjs, R09); this original contract stays red.
test('R09 RoBERTa literal pad in IDs must not shift later positions', () => {
  const { engine, spec } = model('cardiffnlp/twitter-roberta-base-sentiment-latest');
  const ids = [0, 7, 1, 8, 2]; // <s> word <pad> word </s>
  const { mask } = engine.maskAndTypes([{ inputIds: ids }], 128);
  assert.equal(mask[2], 1); // real token, not a mask hole; positions still use the pad ID in HF
  let count = 0;
  const hf = ids.map((id) => id === 1 ? 1 : ++count + 1);
  const actual = ids.map((_, i) => i + spec.embed.positionOffset);
  assert.deepEqual(actual, hf);
});

test('R21 finite f32 scores below the sentinel must not read masked stale rows', () => {
  assert.match(source('src/kernels/mbattention.wgsl'), /scores\[j\] = -1e30/);
  // Real score can be finite and still below -1e30, e.g. Q=1e16, K=-1e16.
  const scores = [-1e31, -1e30, -1e30, -1e30];
  const mx = Math.max(-1e30, ...scores);
  const exp = scores.map((s) => Math.exp(s - mx));
  assert.equal(valueEpilogue(exp, [2, 9, 9, 9]), 2);
});
