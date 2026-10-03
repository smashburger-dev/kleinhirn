// K28.6c: the DeBERTa-v2 convolution layer (conv_kernel_size > 0), without a model or a GPU.
// The spec from config.json, the canonical tensors and the Conv1d weight layout, the plan segments
// (convIn before layer 0, conv after it, the capture slot of layer 0 behind the convolution), and the
// im2col index rule at the sequence edges of a batch of 2. Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildPlan, validateSegments } from '../src/plan/build.ts';
import { PlanExecutor } from '../src/plan/executor.ts';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { conv1dRows } from '../src/convert/manifest.ts';
import { planNames } from '../src/convert/names.ts';
import { bindingMismatches } from './helpers/bindings.ts';
import { createMockGpu, installGpuGlobals } from './helpers/mock-gpu.ts';
import { pilotSpec } from './helpers/pilot.ts';

installGpuGlobals();

const CAMEMBERTA = 'antoinelouis/crossencoder-camemberta-L6-mmarcoFR';
const models = (JSON.parse(readFileSync(
  new URL('../data/k28/models.json', import.meta.url), 'utf8')) as {
  models: { id: string; config: Record<string, unknown> }[] }).models;
const camemberta = models.find((m) => m.id === CAMEMBERTA);

test('camemberta-L6: the spec carries the convolution, other DeBERTa models carry none', () => {
  const { spec } = pilotSpec(CAMEMBERTA);
  assert.deepEqual(spec.conv, { kernel: 3, act: 'gelu' });
  assert.equal(pilotSpec('protectai/deberta-v3-base-prompt-injection-v2').spec.conv, undefined);
  assert.equal(camemberta?.config.conv_groups, undefined);
});

test('Conv1d weight [out, in, k] becomes [out, k * in] with column t * in + ci = W[c, ci, t]', () => {
  const [out, inp, k] = [2, 3, 3];
  const w = Float32Array.from({ length: out * inp * k }, (_, i) => i);
  const r = conv1dRows(w, out, inp, k);
  for (let c = 0; c < out; c += 1) {
    for (let ci = 0; ci < inp; ci += 1) {
      for (let t = 0; t < k; t += 1) assert.equal(r[c * k * inp + t * inp + ci], w[(c * inp + ci) * k + t]);
    }
  }
});

test('names: the convolution adds conv.weight (transformed), conv.bias and the conv LayerNorm', () => {
  const H = 4;
  const config = { model_type: 'deberta-v2', hidden_size: H, num_attention_heads: 2, num_hidden_layers: 1,
    intermediate_size: 8, vocab_size: 5, max_position_embeddings: 8, type_vocab_size: 0,
    relative_attention: true, position_buckets: 2, share_att_key: true, pos_att_type: ['p2c', 'c2p'],
    norm_rel_ebd: 'layer_norm', position_biased_input: false, conv_kernel_size: 3, conv_act: 'gelu',
    hidden_act: 'gelu', pooler_hidden_size: H, id2label: { 0: 'a' }, layer_norm_eps: 1e-7 };
  const { spec, head } = specFromHfConfig(config, { task: 'sequence-classification' });
  const b = 'deberta.encoder.layer.0';
  const names = ['deberta.embeddings.word_embeddings.weight', 'deberta.embeddings.LayerNorm.weight',
    'deberta.embeddings.LayerNorm.bias', 'deberta.encoder.rel_embeddings.weight',
    'deberta.encoder.LayerNorm.weight', 'deberta.encoder.LayerNorm.bias',
    'deberta.encoder.conv.conv.weight', 'deberta.encoder.conv.conv.bias',
    'deberta.encoder.conv.LayerNorm.weight', 'deberta.encoder.conv.LayerNorm.bias',
    'pooler.dense.weight', 'pooler.dense.bias', 'classifier.weight', 'classifier.bias'];
  for (const p of ['query_proj', 'key_proj', 'value_proj']) {
    names.push(`${b}.attention.self.${p}.weight`, `${b}.attention.self.${p}.bias`);
  }
  for (const p of ['attention.output.dense', 'attention.output.LayerNorm', 'intermediate.dense',
    'output.dense', 'output.LayerNorm']) names.push(`${b}.${p}.weight`, `${b}.${p}.bias`);
  const plan = planNames(spec, head, names);
  const conv = plan.tensors.filter((t) => t.name.startsWith('conv.'));
  assert.deepEqual(conv.map((t) => t.name), ['conv.weight', 'conv.bias', 'conv.ln.weight', 'conv.ln.bias']);
  assert.deepEqual(conv[0].shape, [H, 3 * H]);
  assert.equal(conv[0].transform, 'conv1d');
  assert.deepEqual(conv[0].sources, ['deberta.encoder.conv.conv.weight']);
});

const planOf = (batch: number, f16: boolean, length = 128) => {
  const { spec, head } = pilotSpec(CAMEMBERTA);
  return { spec, plan: buildPlan(spec, head, { length, batch, markers: 0, f16 }) };
};

for (const [batch, f16] of [[1, false], [2, false], [2, true]] as const) {
  test(`plan camemberta-L6 B${batch} ${f16 ? 'f16' : 'f32'}: convIn, layer0, conv, one capture slot each`, async () => {
    const { spec, plan } = planOf(batch, f16);
    assert.deepEqual(plan.segments.slice(0, 5).map((s) => s.name), ['embed', 'convIn', 'layer0', 'conv', 'layer1']);
    validateSegments(plan.segments, new Set(plan.buffers.map((b) => b.id)));
    assert.deepEqual(bindingMismatches(plan), []);
    const [convIn, layer0, conv] = [1, 2, 3].map((i) => plan.segments[i]);
    assert.deepEqual(convIn.ops.map((o) => o.kernel), ['im2col', 'matmul']);
    assert.deepEqual(convIn.ops[0].bind, ['x', 'mask', 'cols']);
    assert.deepEqual(convIn.ops[0].constants, { N: 768, L: 128, KS: 3 });
    assert.deepEqual(convIn.ops[1].constants, { M: 128 * batch, N: 768, K: 3 * 768, ACT: 2 });
    assert.deepEqual(convIn.ops[1].bind, ['cols', 'w:conv.weight', 'w:conv.bias', 'convOut']);
    assert.deepEqual(conv.ops.map((o) => `${o.kernel}${o.constants.MODE}`), ['add0', 'layernorm1']);
    assert.deepEqual(conv.ops[1].bind, ['convOut', 'dummy', 'w:conv.ln.weight', 'w:conv.ln.bias', 'mask', 'x']);
    // hidden_states[1] of HF is the state after the convolution: slots embed 0 and 1 (plain and masked LN), conv 2, layer1 3, ...
    assert.equal(layer0.capture, undefined);
    assert.deepEqual(conv.capture, [{ buffer: 'x', slot: 2 }]);
    assert.equal(plan.segments[4].capture![0].slot, 3);
    assert.equal(plan.captureSlots, 2 + spec.layers);
    const bytes = (id: string) => plan.buffers.find((b) => b.id === id)!.bytes;
    const el = f16 ? 2 : 4;
    assert.equal(bytes('cols'), 128 * batch * 3 * 768 * el);
    assert.equal(bytes('convOut'), 128 * batch * 768 * el);

    const gpu = createMockGpu(['timestamp-query']);
    const exec = new PlanExecutor(gpu.device, plan, gpu.weights());
    gpu.markBuilt();
    const rows = 128 * batch;
    exec.upload({ embeddings: f16 ? new Uint16Array(rows * 768) : new Float32Array(rows * 768),
      mask: new Float32Array(rows), typeIds: new Uint32Array(rows) });
    exec.submit(false, { seqLen: rows });
    await exec.readOutput();
    // embed 1 + convIn 2 + 6 layers * 11 (relative attention as 5 matmul-shaped ops, K27) + conv 2
    // + head (gather, pooler, classifier) 3
    assert.equal(gpu.log.filter((ev) => ev.op === 'dispatch').length, 1 + 2 + 66 + 2 + 3);
  });
}

// Same index rule as src/kernels/im2col.wgsl, one workgroup per row.
function im2colRow(emb: number[][], mask: number[], row: number, L: number, KS: number): number[] {
  const out: number[] = [];
  const seq = Math.floor(row / L);
  const i = row % L;
  const pad = (KS - 1) >> 1;
  for (let t = 0; t < KS; t += 1) {
    const j = i + t - pad;
    out.push(...(j >= 0 && j < L && mask[seq * L + j] !== 0 ? emb[seq * L + j] : emb[0].map(() => 0)));
  }
  return out;
}

test('im2col: block t of row i is row i + t - 1, zero outside the own sequence or at mask 0 (B2, L4, kernel 3 and 5)', () => {
  const src = readFileSync(new URL('../src/kernels/im2col.wgsl', import.meta.url), 'utf8');
  assert.match(src, /let pad = i32\(\(KS - 1u\) \/ 2u\);/);
  assert.match(src, /let j = i \+ i32\(t\) - pad;/);
  assert.match(src, /let ok = j >= 0 && j < i32\(L\) && mask\[srcRow\] != 0\.0;/);
  const L = 4;
  const emb = Array.from({ length: 2 * L }, (_, r) => [r + 1, 10 * (r + 1)]); // rows 1..8
  const ones = emb.map(() => 1);
  const rows = emb.map((_, r) => im2colRow(emb, ones, r, L, 3));
  assert.deepEqual(rows[0], [0, 0, 1, 10, 2, 20]);      // first row of sequence 0: left edge zero
  assert.deepEqual(rows[3], [3, 30, 4, 40, 0, 0]);      // last row of sequence 0: right edge zero, not row 4
  assert.deepEqual(rows[4], [0, 0, 5, 50, 6, 60]);      // first row of sequence 1: no read of row 3
  assert.deepEqual(rows[7], [7, 70, 8, 80, 0, 0]);
  assert.deepEqual(im2colRow(emb, ones, 5, L, 5), [0, 0, 5, 50, 6, 60, 7, 70, 8, 80]); // row 5 = second of sequence 1
  // padding rows (mask 0, stale values): sequence 0 has 3 valid rows, so row 3 reads as zero for row 2
  const padded = emb.map((_, r) => (r === 3 ? 0 : 1));
  assert.deepEqual(im2colRow(emb, padded, 2, L, 3), [2, 20, 3, 30, 0, 0]);
  assert.deepEqual(im2colRow(emb, padded, 4, L, 3), [0, 0, 5, 50, 6, 60]);
});
