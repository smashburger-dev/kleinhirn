// K28.6 step 5: the plan for the eleven DeBERTa and ModernBERT pilot models at L128 and L512,
// B1 and B4, f32 and f16: the builder does not throw, segments validate, every operation binds as
// many buffers as its kernel declares, the executor builds under the mock and one call encodes the
// expected number of dispatches and maps once. Plus the two RoPE tables, the LayerNorm head step
// and the DeBERTa head. Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildPlan, ropeTable, validateSegments } from '../src/plan/build.ts';
import { PlanExecutor } from '../src/plan/executor.ts';
import type { HeadSpec, ModelSpec } from '../src/plan/spec.ts';
import { bindingMismatches } from './helpers/bindings.ts';
import { createMockGpu, installGpuGlobals } from './helpers/mock-gpu.ts';
import { PILOT_K286_IDS, pilotSpec } from './helpers/pilot.ts';

installGpuGlobals();

const headDispatches = (head: HeadSpec): number => {
  if (head.type === 'classify' || head.type === 'embed') return 1 + head.steps.length;
  if (head.type === 'token') return head.steps.length;
  throw new Error(head.type);
};

// post-norm layer: 7 dispatches. Pre-norm layer: lnA (not in layer 0), qkv, rope, attn, attnOut,
// addA, lnF, ffnIn, geglu, ffnOut, addF = 11, 10 in layer 0, plus the final norm.
function expectedDispatches(spec: ModelSpec, head: HeadSpec, plan: { segments: { ops: { kernel: string }[] }[] }): number {
  const body = spec.block.order === 'post'
    ? 1 + 7 * spec.layers
    : 1 + (11 * spec.layers - 1) + 1;
  // K27: matmul attention has 3 dispatches (5 with relative terms) instead of 1
  const attnExtra = plan.segments.flatMap((sg) => sg.ops).filter((o) => ['attrel', 'attscore', 'attsoftmax', 'attsoftrel', 'attpv'].includes(o.kernel)).length
    - plan.segments.flatMap((sg) => sg.ops).filter((o) => o.kernel === 'attscore').length;
  return body + attnExtra + headDispatches(head);
}

for (const id of PILOT_K286_IDS) {
  const { spec, head } = pilotSpec(id);
  for (const length of [128, 512]) {
    for (const batch of [1, 4]) {
      for (const f16 of [false, true]) {
        test(`plan ${id} L${length} B${batch} ${f16 ? 'f16' : 'f32'}`, async () => {
          const plan = buildPlan(spec, head, { length, batch, markers: 0, f16 });
          validateSegments(plan.segments, new Set(plan.buffers.map((b) => b.id)));
          assert.deepEqual(bindingMismatches(plan), []);
          for (const b of plan.buffers) assert.equal(b.bytes % 4, 0, `buffer ${b.id}`);
          assert.equal(plan.output.bytes % 4, 0);
          assert.equal(plan.inputs.markers, undefined);
          const absolute = spec.embed.positions === 'absolute';
          assert.equal(!!plan.inputs.typeIds, absolute);
          assert.equal(plan.buffers.some((b) => b.id === 'typeIds'), absolute);

          const gpu = createMockGpu(['timestamp-query']);
          const exec = new PlanExecutor(gpu.device, plan, gpu.weights());
          gpu.markBuilt();
          const rows = length * batch;
          exec.upload({
            embeddings: f16 ? new Uint16Array(rows * spec.embeddingSize)
              : new Float32Array(rows * spec.embeddingSize),
            mask: new Float32Array(rows),
            typeIds: new Uint32Array(rows),
          });
          exec.submit(false, { seqLen: rows });
          const out = await exec.readOutput();
          assert.equal(out.length, plan.output.rows * plan.output.cols);
          const dispatches = gpu.log.filter((ev) => ev.op === 'dispatch').length;
          assert.equal(dispatches, expectedDispatches(spec, head, plan), 'dispatch count');
          assert.equal(gpu.log.filter((ev) => ev.op === 'map').length, 1, 'one mapAsync');
        });
      }
    }
  }
}

const planOf = (id: string, length = 128, batch = 1) => {
  const { spec, head } = pilotSpec(id);
  return buildPlan(spec, head, { length, batch, markers: 0, f16: false });
};
const allOps = (plan: ReturnType<typeof planOf>) => plan.segments.flatMap((s) => s.ops);

test('two RoPE tables: global layers read cossinG, windowed layers cossinL', () => {
  const id = 'ibm-granite/granite-embedding-small-english-r2'; // theta 80000 and 10000
  const plan = planOf(id);
  const ids = plan.buffers.map((b) => b.id);
  assert.ok(ids.includes('cossinG') && ids.includes('cossinL') && !ids.includes('cossin'));
  const spec = pilotSpec(id).spec;
  for (let l = 0; l < spec.layers; l += 1) {
    const seg = plan.segments.find((s) => s.name === `layer${l}`)!;
    const rope = seg.ops.find((o) => o.name === 'rope')!;
    assert.equal(rope.bind[1], l % 3 === 0 ? 'cossinG' : 'cossinL', `layer ${l}`);
    // the op that applies the window: attP (matmul attention, K27) or attn
    const attn = seg.ops.find((o) => o.constants.WINDOW !== undefined)!;
    assert.equal(attn.constants.WINDOW, l % 3 === 0 ? 0 : 64);
  }
  const table = (name: string) => plan.buffers.find((b) => b.id === name)!.init as Float32Array;
  assert.deepEqual(table('cossinG'), ropeTable(128, 80000, spec.headDim));
  assert.deepEqual(table('cossinL'), ropeTable(128, 10000, spec.headDim));
});

test('equal RoPE thetas keep one table cossin', () => {
  const plan = planOf('sheltron-ai/prompt-guard-68m');
  const ids = plan.buffers.map((b) => b.id);
  assert.ok(ids.includes('cossin') && !ids.includes('cossinG') && !ids.includes('cossinL'));
  for (const op of allOps(plan).filter((o) => o.kernel === 'rope')) assert.equal(op.bind[1], 'cossin');
});

test('ModernBERT head: dense, LayerNorm over the pooled rows, classifier', () => {
  const plan = planOf('sheltron-ai/prompt-guard-68m', 128, 4);
  const ops = plan.segments.find((s) => s.name === 'head')!.ops;
  assert.deepEqual(ops.map((o) => o.kernel), ['pool', 'matmul', 'layernorm', 'matmul']);
  const ln = ops[2];
  assert.deepEqual([ln.constants.N, ln.constants.MODE, ln.constants.EPS], [512, 0, 1e-5]);
  assert.deepEqual(ln.bind.slice(0, 4), ['hd0', 'dummy', 'w:head.norm.weight', 'zero']);
  assert.deepEqual(ln.dispatch, [4, 1]); // one row per sequence
  assert.deepEqual(ops[1].bind.slice(0, 3), ['states', 'w:head.dense.weight', 'zero']); // classifier_bias false
  assert.deepEqual(ops[3].bind.slice(0, 3), ['hd1', 'w:head.classifier.weight', 'w:head.classifier.bias']);
  assert.deepEqual([plan.output.rows, plan.output.cols], [4, 22]);
});

test('ModernBERT token head: the LayerNorm runs on every row', () => {
  const plan = planOf('OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M', 128, 4);
  const ops = plan.segments.find((s) => s.name === 'head')!.ops;
  assert.deepEqual(ops.map((o) => o.kernel), ['matmul', 'layernorm', 'matmul']);
  assert.deepEqual(ops[1].dispatch, ['rows', 1]);
  assert.deepEqual([plan.output.rows, plan.output.cols], [128 * 4, 3]);
});

test('ModernBERT pooling follows classifier_pooling', () => {
  const kernels = (id: string) => planOf(id).segments.find((s) => s.name === 'head')!.ops[0].kernel;
  assert.equal(kernels('sheltron-ai/prompt-guard-68m'), 'pool');        // mean
  assert.equal(kernels('hotchpotch/japanese-reranker-xsmall-v2'), 'gather'); // cls
});

test('DeBERTa: relative attention plan, ContextPooler head, no type ids', () => {
  const plan = planOf('protectai/deberta-v3-base-prompt-injection-v2');
  // K27: relative attention as matmuls (c2p, p2c, scores, softmax with the relative terms, context)
  const ops = plan.segments.find((s) => s.name === 'layer0')!.ops;
  const at = (name: string) => ops.find((o) => o.name === name)!;
  assert.deepEqual(['relC', 'relP', 'attS', 'attP', 'attV'].map((n) => at(n).kernel),
    ['attrel', 'attrel', 'attscore', 'attsoftrel', 'attpv']);
  assert.equal(at('attS').constants.SCALE, 1 / Math.sqrt(3 * 64));
  assert.equal(at('attP').constants.INVSCALE, 1 / Math.sqrt(3 * 64));
  assert.deepEqual(at('relC').bind, ['qkv', 'w:layers.0.pos_key', 'kinfo', 'relidx', 'c2p']);
  assert.deepEqual(at('relP').bind, ['qkv', 'w:layers.0.pos_query', 'kinfo', 'relidx', 'p2c']);
  assert.deepEqual(at('attP').bind, ['mask', 'relidx', 'c2p', 'p2c', 'scores']);
  assert.equal(plan.segments[0].ops[0].kernel, 'layernorm');
  assert.equal(plan.segments[0].ops[0].constants.MODE, 1); // LN then mask, like GLiNER
  const head = plan.segments.find((s) => s.name === 'head')!.ops;
  assert.deepEqual(head.map((o) => o.kernel), ['gather', 'matmul', 'matmul']);
  assert.equal(head[1].constants.ACT, 2); // pooler_hidden_act gelu
  assert.deepEqual([plan.output.rows, plan.output.cols], [1, 2]);
});

test('polyBERT: absolute positions through the standard attention, head width 50', () => {
  const plan = planOf('xushijie/polyBERT');
  const embed = plan.segments[0].ops[0];
  assert.equal(embed.kernel, 'embln');
  assert.deepEqual([embed.constants.N, embed.constants.OFFSET, embed.constants.MASKMUL], [600, 0, 1]);
  assert.equal(embed.bind[2], 'zero'); // no type table
  const attn = plan.segments.find((s) => s.name === 'layer0')!.ops.find((o) => o.name === 'attn')!;
  assert.equal(attn.kernel, 'mbattention');
  assert.deepEqual([attn.constants.H, attn.constants.D, attn.constants.WINDOW], [12, 50, 0]);
  assert.equal(attn.constants.SCALE, 50 ** -0.5);
  assert.deepEqual([plan.output.rows, plan.output.cols], [1, 600]);
});

test('RoPE runs on one kernel for every head width, D = 32 and D = 64', () => {
  const kernel = (id: string) => allOps(planOf(id)).find((o) => o.name === 'rope')!;
  assert.equal(kernel('ibm-granite/granite-embedding-small-english-r2').kernel, 'rope'); // 384 / 12 = 32
  assert.equal(kernel('ibm-granite/granite-embedding-small-english-r2').constants.D, 32);
  assert.equal(kernel('sheltron-ai/prompt-guard-68m').kernel, 'rope');
  assert.equal(kernel('ibm-granite/granite-embedding-reranker-english-r2').kernel, 'rope');
  assert.match(readFileSync('src/kernels/rope.wgsl', 'utf8'), /if \(d >= half\) \{ return; \}/);
  assert.ok(!existsSync('src/kernels/ropenarrow.wgsl'));
});
