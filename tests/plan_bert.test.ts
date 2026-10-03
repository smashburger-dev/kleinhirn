// K28.4 step 3, K28.5 step 4: the plan for the eight BERT and the sixteen
// RoBERTa, XLM-R and DistilBERT pilot models (data/k28/models.json,
// configs through specFromHfConfig) at L128 and L512, B1 and B4, f32 and
// f16: the builder does not throw, segments validate, every operation binds
// as many buffers as its kernel declares, the executor builds under the mock
// and one call encodes the expected number of dispatches and maps once.
// Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildPlan, validateSegments } from '../src/plan/build.ts';
import { PlanExecutor } from '../src/plan/executor.ts';
import type { HeadSpec } from '../src/plan/spec.ts';
import { bindingMismatches } from './helpers/bindings.ts';
import { createMockGpu, installGpuGlobals } from './helpers/mock-gpu.ts';
import { PILOT_IDS, PILOT_K285_IDS, pilotSpec } from './helpers/pilot.ts';

// Dispatches after the layers: the head as described in the work order.
function headDispatches(head: HeadSpec): number {
  switch (head.type) {
    case 'classify': case 'embed': return 1 + head.steps.length; // gather or pool, then steps
    case 'token': return head.steps.length;
    default: throw new Error(`head ${head.type}`);
  }
}

installGpuGlobals();

for (const id of [...PILOT_IDS, ...PILOT_K285_IDS]) {
  const { spec, head } = pilotSpec(id);
  for (const length of [128, 512]) {
    for (const batch of [1, 4]) {
      for (const f16 of [false, true]) {
        const tag = `${id} L${length} B${batch} ${f16 ? 'f16' : 'f32'}`;
        test(`plan ${tag}`, async () => {
          const plan = buildPlan(spec, head, { length, batch, markers: 0, f16 });
          validateSegments(plan.segments, new Set(plan.buffers.map((b) => b.id)));
          assert.deepEqual(bindingMismatches(plan), []);
          for (const b of plan.buffers) assert.equal(b.bytes % 4, 0, `buffer ${b.id}`);
          assert.equal(plan.output.bytes % 4, 0);
          assert.ok(plan.inputs.typeIds, 'type ids are an input');
          assert.equal(plan.inputs.markers, undefined);
          const embed = plan.segments[0].ops[0];
          assert.equal(embed.kernel, 'embln');
          assert.equal(embed.constants.OFFSET, spec.embed.positionOffset);
          if (spec.embed.typeVocab === 0) {
            // DistilBERT: the type table is the shared zero buffer, wide enough for one row.
            assert.equal(embed.bind[2], 'zero');
            const zero = plan.buffers.find((b) => b.id === 'zero');
            assert.ok(zero && zero.bytes >= spec.embeddingSize * (f16 ? 2 : 4));
          } else {
            assert.equal(embed.bind[2], 'w:embeddings.type.weight');
          }

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
          // K27: matmul attention has 3 dispatches (5 with relative terms) instead of 1
          const attnExtra = plan.segments.flatMap((sg) => sg.ops).filter((o) => ['attrel', 'attscore', 'attsoftmax', 'attsoftrel', 'attpv'].includes(o.kernel)).length
    - plan.segments.flatMap((sg) => sg.ops).filter((o) => o.kernel === 'attscore').length;
          const expected = (spec.embed.project ? 2 : 1) + 7 * spec.layers + attnExtra + headDispatches(head);
          const dispatches = gpu.log.filter((ev) => ev.op === 'dispatch').length;
          assert.equal(dispatches, expected, 'dispatch count');
          assert.equal(gpu.log.filter((ev) => ev.op === 'map').length, 1, 'one mapAsync');
        });
      }
    }
  }
}

test('output shapes: sequence, reranking, token, embeddings', () => {
  const shape = (id: string, batch: number) => {
    const { spec, head } = pilotSpec(id);
    return buildPlan(spec, head, { length: 128, batch, markers: 0, f16: true }).output;
  };
  const rerank = shape('cross-encoder/ms-marco-MiniLM-L4-v2', 1);
  assert.deepEqual([rerank.dtype, rerank.rows, rerank.cols, rerank.bytes], ['storage', 1, 1, 4]);
  const nli = shape('MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33', 4);
  assert.deepEqual([nli.rows, nli.cols], [4, 2]);
  const token = shape('dslim/bert-base-NER', 4);
  assert.deepEqual([token.rows, token.cols], [128 * 4, 9]);
  const emb = shape('BAAI/bge-small-en-v1.5', 4);
  assert.deepEqual([emb.rows, emb.cols], [4, 384]);
});

test('K28.5 output shapes: RoBERTa head, XLM-R reranker, DistilBERT heads, Dense embeddings', () => {
  const shape = (id: string, batch: number) => {
    const { spec, head } = pilotSpec(id);
    return buildPlan(spec, head, { length: 128, batch, markers: 0, f16: true }).output;
  };
  const seq = shape('cardiffnlp/twitter-roberta-base-sentiment-latest', 4);
  assert.deepEqual([seq.dtype, seq.rows, seq.cols], ['storage', 4, 3]);
  const rerank = shape('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 1);
  assert.deepEqual([rerank.rows, rerank.cols, rerank.bytes], [1, 1, 4]);
  const tok = shape('OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M', 4);
  assert.deepEqual([tok.rows, tok.cols], [128 * 4, 3]);
  const dense = shape('sentence-transformers/distiluse-base-multilingual-cased-v1', 4);
  assert.deepEqual([dense.rows, dense.cols], [4, 512]);
  const distil = shape('typeform/distilbert-base-uncased-mnli', 1);
  assert.deepEqual([distil.rows, distil.cols], [1, 3]);
});

test('Dense step: a tanh matmul from 768 to 512 after the pooling', () => {
  const { spec, head } = pilotSpec('sentence-transformers/distiluse-base-multilingual-cased-v1');
  const plan = buildPlan(spec, head, { length: 128, batch: 4, markers: 0, f16: false });
  const ops = plan.segments.find((s) => s.name === 'head')!.ops;
  assert.deepEqual(ops.map((o) => o.kernel), ['pool', 'matmul']);
  const mm = ops[1];
  assert.deepEqual([mm.constants.N, mm.constants.K, mm.constants.ACT], [512, 768, 3]);
  assert.deepEqual(mm.bind.slice(0, 3), ['states', 'w:head.dense0.weight', 'w:head.dense0.bias']);
});
