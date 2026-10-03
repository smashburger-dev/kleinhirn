// K28.1 gate G1/G3: the dispatch lists of the plan executor must equal the
// lists frozen from the classes before K28 (src/graph/deberta.ts and
// julia.ts at a9c89d1, recorded with the test at bbf84e2; the frozen file
// carries the source commit). Run with the Node hooks:
//   node --import ./tests/helpers/node-hooks.mjs --test tests/plan_trace.test.ts

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { buildPlan, validateSegments } from '../src/plan/build.ts';
import { PlanExecutor } from '../src/plan/executor.ts';
import type { Plan } from '../src/plan/ir.ts';
import { specFromGlinerManifest, specFromJuliaManifest } from '../src/plan/spec.ts';
import {
  GLINER_HEAD_HIDDEN, GLINER_SPECS, GLINER_TEMPERATURE, JULIA_SPEC,
  allConfigs, firstDifference, recordTrace,
  type Impl, type PlanLike, type Trace, type TraceConfig,
} from './helpers/plan-trace.ts';

const FROZEN = new URL('./golden/k28/dispatch-traces.json.gz', import.meta.url);

interface Frozen {
  // last commit that touched src/ when the lists were recorded
  sourceCommit: string;
  traces: Record<string, Trace>;
}

const newImpl: Impl = {
  build(cfg: TraceConfig, device: GPUDevice, t: Map<string, GPUBuffer>): PlanLike {
    const { spec, head } = cfg.shape === 'julia-1'
      ? specFromJuliaManifest(JULIA_SPEC)
      : specFromGlinerManifest(GLINER_SPECS[cfg.shape], {
        temperature: GLINER_TEMPERATURE, hiddenSize: GLINER_HEAD_HIDDEN[cfg.shape] });
    const plan = buildPlan(spec, head, {
      length: cfg.length, batch: cfg.batch, markers: cfg.markers, f16: cfg.f16 });
    return new PlanExecutor(device, plan, t) as unknown as PlanLike;
  },
  submit(plan, cfg, opts): void {
    (plan as unknown as PlanExecutor).submit(opts.capture, {
      seqLen: opts.seqLen, skip: opts.skip,
      qtype: cfg.shape === 'julia-1' ? (cfg.batch === 1 ? 0 : [0, 1, 2, 0]) : undefined,
    });
  },
};

function loadFrozen(): Frozen {
  return JSON.parse(gunzipSync(readFileSync(FROZEN)).toString('utf8')) as Frozen;
}

const configs = allConfigs();

{
  test('frozen file covers every configuration', () => {
    assert.ok(existsSync(FROZEN), 'run once with K28_TRACE_WRITE=1');
    const frozen = loadFrozen();
    assert.deepEqual(Object.keys(frozen.traces), configs.map((c) => c.id));
  });

  test('normal calls map exactly once', () => {
    const frozen = loadFrozen();
    for (const cfg of configs) {
      if (cfg.mode === 'normal' || cfg.mode === 'normal100' || cfg.mode === 'skip') {
        assert.equal(frozen.traces[cfg.id].maps, 1, cfg.id);
      }
    }
  });

  test('plan executor reproduces the frozen dispatch lists (G1, G3 plan part)', async () => {
    const frozen = loadFrozen();
    const bad: string[] = [];
    for (const cfg of configs) {
      const diff = firstDifference(frozen.traces[cfg.id], await recordTrace(cfg, newImpl));
      if (diff) bad.push(`${cfg.id}: ${diff}`);
    }
    assert.equal(bad.length, 0, `${bad.length} of ${configs.length} differ\n${bad.slice(0, 5).join('\n')}`);
  });

  test('plan build rejects a writable buffer bound twice', () => {
    const { spec, head } = specFromGlinerManifest(GLINER_SPECS.small, {
      temperature: 1, hiddenSize: 768 });
    const plan: Plan = buildPlan(spec, head, {
      length: 128, batch: 1, markers: 16, f16: false });
    const known = new Set(plan.buffers.map((b) => b.id));
    validateSegments(plan.segments, known);
    // lnA writes tmp; binding tmp as its residual input as well is invalid.
    const layer = plan.segments[1];
    const lnA = layer.ops.find((o) => o.name === 'lnA')!;
    lnA.bind[1] = lnA.bind[5];
    assert.throws(() => validateSegments(plan.segments, known), /L0\.lnA.*tmp.*twice/);
  });
}
