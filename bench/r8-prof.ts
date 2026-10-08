// R8 op profile (docs/R8_WORKORDER.md Phase 0 step 2, Phase 1): one model on the WASM path from
// the source tree, full-length inputs of K28.8, `warmup` calls, then one call whose forward the
// worker host runs op by op `reps` times (WasmPlan.profile). Result: mean ms per op, per kernel
// kind, and the mean of whole forwards for the check that the parts add up.
// Query: model=<slug>  lens=128,512  warmup=3  reps=20  [khpost=<url>]

import { EncoderModel } from '../src/encoder.ts';
import type { OpProfile } from '../src/plan/wasm.ts';
import { workerTransport, type Transport } from '../src/wasm-backend.ts';
import { isolation, loadInputs } from './k28/k28-8-common.ts';
import { startPosting } from './kbench/post.ts';

interface LenResult {
  len: number;
  profile: OpProfile;
  byKernel: Record<string, { ms: number; ops: number; macs: number }>;
  sumOpsMs: number;
}

interface Result {
  stage: string; done?: boolean; error?: string;
  model?: string; ua?: string; crossOriginIsolated?: boolean; timerStepMs?: number;
  hardwareConcurrency?: number; info?: unknown; results: LenResult[];
}

declare global {
  interface Window { khR8Prof?: Result }
}

async function main(): Promise<void> {
  const p = new URLSearchParams(location.search);
  const slug = p.get('model') ?? '';
  const lens = (p.get('lens') ?? '128').split(',').map(Number);
  const warmup = Number(p.get('warmup') ?? '3');
  const reps = Number(p.get('reps') ?? '20');
  const result: Result = { stage: 'boot', model: slug, ua: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, results: [] };
  window.khR8Prof = result;
  startPosting(() => window.khR8Prof);
  try {
    Object.assign(result, isolation());
    for (const len of lens) {
      let arm = 0;
      let last: OpProfile | undefined;
      const transport = (): Transport => {
        const t = workerTransport();
        return {
          async request(msg, transfer) {
            const m = msg.type === 'run' && arm ? { ...msg, profile: arm } : msg;
            const r = await t.request(m, transfer);
            if (r.profile) last = r.profile;
            return r;
          },
          close: () => t.close(),
        };
      };
      result.stage = `loading L${len}`;
      const enc = await EncoderModel.load({
        manifestUrl: `/models/k28/${slug}/f32/manifest.json`, precision: 'f32', buckets: [len],
        limits: 'minimum', backend: 'wasm',
      }, { wasmTransport: transport });
      const inputs = await loadInputs(slug, len);
      result.stage = `warmup L${len}`;
      for (let i = 0; i < warmup; i += 1) await enc.runIds({ inputIds: inputs.row(i) });
      result.stage = `profile L${len}`;
      arm = reps;
      await enc.runIds({ inputIds: inputs.row(warmup) });
      arm = 0;
      result.info = enc.info();
      enc.dispose();
      if (!last) throw new Error('no profile came back');
      const byKernel: LenResult['byKernel'] = {};
      last.ops.forEach((op, i) => {
        const k = (byKernel[op.kernel] ??= { ms: 0, ops: 0, macs: 0 });
        k.ms += last!.ms[i];
        k.ops += 1;
        if (op.N !== undefined && op.K !== undefined) k.macs += Math.min(op.M ?? len, len) * op.N * op.K;
      });
      result.results.push({ len, profile: last, byKernel, sumOpsMs: last.ms.reduce((a, b) => a + b, 0) });
    }
    result.stage = 'done';
    result.done = true;
  } catch (e) {
    result.stage = 'error';
    result.error = String(e);
    result.done = true;
  }
}

void main();
