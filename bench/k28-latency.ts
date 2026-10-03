// K28.8 kleinhirn side (docs/K28_8_WORKORDER.md, phase 1 step 7). The library
// bundle dist/kleinhirn.js (the runner checks its build id against the file on
// disk), EncoderModel.load with the f16 manifest, buckets [L], minimum limits.
// Inputs: the measurement inputs of convert/k28_8_inputs.py, exactly L tokens;
// inputs 0..19 warm up, then n measured calls (inputs 20..19+n), one at a time.
// Window (Festlegung 4): from runIds until the output lies in JS as a
// Float32Array (row gather, upload, validation scope, submit, readback,
// finiteness check, all inside runIds). The outputs of the measured calls are
// kept for Festlegung 10. After the measurement the golden cases up to L
// tokens give the accuracy (outside the window).
// Query: ?model=<slug>&L=128|512&n=300

// @ts-expect-error runtime bundle built by vite lib mode has no d.ts
import { EncoderModel as BundleEncoderModel } from '../dist/kleinhirn.js';
import type { EncoderModel as EncoderModelType } from '../src/encoder.ts';
import { summarizeLatency } from './metrics.ts';
import {
  OutputLog, allFinite, goldenAccuracy, isolation, loadInputs, resourceBytes,
} from './k28/k28-8-common.ts';

interface Result {
  stage: string;
  model?: string; length?: number; n?: number;
  crossOriginIsolated?: boolean; timerStepMs?: number;
  info?: Record<string, unknown>; loadMs?: number;
  download?: { bytes: number; files: Record<string, number> };
  warmupMs?: number[]; warmupFinite?: boolean; latency?: unknown; samplesMs?: number[];
  outputs?: { rows: number; width: number; nonFinite: number; base64: string; firstInput: number };
  accuracy?: Record<string, unknown>;
  error?: string; done?: boolean;
}

const EncoderModel = BundleEncoderModel as typeof EncoderModelType;

declare global {
  interface Window { khK28LatencyResult?: Result }
}

async function main(): Promise<void> {
  const p = new URLSearchParams(location.search);
  const slug = p.get('model') ?? '';
  const length = Number(p.get('L') ?? '128');
  const n = Number(p.get('n') ?? '300');
  const result: Result = { stage: 'boot', model: slug, length, n };
  window.khK28LatencyResult = result;
  try {
    Object.assign(result, isolation());
    const inputs = await loadInputs(slug, length);
    if (inputs.meta.warmup + n > inputs.meta.rows) throw new Error(`n ${n} exceeds the inputs`);

    result.stage = 'loading';
    const t0 = performance.now();
    const enc = await EncoderModel.load({
      manifestUrl: `/models/k28/${slug}/f16/manifest.json`, precision: 'f16',
      buckets: [length], limits: 'minimum',
    });
    result.loadMs = performance.now() - t0;
    const info = enc.info();
    result.info = info;
    if (info.precision !== 'f16') throw new Error(`engine runs ${String(info.precision)}`);

    result.warmupMs = [];
    let warmFinite = true;
    for (let i = 0; i < inputs.meta.warmup; i += 1) {
      const tw = performance.now();
      const o = await enc.runIds({ inputIds: inputs.row(i) });
      result.warmupMs.push(performance.now() - tw);
      if (!allFinite(o.data)) warmFinite = false;
    }
    result.warmupFinite = warmFinite;

    result.stage = 'running';
    const samples: number[] = [];
    const log = new OutputLog();
    for (let k = 0; k < n; k += 1) {
      const row = inputs.row(inputs.meta.warmup + k);
      const t1 = performance.now();
      const o = await enc.runIds({ inputIds: row });
      const t2 = performance.now();
      samples.push(t2 - t1);
      log.push(o.data);
      if (k % 50 === 0) result.stage = `running ${k}/${n}`;
    }
    result.latency = summarizeLatency(samples);
    result.samplesMs = samples;
    result.outputs = {
      rows: log.count, width: log.width, nonFinite: log.nonFinite, base64: log.base64(),
      firstInput: inputs.meta.warmup,
    };
    result.info = enc.info(); // gpuBytes after the runs

    result.stage = 'accuracy';
    result.accuracy = await goldenAccuracy(slug, inputs.meta.task, length,
      async (ids) => (await enc.runIds({ inputIds: ids })).data);
    const shards = resourceBytes(new RegExp(`/models/k28/${slug}/f16/`));
    const bundle = resourceBytes(/\/dist\/kleinhirn\.js/);
    result.download = { bytes: shards.bytes + bundle.bytes, files: { ...shards.files, ...bundle.files } };
    enc.dispose();
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
