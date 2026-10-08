// R1 bench page (docs/R1_WORKORDER.md). One loop for every engine behind bench/kbench/engine.ts:
// load, warm-up, measured window, output log, parity afterwards (outside the window).
//
// Query:
//   mode=env                         WebGPU, shader-f16, isolation, timer step; no model
//   engine=kleinhirn|ort  model=<slug>  task=<task>  way=webgpu-f16|webgpu-f32|wasm
//   khpost=<url>                     post stage and result there (bench/kbench/post.ts, firefox-reg)
//   len=128|512|real                 full length (inputs of K28.8) or the 200 golden cases
//   n=<measured calls>  warmup=<calls>  block=auto|1|10  acc=0 (skip the parity pass)
//   ORT only: build=jsep|webgpu|jspi  capture=0|1  graph=std|opt  dynamic=0|1  threads=1|hc|<n>
// Full length: inputs 0..warmup-1 warm up, then n calls on inputs 20..19+n (as K28.8).
// Real length: the golden cases in golden order, ungepadded; warm-up on cases 0..warmup-1, the
// measured window runs cases 0..n-1 (n = 200 is all of them). The smallest of {128, 512} that
// fits is the bucket for both engines (Festlegung 1).

import { summarizeLatency } from './metrics.ts';
import type { BenchEngine, Way } from './kbench/engine.ts';
import { BUCKETS } from './kbench/engine.ts';
import { KleinhirnEngine } from './kbench/adapters/kleinhirn.ts';
import { OrtEngine } from './kbench/adapters/ort.ts';
import { blockForTimerStep, measureWindow } from './kbench/measure.ts';
import { startPosting } from './kbench/post.ts';
import {
  OutputLog, allFinite, goldenAccuracy, isolation, loadInputs, resourceBytes,
} from './k28/k28-8-common.ts';

interface Result {
  stage: string;
  mode?: string; engine?: string; model?: string; way?: string; len?: string; n?: number; warmup?: number;
  block?: number; crossOriginIsolated?: boolean; timerStepMs?: number;
  env?: Record<string, unknown>;
  info?: unknown; loadMs?: number;
  download?: { bytes: number; files: Record<string, number> };
  engineFiles?: string[];
  graphBytes?: number;
  warmupMs?: number[]; warmupFinite?: boolean; latency?: unknown; samplesMs?: number[];
  outputs?: { rows: number; width: number; nonFinite: number; base64: string; firstInput: number };
  accuracy?: Record<string, unknown>;
  error?: string; done?: boolean;
}

declare global {
  interface Window { khKbenchResult?: Result }
}

async function envProbe(): Promise<Record<string, unknown>> {
  const env: Record<string, unknown> = {
    userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined', webgpu: 'gpu' in navigator,
  };
  if ('gpu' in navigator) {
    const adapter = await navigator.gpu.requestAdapter();
    env.adapter = adapter ? {
      vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
      features: [...adapter.features], shaderF16: adapter.features.has('shader-f16'),
      subgroups: adapter.features.has('subgroups'),
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    } : null;
  }
  env.wasm = wasmProbe();
  return env;
}

// R2 stage 0: what the WASM path can use. SIMD and relaxed SIMD by validating one-function
// modules (v128.const; three v128.const and f32x4.relaxed_madd); the largest memory that grows,
// in steps of 64 MiB from one page, plain and shared.
function wasmProbe(): Record<string, unknown> {
  const c = [0xfd, 0x0c, ...new Array<number>(16).fill(0)];
  const mod = (body: number[]) => new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 1, 5, 1, 0x60, 0, 1, 0x7b,
    3, 2, 1, 0, 10, body.length + 2, 1, body.length, ...body]);
  const grow = (shared: boolean): { maxMiB: number; error?: string } => {
    let mem: WebAssembly.Memory | undefined;
    let maximum = 65536;
    for (; maximum >= 1024 && !mem; maximum /= 2) {
      try { mem = new WebAssembly.Memory({ initial: 1, maximum, shared }); } catch { /* next smaller maximum */ }
    }
    if (!mem) return { maxMiB: 0, error: 'no memory with a maximum of 1024 pages or more' };
    let pages = 1;
    let error: string | undefined;
    while (pages < 65536) {
      const step = Math.min(1024, 65536 - pages);
      try { mem.grow(step); pages += step; } catch (e) { error = String(e instanceof Error ? e.message : e).slice(0, 120); break; }
    }
    return { maxMiB: pages / 16, ...(error ? { error } : {}) };
  };
  return {
    simd: WebAssembly.validate(mod([0, ...c, 0x0b])),
    relaxedSimd: WebAssembly.validate(mod([0, ...c, ...c, ...c, 0xfd, 0x85, 0x02, 0x0b])),
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    hardwareConcurrency: navigator.hardwareConcurrency,
    memory: grow(false),
    sharedMemory: typeof SharedArrayBuffer !== 'undefined' ? grow(true) : null,
  };
}

function makeEngine(name: string): BenchEngine {
  if (name === 'kleinhirn') return new KleinhirnEngine();
  if (name === 'kleinhirn-ref') return new KleinhirnEngine('dist-ref/kleinhirn.js');
  if (name === 'ort') return new OrtEngine();
  throw new Error(`unknown engine ${name}`);
}

async function main(): Promise<void> {
  const p = new URLSearchParams(location.search);
  const mode = p.get('mode') ?? 'run';
  const slug = p.get('model') ?? '';
  const way = (p.get('way') ?? 'webgpu-f16') as Way;
  const len = p.get('len') ?? '128';
  const result: Result = { stage: 'boot', mode, engine: p.get('engine') ?? undefined, model: slug, way, len };
  window.khKbenchResult = result;
  startPosting(() => window.khKbenchResult);
  let engine: BenchEngine | null = null;
  try {
    Object.assign(result, isolation());
    result.block = blockForTimerStep(result.timerStepMs as number);
    if (mode === 'env') {
      result.env = await envProbe();
      result.stage = 'done';
      result.done = true;
      return;
    }
    const task = p.get('task') ?? '';
    const warmup = Number(p.get('warmup') ?? (way === 'wasm' ? '5' : '20'));
    const n = Number(p.get('n') ?? (way === 'wasm' ? '50' : '100'));
    const blockParam = p.get('block') ?? 'auto';
    const block = blockParam === 'auto' ? (result.block as number) : Number(blockParam);
    result.n = n; result.warmup = warmup; result.block = block;

    engine = makeEngine(p.get('engine') ?? '');
    const real = len === 'real';
    const length = real ? 0 : Number(len);
    const buckets = real ? [...BUCKETS] : [length];

    // Inputs
    const meta128 = await loadInputs(slug, 128); // pad id, also for the real mode
    const padId = meta128.meta.padId ?? 0;
    let rowAt: (i: number) => Int32Array;
    let firstMeasured: number;
    let rows: number;
    let goldenIds: number[][] = [];
    if (real) {
      const golden = await (await fetch(`/tests/golden/k28/${slug}/${task}.json`)).json() as { items: { input_ids: number[] }[] };
      goldenIds = golden.items.map((it) => it.input_ids);
      rows = goldenIds.length;
      rowAt = (i) => Int32Array.from(goldenIds[i]);
      firstMeasured = 0;
    } else {
      const inputs = await loadInputs(slug, length);
      rows = inputs.meta.rows;
      rowAt = (i) => inputs.row(i);
      firstMeasured = inputs.meta.warmup;
    }
    if (firstMeasured + n > rows) throw new Error(`n ${n} exceeds the inputs (${rows})`);

    result.stage = 'loading';
    const loadInfo = await engine.load({
      model: slug, task, way, buckets,
      options: {
        build: p.get('build') ?? 'webgpu', capture: p.get('capture') === '1', graph: p.get('graph') ?? 'std',
        dynamic: p.get('dynamic') === '1', threads: p.get('threads') ?? '1', padId,
        ...(p.get('split') ? { split: p.get('split') } : {}),
      },
    });
    result.loadMs = loadInfo.loadMs;

    result.stage = 'warmup';
    result.warmupMs = [];
    let warmFinite = true;
    for (let i = 0; i < warmup; i += 1) {
      const tw = performance.now();
      const o = await engine.run({ ids: rowAt(i) });
      result.warmupMs.push(performance.now() - tw);
      if (!allFinite(o)) warmFinite = false;
    }
    result.warmupFinite = warmFinite;

    result.stage = 'running';
    const win = await measureWindow((i) => engine!.run({ ids: rowAt(i) }), firstMeasured, n, block,
      () => performance.now(), (done) => { result.stage = `running ${done}/${n}`; });
    const log = new OutputLog();
    for (const o of win.outputs) log.push(o);
    result.latency = summarizeLatency(win.samplesMs);
    result.samplesMs = win.samplesMs;
    result.outputs = { rows: log.count, width: log.width, nonFinite: log.nonFinite, base64: log.base64(), firstInput: firstMeasured };
    result.info = engine.info(); // gpuBytes after the runs
    // hold=<ms> (memory pass): keep the model loaded and busy on the measured inputs, so the
    // 1 Hz footprint sampler sees the compute phase even when the measured calls take under a second.
    const hold = Number(p.get('hold') ?? '0');
    if (hold > 0) {
      result.stage = 'hold';
      const th = performance.now();
      for (let i = 0; performance.now() - th < hold; i += 1) await engine.run({ ids: rowAt(firstMeasured + (i % n)) });
    }

    result.stage = 'accuracy';
    if (p.get('acc') === '0') {
      result.accuracy = { skipped: true };
    } else if (real && n === rows) {
      // Parity straight from the measured outputs: the goldens, case by case in golden order.
      let k = 0;
      result.accuracy = await goldenAccuracy(slug, task, Number.MAX_SAFE_INTEGER, async () => win.outputs[k++]);
    } else {
      const eng = engine;
      result.accuracy = await goldenAccuracy(slug, task, real ? Number.MAX_SAFE_INTEGER : length,
        async (ids) => Float32Array.from(await eng.run({ ids: Int32Array.from(ids) })));
    }
    const kh = engine.name.startsWith('kleinhirn');
    const files = resourceBytes(kh ? new RegExp(`/models/k28/${slug}/(f16|f32)/|/dist(-ref)?/kleinhirn\\.js`) : /onnxruntime-web|ort-wasm/);
    const ortEngineFiles = kh ? { bytes: 0, files: {} } : files;
    if (!kh) {
      const o = engine as OrtEngine;
      result.graphBytes = o.graphBytes;
      result.download = { bytes: o.graphBytes + o.tokenizerBytes + files.bytes,
        files: { graph: o.graphBytes, tokenizer: o.tokenizerBytes, ...files.files } };
      result.engineFiles = Object.keys(ortEngineFiles.files);
    } else {
      result.download = { bytes: files.bytes, files: files.files };
      result.engineFiles = [engine.name === 'kleinhirn' ? '/dist/kleinhirn.js' : '/dist-ref/kleinhirn.js'];
    }
    await engine.dispose();
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
    try { await engine?.dispose(); } catch { /* ignore */ }
  }
}

void main();
