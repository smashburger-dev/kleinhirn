// K28.8 ORT side (docs/K28_8_WORKORDER.md, phase 1 step 6). One graph of
// convert/k28_8_export.py in one ORT Web build, f16, fixed shape [1, L].
// Inputs: the measurement inputs of convert/k28_8_inputs.py, exactly L tokens,
// mask all 1; inputs 0..19 warm up, then n measured calls (inputs 20..19+n),
// one run at a time, one readback per call.
// Window (Festlegung 4): without capture from the creation of the input
// tensors until the output lies in JS as a Float32Array; with capture from the
// first writeBuffer of the inputs until the output lies in JS (copy to a
// staging buffer, mapAsync). The outputs of the measured calls are kept for
// Festlegung 10 (read outside the window). After the measurement the golden
// cases up to L tokens (padded to L, mask 0 on padding) give the accuracy.
// Query: ?model=<slug>&L=128|512&graph=std|opt|opt-noemb|opt-mha&build=jsep|webgpu|jspi
//        &capture=0|1&n=300&log=verbose
// log=verbose: diagnostic only (node placement in the console): session, one call, no latency.
// Native WebGPU builds (webgpu, jspi) get validationMode 'wgpuOnly' (Festlegung 7).

import type * as OrtNs from 'onnxruntime-web';
import { summarizeLatency } from './metrics.ts';
import {
  OutputLog, allFinite, goldenAccuracy, isolation, loadInputs, resourceBytes,
} from './k28/k28-8-common.ts';

type Ort = typeof OrtNs;

interface Result {
  stage: string;
  model?: string; length?: number; graph?: string; build?: string; capture?: boolean; n?: number;
  ortVersion?: string; validationMode?: string | null;
  crossOriginIsolated?: boolean; timerStepMs?: number;
  adapterInfo?: unknown; loadMs?: number; onnxBytes?: number;
  download?: { bytes: number; files: Record<string, number> };
  warmupMs?: number[]; latency?: unknown; samplesMs?: number[];
  outputs?: { rows: number; width: number; nonFinite: number; base64: string; firstInput: number };
  warmupFinite?: boolean; accuracy?: Record<string, unknown>;
  error?: string; done?: boolean;
}

declare global {
  interface Window { khK28OrtResult?: Result }
}

async function importBuild(build: string): Promise<Ort> {
  if (build === 'jsep') return (await import('onnxruntime-web')) as unknown as Ort;
  if (build === 'webgpu') return (await import('onnxruntime-web/webgpu')) as unknown as Ort;
  if (build === 'jspi') return (await import('onnxruntime-web/jspi')) as unknown as Ort;
  throw new Error(`unknown build ${build}`);
}

async function main(): Promise<void> {
  const p = new URLSearchParams(location.search);
  const slug = p.get('model') ?? '';
  const length = Number(p.get('L') ?? '128');
  const graph = p.get('graph') ?? 'opt';
  const build = p.get('build') ?? 'webgpu';
  const capture = p.get('capture') === '1';
  const n = Number(p.get('n') ?? '300');
  const verbose = p.get('log') === 'verbose';
  const result: Result = { stage: 'boot', model: slug, length, graph, build, capture, n };
  window.khK28OrtResult = result;
  try {
    Object.assign(result, isolation());
    if (!navigator.gpu) throw new Error('navigator.gpu missing');
    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter?.info;
    result.adapterInfo = info ? { vendor: info.vendor, architecture: info.architecture } : null;

    const inputs = await loadInputs(slug, length);
    if (inputs.meta.warmup + n > inputs.meta.rows) throw new Error(`n ${n} exceeds the inputs`);
    const ort = await importBuild(build);
    result.ortVersion = ort.env.versions?.web;

    result.stage = 'loading';
    const graphRes = await fetch(`/models/k28/${slug}/k28.8/onnx/L${length}-${graph}-f16.onnx`);
    if (!graphRes.ok) throw new Error(`graph download failed: ${graphRes.status}`);
    const graphBytes = await graphRes.arrayBuffer();
    result.onnxBytes = graphBytes.byteLength;
    // The tokenizer a real page would load; fetched only to count its bytes.
    const tok = await fetch(`/models/k28/${slug}/f16/tokenizer.json`);
    const tokenizerBytes = (await tok.arrayBuffer()).byteLength;

    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
    if (verbose) ort.env.logLevel = 'verbose';
    const native = build !== 'jsep';
    result.validationMode = native ? 'wgpuOnly' : null;
    const ep = { name: 'webgpu', ...(native ? { validationMode: 'wgpuOnly' } : {}) };
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(graphBytes, {
      executionProviders: [ep as OrtNs.InferenceSession.ExecutionProviderConfig],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: capture ? 'gpu-buffer' : 'cpu',
      ...(capture ? { enableGraphCapture: true } : {}),
      ...(verbose ? { logSeverityLevel: 0 as const, logVerbosityLevel: 0 } : {}),
    });
    result.loadMs = performance.now() - t0;

    const L = length;
    const mask = new Int32Array(L).fill(1);
    let call: (ids: Int32Array<ArrayBuffer>, m: Int32Array<ArrayBuffer>) => Promise<Float32Array>;
    if (capture) {
      const device = (await ort.env.webgpu.device) as GPUDevice;
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      const idsBuf = device.createBuffer({ size: L * 4, usage });
      const maskBuf = device.createBuffer({ size: L * 4, usage });
      const feeds = {
        input_ids: ort.Tensor.fromGpuBuffer(idsBuf, { dataType: 'int32', dims: [1, L] }),
        attention_mask: ort.Tensor.fromGpuBuffer(maskBuf, { dataType: 'int32', dims: [1, L] }),
      };
      let staging: GPUBuffer | null = null;
      call = async (ids, m) => {
        device.queue.writeBuffer(idsBuf, 0, ids);
        device.queue.writeBuffer(maskBuf, 0, m);
        const out = await session.run(feeds);
        const t = out.output;
        const bytes = t.dims.reduce((a, b) => a * b, 1) * 4;
        staging ??= device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(t.gpuBuffer as GPUBuffer, 0, staging, 0, bytes);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return data;
      };
    } else {
      call = async (ids, m) => {
        const out = await session.run({
          input_ids: new ort.Tensor('int32', ids, [1, L]),
          attention_mask: new ort.Tensor('int32', m, [1, L]),
        });
        return out.output.data as Float32Array;
      };
    }

    if (verbose) {
      await call(inputs.row(0), mask);
      result.stage = 'done';
      result.done = true;
      return;
    }

    result.warmupMs = [];
    let warmFinite = true;
    for (let i = 0; i < inputs.meta.warmup; i += 1) {
      const tw = performance.now();
      const o = await call(inputs.row(i), mask);
      result.warmupMs.push(performance.now() - tw);
      if (!allFinite(o)) warmFinite = false;
    }
    result.warmupFinite = warmFinite;

    result.stage = 'running';
    const samples: number[] = [];
    const log = new OutputLog();
    for (let k = 0; k < n; k += 1) {
      const row = inputs.row(inputs.meta.warmup + k);
      const t1 = performance.now();
      const o = await call(row, mask);
      const t2 = performance.now();
      samples.push(t2 - t1);
      log.push(o);
      if (k % 50 === 0) result.stage = `running ${k}/${n}`;
    }
    result.latency = summarizeLatency(samples);
    result.samplesMs = samples;
    result.outputs = {
      rows: log.count, width: log.width, nonFinite: log.nonFinite, base64: log.base64(),
      firstInput: inputs.meta.warmup,
    };

    result.stage = 'accuracy';
    const pad = inputs.meta.padId ?? 0;
    result.accuracy = await goldenAccuracy(slug, inputs.meta.task, L, async (ids) => {
      const x = new Int32Array(L).fill(pad);
      const m = new Int32Array(L);
      x.set(ids);
      m.fill(1, 0, ids.length);
      return Float32Array.from(await call(x, m));
    });
    // ORT JS (Vite may serve it pre-bundled from .vite/deps) and the wasm file of the build.
    const ortFiles = resourceBytes(/onnxruntime-web|ort-wasm/);
    result.download = {
      bytes: result.onnxBytes + tokenizerBytes + ortFiles.bytes,
      files: { graph: result.onnxBytes, tokenizer: tokenizerBytes, ...ortFiles.files },
    };
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
