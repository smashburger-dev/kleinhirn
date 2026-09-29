// ORT Web (WebGPU EP) baseline bench, "model only" per docs/ARCHITECTURE.md:
// inputs come from the corpus goldens (padded L128 K16), 20 texts warmup
// discarded, then all items sequentially, one run at a time; clock from
// before input-tensor creation until the probabilities Float32Array lies in
// JS. Latency and parity are computed only with bench/metrics.ts.
// Query: ?model=small-upstream&precision=f32|f16&ep=webgpu|wasm&graph=std|opt|optgc
//   graph=opt loads model_<precision>_opt.onnx (offline-fused by
//   convert/optimize_onnx.py, K20); std is the plain export; optgc is opt
//   with the mask chain rebuilt for graph capture (convert/capture_surgery.py).
//   capture=1 (webgpu only): session option enableGraphCapture (ORT Web only
//   forwards it to the WebGPU EP when the EP is given as an object). Inputs and
//   outputs are persistent GPU buffers (Tensor.fromGpuBuffer), reused for
//   every run; the first run records, later runs replay. Clock: from the
//   first queue.writeBuffer of the four inputs until the probabilities
//   Float32Array lies in JS (one readback inside the window, ORT's cheapest
//   boundary); logits are read back outside the window for parity.
//   log=verbose: diagnostic only (node placement in the console), creates
//   the session, runs one item and reports no latency.

import * as ort from 'onnxruntime-web/webgpu';
import { compareLogits, summarizeLatency } from './metrics';

interface GoldenItem {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  logits: number[];
  label_index: number;
}

interface GoldenFile {
  labels: string[];
  count: number;
  items: GoldenItem[];
}

interface OrtResult {
  stage: string;
  model?: string;
  precision?: string;
  ep?: string;
  graph?: string;
  capture?: boolean;
  adapterInfo?: unknown;
  loadMs?: number;
  downloadBytes?: number;
  onnxBytes?: number;
  warmupMs?: number[];
  latency?: unknown;
  parity?: unknown;
  n?: number;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window {
    khOrtResult?: OrtResult;
  }
}

const LENGTH = 128;
const MAX_OPTIONS = 16;

function feedsFor(item: GoldenItem): Record<string, ort.Tensor> {
  const ids = new Int32Array(LENGTH);
  ids.set(item.input_ids);
  const mask = new Int32Array(LENGTH);
  mask.set(item.attention_mask);
  const markers = new Int32Array(MAX_OPTIONS);
  markers.set(item.marker_indices);
  const mmask = new Float32Array(MAX_OPTIONS);
  mmask.set(item.marker_mask);
  return {
    input_ids: new ort.Tensor('int32', ids, [1, LENGTH]),
    attention_mask: new ort.Tensor('int32', mask, [1, LENGTH]),
    marker_indices: new ort.Tensor('int32', markers, [1, MAX_OPTIONS]),
    marker_mask: new ort.Tensor('float32', mmask, [1, MAX_OPTIONS]),
  };
}

function resourceBytes(): number {
  let total = 0;
  for (const e of performance.getEntriesByType('resource') as PerformanceResourceTiming[]) {
    if (/model_f(16|32)(_opt(_gc)?)?\.onnx/.test(e.name)) continue; // counted as onnxBytes
    if (/onnx|ort-|\.wasm|ort\.ts|vite|metrics\.ts/.test(e.name)) {
      total += e.transferSize || e.encodedBodySize || e.decodedBodySize || 0;
    }
  }
  return total;
}

interface CaptureIo {
  feeds: Record<string, ort.Tensor>;
  write(item: GoldenItem): void;
  readProbabilities(out: ort.InferenceSession.OnnxValueMapType): Promise<Float32Array>;
  readLogits(out: ort.InferenceSession.OnnxValueMapType): Promise<Float32Array>;
}

async function createCaptureIo(): Promise<CaptureIo> {
  const device = await ort.env.webgpu.device;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buf = (bytes: number, u: number) => device.createBuffer({ size: bytes, usage: u });
  const idsBuf = buf(LENGTH * 4, usage);
  const maskBuf = buf(LENGTH * 4, usage);
  const markersBuf = buf(MAX_OPTIONS * 4, usage);
  const mmaskBuf = buf(MAX_OPTIONS * 4, usage);
  const staging = buf(MAX_OPTIONS * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const ids = new Int32Array(LENGTH);
  const mask = new Int32Array(LENGTH);
  const markers = new Int32Array(MAX_OPTIONS);
  const mmask = new Float32Array(MAX_OPTIONS);
  const gpu = (b: GPUBuffer, dataType: 'int32' | 'float32', len: number) =>
    ort.Tensor.fromGpuBuffer(b, { dataType, dims: [1, len] });
  const read = async (src: GPUBuffer): Promise<Float32Array> => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, MAX_OPTIONS * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  };
  return {
    feeds: {
      input_ids: gpu(idsBuf, 'int32', LENGTH),
      attention_mask: gpu(maskBuf, 'int32', LENGTH),
      marker_indices: gpu(markersBuf, 'int32', MAX_OPTIONS),
      marker_mask: gpu(mmaskBuf, 'float32', MAX_OPTIONS),
    },
    write(item) {
      ids.fill(0); ids.set(item.input_ids);
      mask.fill(0); mask.set(item.attention_mask);
      markers.fill(0); markers.set(item.marker_indices);
      mmask.fill(0); mmask.set(item.marker_mask);
      device.queue.writeBuffer(idsBuf, 0, ids);
      device.queue.writeBuffer(maskBuf, 0, mask);
      device.queue.writeBuffer(markersBuf, 0, markers);
      device.queue.writeBuffer(mmaskBuf, 0, mmask);
    },
    readProbabilities: (out) => read(out.probabilities.gpuBuffer),
    readLogits: (out) => read(out.logits.gpuBuffer),
  };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f32';
  const ep = params.get('ep') ?? 'webgpu';
  const graph = params.get('graph') ?? 'std';
  const verbose = params.get('log') === 'verbose';
  const capture = params.get('capture') === '1';
  const result: OrtResult = { stage: 'boot', model, precision, ep, graph, capture };
  window.khOrtResult = result;
  try {
    if (graph !== 'std' && graph !== 'opt' && graph !== 'optgc') throw new Error(`unknown graph ${graph}`);
    if (capture && ep !== 'webgpu') throw new Error('capture=1 needs ep=webgpu');
    if (ep === 'webgpu') {
      if (!navigator.gpu) throw new Error('navigator.gpu missing');
      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter?.info;
      result.adapterInfo = info ? {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      } : null;
    }

    const goldenRes = await fetch(`/tests/golden/${model}/texts1000_l128k16.json`);
    const golden = (await goldenRes.json()) as GoldenFile;
    const nLabels = golden.labels.length;

    result.stage = 'loading';
    const suffix = graph === 'std' ? '' : `_${graph.replace('optgc', 'opt_gc')}`;
    const modelRes = await fetch(`/models/${model}/onnx/model_${precision}${suffix}.onnx`);
    if (!modelRes.ok) throw new Error(`model download failed: ${modelRes.status}`);
    const modelBytes = await modelRes.arrayBuffer();
    result.onnxBytes = modelBytes.byteLength;

    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
    if (verbose) ort.env.logLevel = 'verbose';
    const tLoad = performance.now();
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: [capture ? { name: 'webgpu' } : ep],
      graphOptimizationLevel: 'all',
      ...(capture ? { enableGraphCapture: true } : {}),
      ...(verbose ? { logSeverityLevel: 0 as const, logVerbosityLevel: 0 } : {}),
    });
    result.loadMs = performance.now() - tLoad;
    result.downloadBytes = result.onnxBytes + resourceBytes();

    const io = capture ? await createCaptureIo() : null;

    if (verbose) {
      if (io) {
        io.write(golden.items[0]);
        await session.run(io.feeds);
      } else {
        await session.run(feedsFor(golden.items[0]));
      }
      result.stage = 'done';
      result.done = true;
      return;
    }

    const items = golden.items;
    const n = items.length;
    const ref = items.map((it) => it.logits.slice(0, nLabels));
    const cand: number[][] = new Array(n);
    const samplesMs: number[] = [];

    result.warmupMs = [];
    for (const item of items.slice(0, 20)) {
      const tw = performance.now();
      if (io) {
        io.write(item);
        await session.run(io.feeds);
      } else {
        const res = await session.run(feedsFor(item));
        void res;
      }
      if (result.warmupMs.length < 4) result.warmupMs.push(performance.now() - tw);
    }

    result.stage = 'running';
    for (let i = 0; i < n; i += 1) {
      if (io) {
        const t0 = performance.now();
        io.write(items[i]);
        const out = await session.run(io.feeds);
        const probs = await io.readProbabilities(out);
        const t1 = performance.now();
        void probs;
        samplesMs.push(t1 - t0);
        cand[i] = Array.from((await io.readLogits(out)).slice(0, nLabels));
        if (i % 100 === 0) result.stage = `running ${i}/${n}`;
        continue;
      }
      const t0 = performance.now();
      const res = await session.run(feedsFor(items[i]));
      const probsOut = Float32Array.from(
        (res.probabilities.data as Float32Array).slice(0, nLabels));
      const logitsRow = Array.from(
        (res.logits.data as Float32Array).slice(0, nLabels));
      const t1 = performance.now();
      void probsOut;
      samplesMs.push(t1 - t0);
      cand[i] = logitsRow;
      if (i % 100 === 0) result.stage = `running ${i}/${n}`;
    }

    result.latency = summarizeLatency(samplesMs);
    result.parity = compareLogits(ref, cand);
    result.n = n;
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
