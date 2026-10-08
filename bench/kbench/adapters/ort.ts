// R1 ORT Web adapter (patterns of bench/k28-ort.ts). Options:
//   build     jsep | webgpu | jspi   (wasm way: jsep; the wasm file is the same CPU runtime in all builds)
//   capture   WebGPU graph capture, fixed shapes only
//   graph     std | opt              graph type of the K28.8 winner
//   dynamic   one graph with a dynamic sequence axis, run at the real length without padding
//   threads   wasm only: 1 or a number (numThreads)
// Fixed graphs: one session per bucket, the input is padded to the smallest bucket that fits
// (pad id of the model, mask 0 on padding). Graph files: models/k28/<slug>/k28.8/onnx/
// L<bucket>-<graph>-<f16|f32>.onnx, dynamic Ldyn-<graph>-<f16|f32>.onnx.

import type * as OrtNs from 'onnxruntime-web';
import { bucketFor, type BenchEngine, type BenchInput, type EngineInfo, type LoadInfo, type LoadSpec } from '../engine.ts';

type Ort = typeof OrtNs;

interface Slot {
  session: OrtNs.InferenceSession;
  length: number | null; // null: dynamic
  call: (ids: Int32Array, mask: Int32Array) => Promise<Float32Array>;
}

async function importBuild(build: string): Promise<Ort> {
  if (build === 'jsep') return (await import('onnxruntime-web')) as unknown as Ort;
  if (build === 'webgpu') return (await import('onnxruntime-web/webgpu')) as unknown as Ort;
  if (build === 'jspi') return (await import('onnxruntime-web/jspi')) as unknown as Ort;
  throw new Error(`unknown build ${build}`);
}

export class OrtEngine implements BenchEngine {
  readonly name = 'ort-web';
  private ort: Ort | null = null;
  private slots: Slot[] = [];
  private spec: LoadSpec | null = null;
  private padId = 0;
  private threads = 1;
  graphBytes = 0;
  tokenizerBytes = 0;
  private adapter: unknown = null;

  async load(spec: LoadSpec): Promise<LoadInfo> {
    this.spec = spec;
    const o = spec.options;
    const wasm = spec.way === 'wasm';
    const build = wasm ? 'jsep' : String(o.build ?? 'webgpu');
    const capture = !wasm && o.capture === true;
    const dynamic = o.dynamic === true;
    if (capture && dynamic) throw new Error('graph capture needs fixed shapes');
    const graph = String(o.graph ?? 'std');
    const prec = spec.way === 'webgpu-f16' ? 'f16' : 'f32';
    this.padId = Number(o.padId ?? 0);
    this.ort = await importBuild(build);
    const ort = this.ort;
    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
    if (wasm) {
      this.threads = o.threads === 'hc' ? navigator.hardwareConcurrency : Number(o.threads ?? 1);
      ort.env.wasm.numThreads = this.threads;
    } else {
      const adapter = await navigator.gpu.requestAdapter();
      const i = adapter?.info;
      this.adapter = i ? { vendor: i.vendor, architecture: i.architecture } : null;
    }
    const tok = await fetch(`/models/k28/${spec.model}/f16/tokenizer.json`); // a real page loads it; counted only
    this.tokenizerBytes = (await tok.arrayBuffer()).byteLength;

    const t0 = performance.now();
    const native = !wasm && build !== 'jsep';
    const ep = wasm ? 'wasm' : { name: 'webgpu', ...(native ? { validationMode: 'wgpuOnly' } : {}) };
    const files = dynamic ? [`Ldyn-${graph}-${prec}.onnx`] : spec.buckets.map((b) => `L${b}-${graph}-${prec}.onnx`);
    for (const [idx, file] of files.entries()) {
      const res = await fetch(`/models/k28/${spec.model}/k28.8/onnx/${file}`);
      if (!res.ok) throw new Error(`graph download failed: ${file} ${res.status}`);
      const bytes = await res.arrayBuffer();
      this.graphBytes += bytes.byteLength;
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep as OrtNs.InferenceSession.ExecutionProviderConfig],
        graphOptimizationLevel: 'all',
        preferredOutputLocation: capture ? 'gpu-buffer' : 'cpu',
        ...(capture ? { enableGraphCapture: true } : {}),
      });
      const length = dynamic ? null : spec.buckets[idx];
      this.slots.push({ session, length, call: await this.makeCall(session, length, capture) });
    }
    return { loadMs: performance.now() - t0 };
  }

  private async makeCall(session: OrtNs.InferenceSession, length: number | null, capture: boolean): Promise<Slot['call']> {
    const ort = this.ort as Ort;
    if (!capture) {
      return async (ids, mask) => {
        const n = ids.length;
        const out = await session.run({
          input_ids: new ort.Tensor('int32', ids, [1, n]),
          attention_mask: new ort.Tensor('int32', mask, [1, n]),
        });
        return out.output.data as Float32Array;
      };
    }
    const L = length as number;
    const device = (await ort.env.webgpu.device) as GPUDevice;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const idsBuf = device.createBuffer({ size: L * 4, usage });
    const maskBuf = device.createBuffer({ size: L * 4, usage });
    const feeds = {
      input_ids: ort.Tensor.fromGpuBuffer(idsBuf, { dataType: 'int32', dims: [1, L] }),
      attention_mask: ort.Tensor.fromGpuBuffer(maskBuf, { dataType: 'int32', dims: [1, L] }),
    };
    let staging: GPUBuffer | null = null;
    return async (ids, mask) => {
      device.queue.writeBuffer(idsBuf, 0, ids as Int32Array<ArrayBuffer>);
      device.queue.writeBuffer(maskBuf, 0, mask as Int32Array<ArrayBuffer>);
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
  }

  async run(input: BenchInput): Promise<Float32Array> {
    const n = input.ids.length;
    const dyn = this.slots.find((s) => s.length === null);
    if (dyn) return dyn.call(input.ids, new Int32Array(n).fill(1));
    const bucket = bucketFor(n, this.slots.map((s) => s.length as number));
    const slot = this.slots.find((s) => s.length === bucket) as Slot;
    const ids = new Int32Array(bucket).fill(this.padId);
    const mask = new Int32Array(bucket);
    ids.set(input.ids);
    mask.fill(1, 0, n);
    return slot.call(ids, mask);
  }

  info(): EngineInfo {
    const o = this.spec?.options ?? {};
    return {
      name: this.name, version: this.ort?.env.versions?.web ?? '', build: this.spec?.way === 'wasm' ? 'jsep' : String(o.build ?? ''),
      config: { ...o, threads: this.spec?.way === 'wasm' ? this.threads : null, way: this.spec?.way },
      gpuBytes: null, adapter: this.adapter,
    };
  }

  async dispose(): Promise<void> {
    for (const s of this.slots) await s.session.release();
    this.slots = [];
  }
}
