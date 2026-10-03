// K27 diagnosis: what ORT Web does on the GPU for one K28.8 graph. Not a latency page.
// Records, by wrapping the WebGPU API before ORT loads:
// - requestDevice: required limits and features (does ORT stay inside the minimum limits?);
// - every WGSL shader ORT creates (deduplicated by text) and its compute pipelines;
// - the dispatches of one session.run (pipeline, workgroup counts) after warmup;
// - per-dispatch GPU time: every compute pass of the traced run gets timestamp writes
//   at its start and end, and passes with several dispatches are reported as one entry.
// Query: ?model=<slug>&L=128&graph=std&build=webgpu&calls=10

import type * as OrtNs from 'onnxruntime-web';
import { loadInputs } from './k28/k28-8-common.ts';

type Ort = typeof OrtNs;

interface Shader { id: number; hash: string; code: string; workgroupSize: string | null; workgroupVars: string[] }
interface Dispatch { pass: number; shader: number; label: string; x: number; y: number; z: number }
interface Result {
  stage: string; device?: unknown; adapterLimits?: Record<string, number>; shaders?: Shader[];
  dispatches?: Dispatch[]; passTimesUs?: { pass: number; us: number; dispatches: number }[];
  runMs?: number[]; error?: string; done?: boolean;
}
declare global {
  interface Window { khK28OrtTrace?: Result }
}

const result: Result = { stage: 'boot' };
window.khK28OrtTrace = result;

const shaders: Shader[] = [];
const shaderOf = new WeakMap<GPUShaderModule, number>();
const pipelineShader = new WeakMap<GPUComputePipeline, number>();
let tracing = false;
let passNo = 0;
const dispatches: Dispatch[] = [];
const currentPipeline = new WeakMap<GPUComputePassEncoder, GPUComputePipeline>();
const passOf = new WeakMap<GPUComputePassEncoder, number>();
let query: { set: GPUQuerySet; resolve: GPUBuffer; read: GPUBuffer; used: number } | null = null;
const QUERIES = 4096;

function hashText(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(16);
}

function wrap(): void {
  const reqDev = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function patched(this: GPUAdapter, desc?: GPUDeviceDescriptor) {
    const limits: Record<string, number> = {};
    for (const k in this.limits) limits[k] = (this.limits as unknown as Record<string, number>)[k];
    result.adapterLimits = limits;
    const features = new Set(desc?.requiredFeatures ?? []);
    if (this.features.has('timestamp-query')) features.add('timestamp-query');
    const device = await reqDev.call(this, { ...desc, requiredFeatures: [...features] });
    result.device = { requiredLimits: desc?.requiredLimits ?? null, requiredFeatures: desc?.requiredFeatures ?? [] };
    return device;
  };
  const csm = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function patched(this: GPUDevice, d: GPUShaderModuleDescriptor) {
    const m = csm.call(this, d);
    const code = d.code;
    const hash = hashText(code);
    let s = shaders.find((x) => x.hash === hash);
    if (!s) {
      s = {
        id: shaders.length, hash, code,
        workgroupSize: code.match(/@workgroup_size\(([^)]*)\)/)?.[1] ?? null,
        workgroupVars: [...code.matchAll(/var<workgroup>\s*([^;]*);/g)].map((x) => x[1]),
      };
      shaders.push(s);
    }
    shaderOf.set(m, s.id);
    return m;
  };
  const note = (p: GPUComputePipeline, d: GPUComputePipelineDescriptor) => {
    pipelineShader.set(p, shaderOf.get(d.compute.module) ?? -1);
    return p;
  };
  const ccp = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function patched(this: GPUDevice, d: GPUComputePipelineDescriptor) {
    return note(ccp.call(this, d), d);
  };
  const ccpa = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = async function patched(this: GPUDevice, d: GPUComputePipelineDescriptor) {
    return note(await ccpa.call(this, d), d);
  };
  const bcp = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function patched(this: GPUCommandEncoder, d?: GPUComputePassDescriptor) {
    if (!tracing || !query || d?.timestampWrites || query.used + 2 > QUERIES) return bcp.call(this, d);
    const n = passNo++;
    const begin = query.used; query.used += 2;
    const pass = bcp.call(this, { ...d, timestampWrites: { querySet: query.set, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 } });
    passOf.set(pass, n);
    return pass;
  };
  const sp = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function patched(this: GPUComputePassEncoder, p: GPUComputePipeline) {
    currentPipeline.set(this, p);
    return sp.call(this, p);
  };
  const dw = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function patched(this: GPUComputePassEncoder, x: number, y = 1, z = 1) {
    if (tracing) {
      const p = currentPipeline.get(this);
      dispatches.push({ pass: passOf.get(this) ?? -1, shader: p ? pipelineShader.get(p) ?? -1 : -1, label: p?.label ?? '', x, y, z });
    }
    return dw.call(this, x, y, z);
  };
}

async function importBuild(build: string): Promise<Ort> {
  if (build === 'jsep') return (await import('onnxruntime-web')) as unknown as Ort;
  if (build === 'webgpu') return (await import('onnxruntime-web/webgpu')) as unknown as Ort;
  return (await import('onnxruntime-web/jspi')) as unknown as Ort;
}

async function main(): Promise<void> {
  try {
    wrap();
    const p = new URLSearchParams(location.search);
    const slug = p.get('model') ?? '';
    const L = Number(p.get('L') ?? '128');
    const graph = p.get('graph') ?? 'std';
    const build = p.get('build') ?? 'webgpu';
    const calls = Number(p.get('calls') ?? '10');
    const inputs = await loadInputs(slug, L);
    const ort = await importBuild(build);
    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
    const bytes = await (await fetch(`/models/k28/${slug}/k28.8/onnx/L${L}-${graph}-f16.onnx`)).arrayBuffer();
    const ep = { name: 'webgpu', ...(build !== 'jsep' ? { validationMode: 'wgpuOnly' } : {}) };
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: [ep as OrtNs.InferenceSession.ExecutionProviderConfig], graphOptimizationLevel: 'all',
    });
    const device = (await ort.env.webgpu.device) as GPUDevice;
    const mask = new Int32Array(L).fill(1);
    const run = async (i: number) => session.run({
      input_ids: new ort.Tensor('int32', inputs.row(i), [1, L]),
      attention_mask: new ort.Tensor('int32', mask, [1, L]),
    });
    result.runMs = [];
    for (let i = 0; i < calls; i += 1) {
      const t = performance.now();
      await run(i);
      result.runMs.push(performance.now() - t);
    }
    if (device.features.has('timestamp-query')) {
      query = {
        set: device.createQuerySet({ type: 'timestamp', count: QUERIES }),
        resolve: device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        read: device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        used: 0,
      };
    }
    result.stage = 'tracing';
    tracing = true;
    await run(calls);
    tracing = false;
    result.dispatches = dispatches;
    result.shaders = shaders;
    if (query && query.used) {
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(query.set, 0, query.used, query.resolve, 0);
      enc.copyBufferToBuffer(query.resolve, 0, query.read, 0, query.used * 8);
      device.queue.submit([enc.finish()]);
      await query.read.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(query.read.getMappedRange().slice(0));
      query.read.unmap();
      result.passTimesUs = [];
      for (let k = 0; k < query.used / 2; k += 1) {
        result.passTimesUs.push({ pass: k, us: Number(ts[2 * k + 1] - ts[2 * k]) / 1000,
          dispatches: dispatches.filter((d) => d.pass === k).length });
      }
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
