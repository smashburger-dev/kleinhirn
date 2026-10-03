/// <reference lib="esnext.float16" />
// K28 probe: GPU-side helpers for variant B. Caster batches f32 -> f16 cast
// dispatches (cast.wgsl); precomputeDeberta reproduces the exporter's
// LayerNorm + key_proj/query_proj precompute on the GPU with the engine's own
// layernorm and matmul kernels (f32).

import { KERNELS, wgsl } from '../../src/kernels/index.ts';
import castSrc from './cast.wgsl?raw';
import { padded, STORAGE_USAGE, type PreSources, type Spec, type StFile, f32View } from './st-loader.ts';

const ALIGN = 256; // minUniformBufferOffsetAlignment

interface CastJob { src: GPUBuffer; dst: GPUBuffer; n: number; outOffset: number }

export class Caster {
  private jobs: CastJob[] = [];
  private pipe: GPUComputePipeline;
  private layout: GPUBindGroupLayout;

  constructor(private device: GPUDevice) {
    const module = device.createShaderModule({ code: castSrc });
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } },
    ] });
    this.pipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: 'main' },
    });
  }

  queue(src: GPUBuffer, dst: GPUBuffer, n: number, outOffset = 0): void {
    this.jobs.push({ src, dst, n, outOffset });
  }

  // Encode every queued cast in one pass and submit. Returns after submit
  // (not after completion); callers await queue.onSubmittedWorkDone().
  flush(): void {
    const dev = this.device;
    if (!this.jobs.length) return;
    const params = new Uint32Array(this.jobs.length * (ALIGN / 4));
    const ubuf = dev.createBuffer({
      size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.jobs.forEach((j, i) => {
      params[i * (ALIGN / 4)] = j.n;
      params[i * (ALIGN / 4) + 1] = j.outOffset;
    });
    dev.queue.writeBuffer(ubuf, 0, params);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipe);
    this.jobs.forEach((j, i) => {
      const bg = dev.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: j.src } },
        { binding: 1, resource: { buffer: j.dst } },
        { binding: 2, resource: { buffer: ubuf, size: 16 } },
      ] });
      const groups = Math.ceil(j.n / 256);
      const x = Math.min(groups, 65535);
      pass.setBindGroup(0, bg, [i * ALIGN]);
      pass.dispatchWorkgroups(x, Math.ceil(groups / x));
    });
    pass.end();
    dev.queue.submit([enc.finish()]);
    this.jobs = [];
    this.uniforms.push(ubuf);
  }

  uniforms: GPUBuffer[] = [];
  destroy(): void { for (const u of this.uniforms) u.destroy(); this.uniforms = []; }
}

export interface RawCastResult {
  tensors: Map<string, GPUBuffer>;
  embeddings: Uint16Array;
  temps: GPUBuffer[];
  tempBytes: number;
  gpuBytes: number;
  castMs: number; // CPU cast of the embedding table only
  uploadMs: number; // createBuffer + writeBuffer calls
  gpuCastStart: number;
}

// Variant Bgpu: every GPU-resident tensor part goes up raw (f32) into a
// temporary STORAGE buffer; cast dispatches (queued, not yet submitted) write
// f16 into the final buffer. The embedding table is cast on the CPU.
export function uploadRaw(
  device: GPUDevice, st: StFile, specs: Spec[], caster: Caster,
): RawCastResult {
  const tensors = new Map<string, GPUBuffer>();
  const temps: GPUBuffer[] = [];
  let embeddings = new Uint16Array(0);
  let gpuBytes = 0;
  let tempBytes = 0;
  let castMs = 0;
  let uploadMs = 0;
  for (const s of specs) {
    if (s.kind === 'pre') continue;
    if (s.kind === 'cpu') {
      const t0 = performance.now();
      const arr = new Float16Array(f32View(st, s.parts[0].src));
      castMs += performance.now() - t0;
      embeddings = new Uint16Array(arr.buffer, 0, arr.length);
      continue;
    }
    const total = s.shape.reduce((a, b) => a * b, 1);
    const t0 = performance.now();
    const dst = device.createBuffer({ size: padded(total * 2), usage: STORAGE_USAGE });
    uploadMs += performance.now() - t0;
    gpuBytes += padded(total * 2);
    tensors.set(s.name, dst);
    for (const part of s.parts) {
      const view = f32View(st, part.src);
      const t1 = performance.now();
      const tmp = device.createBuffer({
        size: view.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(tmp, 0, view.buffer, view.byteOffset, view.byteLength);
      uploadMs += performance.now() - t1;
      temps.push(tmp);
      tempBytes += view.byteLength;
      caster.queue(tmp, dst, view.length, part.elemOffset);
    }
  }
  return { tensors, embeddings, temps, tempBytes, gpuBytes, castMs, uploadMs, gpuCastStart: 0 };
}

// DeBERTa share_att_key precompute on the GPU. Produces f16 buffers for
// rel_embeddings (post-LayerNorm) and layers.N.pos_key / pos_query and
// destroys its f32 temporaries after the cast has completed.
export async function precomputeDeberta(
  device: GPUDevice, pre: PreSources, eps: number,
): Promise<{ tensors: Map<string, GPUBuffer>; gpuBytes: number }> {
  const caster = new Caster(device); // own instance: flush must not touch the main batch
  const H = pre.lnW.length;
  const R = pre.rel.length / H;
  const temps: GPUBuffer[] = [];
  const f32buf = (data: Float32Array | null, size: number, extra = 0) => {
    const b = device.createBuffer({ size, usage: STORAGE_USAGE | extra });
    if (data) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    temps.push(b);
    return b;
  };
  const pipe = (src: string, constants: Record<string, number>) => device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: wgsl(src, false) }),
      entryPoint: 'main', constants,
    },
  });
  const bg = (p: GPUComputePipeline, bufs: GPUBuffer[]) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  const rel = f32buf(pre.rel, pre.rel.byteLength);
  const lnW = f32buf(pre.lnW, pre.lnW.byteLength);
  const lnB = f32buf(pre.lnB, pre.lnB.byteLength);
  const dummy = f32buf(null, 16);
  const relLn = f32buf(null, pre.rel.byteLength);
  const lnPipe = pipe(KERNELS.layernorm, { N: H, MODE: 0, EPS: eps });
  const mmPipe = pipe(KERNELS.matmul, { M: R, N: H, K: H, ACT: 0 });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(lnPipe);
  pass.setBindGroup(0, bg(lnPipe, [rel, dummy, lnW, lnB, dummy, relLn]));
  pass.dispatchWorkgroups(R);
  pass.setPipeline(mmPipe);
  const out = new Map<string, GPUBuffer>();
  const finals: { name: string; f32: GPUBuffer }[] = [];
  pre.layers.forEach((l, i) => {
    for (const [nm, w, b] of [['pos_key', l.kw, l.kb], ['pos_query', l.qw, l.qb]] as const) {
      const wb = f32buf(w, w.byteLength);
      const bb = f32buf(b, b.byteLength);
      const ob = f32buf(null, R * H * 4);
      pass.setBindGroup(0, bg(mmPipe, [relLn, wb, bb, ob]));
      pass.dispatchWorkgroups(Math.ceil(H / 16), Math.ceil(R / 16));
      finals.push({ name: `layers.${i}.${nm}`, f32: ob });
    }
  });
  pass.end();
  device.queue.submit([enc.finish()]);

  let gpuBytes = 0;
  const mkFinal = (name: string, src: GPUBuffer, n: number) => {
    const dst = device.createBuffer({ size: padded(n * 2), usage: STORAGE_USAGE });
    gpuBytes += padded(n * 2);
    out.set(name, dst);
    caster.queue(src, dst, n, 0);
  };
  mkFinal('rel_embeddings', relLn, R * H);
  for (const f of finals) mkFinal(f.name, f.f32, R * H);
  caster.flush();
  await device.queue.onSubmittedWorkDone();
  for (const t of temps) t.destroy();
  caster.destroy();
  return { tensors: out, gpuBytes };
}
