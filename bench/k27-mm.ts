/// <reference types="vite/client" />
// K27 matmul micro-benchmark: kernel variants on the encoder matmul shapes, f16 storage.
// Every variant computes C[M,N] = A[M,K] * W[N,K]^T + B[N] with the binding layout of
// src/kernels/matmul.wgsl (a, w, bias, c); its dispatch comes from the `// dispatch:`
// header line (tile rows and cols). GPU time per dispatch by timestamp queries
// (median of 50 after 30 warmup), output compared with the 16x16 kernel (bit-equal
// count and largest deviation). Minimum limits on the device.
// Query: ?variants=matmul,mmtile,...&shapes=128x2304x768,...  (M x N x K)

import { KERNELS, wgsl } from '../src/kernels/index.ts';
import { median } from './metrics.ts';

const variantSrc = import.meta.glob('./k27/variants/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

interface Row { variant: string; shape: string; medianUs: number; bitEqual: number; maxAbsDiff: number; error?: string }
interface Result { stage: string; rows?: Row[]; error?: string; done?: boolean; limits?: Record<string, number> }
declare global {
  interface Window { khK27Mm?: Result }
}

function source(name: string): string {
  if (name in KERNELS) return KERNELS[name as keyof typeof KERNELS];
  const s = variantSrc[`./k27/variants/${name}.wgsl`];
  if (!s) throw new Error(`variant ${name} missing`);
  return s;
}

function tileOf(src: string): [number, number] {
  const m = src.match(/\/\/ dispatch: (\d+)x(\d+)/);
  return m ? [Number(m[1]), Number(m[2])] : [16, 16];
}

function f16bits(x: number): number {
  const f = new Float32Array([x]); const u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000; let e = ((u >>> 23) & 0xff) - 112; let m = u & 0x7fffff;
  if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
  m += 0x1000; if (m & 0x800000) { m = 0; e += 1; }
  return s | (e << 10) | (m >>> 13);
}

async function main(): Promise<void> {
  const result: Result = { stage: 'boot', rows: [] };
  window.khK27Mm = result;
  try {
    const p = new URLSearchParams(location.search);
    const variants = (p.get('variants') ?? 'matmul,mmtile').split(',');
    const shapes = (p.get('shapes') ?? '128x2304x768,128x768x768,128x3072x768,128x768x3072').split(',')
      .map((s) => s.split('x').map(Number));
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'timestamp-query'] });
    const lim: Record<string, number> = {};
    for (const k in device.limits) lim[k] = (device.limits as unknown as Record<string, number>)[k];
    result.limits = lim;
    let seed = 1;
    const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return (seed / 4294967296) * 2 - 1; };
    for (const [M, N, K] of shapes) {
      const shape = `${M}x${N}x${K}`;
      const mk = (n: number, scale: number) => {
        const h = new Uint16Array(n + (n % 2));
        for (let i = 0; i < n; i += 1) h[i] = f16bits(rnd() * scale);
        const b = device.createBuffer({ size: Math.ceil(h.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(b, 0, h);
        return b;
      };
      const A = mk(M * K, 1); const W = mk(N * K, 0.05); const B = mk(N, 0.1);
      const outBytes = Math.ceil((M * N * 2) / 16) * 16;
      const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
      const qbuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const qread = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      let reference: Uint16Array | null = null;
      for (const v of variants) {
        const row: Row = { variant: v, shape, medianUs: NaN, bitEqual: 0, maxAbsDiff: NaN };
        try {
          const [vname, splitText] = v.split('@');
          const split = Number(splitText ?? '1');
          const src = source(vname);
          if (/needs K % (\d+)/i.test(src) && K % (Number(src.match(/needs K % (\d+)/i)?.[1]) * split) !== 0) throw new Error('K does not fit');
          const [tm, tn] = tileOf(src);
          device.pushErrorScope('validation');
          const pipe = device.createComputePipeline({ layout: 'auto', compute: {
            module: device.createShaderModule({ code: wgsl(src, true) }), entryPoint: 'main',
            constants: { M, N, K, ACT: split > 1 ? 0 : 2, ...(split > 1 ? { SPLIT: split } : {}) } } });
          const C = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          // Split-K (variant@S): partial sums into P, then mmreduce adds them with bias and GELU into C.
          const P = split > 1 ? device.createBuffer({ size: split * M * N * 4, usage: GPUBufferUsage.STORAGE }) : null;
          const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: A } }, { binding: 1, resource: { buffer: W } },
            ...(P ? [] : [{ binding: 2, resource: { buffer: B } }]), { binding: 3, resource: { buffer: P ?? C } }] });
          const red = P ? device.createComputePipeline({ layout: 'auto', compute: {
            module: device.createShaderModule({ code: wgsl(source('mmreduce'), true) }), entryPoint: 'main',
            constants: { M, N, ACT: 2, SPLIT: split } } }) : null;
          const redBg = red && P ? device.createBindGroup({ layout: red.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: P } }, { binding: 1, resource: { buffer: B } },
            { binding: 2, resource: { buffer: C } }] }) : null;
          const err = await device.popErrorScope();
          if (err) throw new Error(err.message);
          const times: number[] = [];
          for (let it = 0; it < 80; it += 1) {
            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
            pass.setPipeline(pipe); pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(N / tn) * split, Math.ceil(M / tm));
            if (red && redBg) {
              pass.setPipeline(red); pass.setBindGroup(0, redBg);
              pass.dispatchWorkgroups(Math.ceil((M * N) / 256));
            }
            pass.end();
            enc.resolveQuerySet(qs, 0, 2, qbuf, 0);
            enc.copyBufferToBuffer(qbuf, 0, qread, 0, 16);
            device.queue.submit([enc.finish()]);
            await qread.mapAsync(GPUMapMode.READ);
            const t = new BigUint64Array(qread.getMappedRange().slice(0));
            qread.unmap();
            if (it >= 30) times.push(Number(t[1] - t[0]) / 1000);
          }
          row.medianUs = median([...times].sort((x, y) => x - y));
          const read = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(C, 0, read, 0, outBytes);
          device.queue.submit([enc.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const out = new Uint16Array(read.getMappedRange().slice(0, M * N * 2));
          read.unmap();
          if (!reference) reference = out;
          const half = (h: number) => { const s = h & 0x8000 ? -1 : 1; const e = (h >> 10) & 31; const m = h & 1023;
            return e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15); };
          let eq = 0; let md = 0;
          for (let i = 0; i < out.length; i += 1) {
            if (out[i] === reference[i]) eq += 1;
            md = Math.max(md, Math.abs(half(out[i]) - half(reference[i])));
          }
          row.bitEqual = eq / out.length;
          row.maxAbsDiff = md;
          C.destroy(); read.destroy();
        } catch (e) {
          row.error = String(e).slice(0, 300);
        }
        result.rows?.push(row);
        result.stage = `${shape} ${v}`;
      }
      A.destroy(); W.destroy(); B.destroy();
    }
    device.destroy();
    result.stage = 'done';
    result.done = true;
  } catch (e) {
    result.stage = 'error';
    result.error = String(e);
    result.done = true;
  }
}

void main();
