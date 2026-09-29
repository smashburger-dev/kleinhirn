// DeBERTa-v2 encoder + GLiNER2 classification head as a WebGPU compute plan.
// Per bucket (L, K=16): all buffers, pipelines and bind groups are created at
// load; a call uploads embeddings/mask/markers, encodes one command buffer,
// submits, and reads back K logits. Exact one-call-at-a-time discipline lives
// in index.ts.
// Batch (K16): with batch > 1 every row buffer holds B sequences packed
// contiguously (global row b*L + i), the marker buffers hold B blocks of
// 3*K, and the readback carries B*K logits. Position-aware kernels
// (attention, gather, masklogits) resolve the sequence from the row index;
// row-wise kernels need no change. Capture stays a B=1 tool.

import { KERNELS, wgsl } from '../kernels/index.ts';

export interface EncoderSpec {
  hiddenSize: number;
  layers: number;
  heads: number;
  intermediateSize: number;
  positionBuckets: number;
  maxRelativePositions: number;
  layerNormEps: number;
}

export interface RunInput {
  embeddings: Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>;
  mask: Float32Array<ArrayBuffer>;        // L
  packedMarkers: Uint32Array<ArrayBuffer>; // 3*K: idx, mask f32 bits, groups
}

// Marker budget and head hidden width are per-plan: banking77 needs up to
// 72 label markers in one schema, the base/multi head is 2 * hiddenSize.

function relBucketIndex(i: number, j: number, span: number, maxPos: number): number {
  const rel = i - j;
  const mid = span >> 1;
  const abs = Math.abs(rel);
  let r = rel;
  if (abs > mid) {
    const logPos = Math.ceil(
      Math.log(abs / mid) / Math.log((maxPos - 1) / mid) * (mid - 1)) + mid;
    r = logPos * Math.sign(rel);
  }
  return Math.min(Math.max(r + span, 0), 2 * span - 1);
}

export function relPosTable(
  length: number, span: number, maxPos: number,
): Uint32Array<ArrayBuffer> {
  const t = new Uint32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      t[i * length + j] = relBucketIndex(i, j, span, maxPos);
    }
  }
  return t;
}

export class EncoderPlan {
  private device: GPUDevice;
  private f16: boolean;
  private pipeLn: GPUComputePipeline[] = []; // modes 0,1,2
  private pipeMmQkv!: GPUComputePipeline;
  private pipeMmAttn!: GPUComputePipeline;
  private pipeMmFfn1!: GPUComputePipeline;
  private pipeMmFfn2!: GPUComputePipeline;
  private pipeMmFc1!: GPUComputePipeline;
  private pipeMmFc2!: GPUComputePipeline;
  private pipeAttn!: GPUComputePipeline;
  private pipeGather!: GPUComputePipeline;
  private pipeMaskL!: GPUComputePipeline;

  private emb!: GPUBuffer;
  private x!: GPUBuffer;
  private tmp!: GPUBuffer;
  private qkv!: GPUBuffer;
  private ctx!: GPUBuffer;
  private attnOut!: GPUBuffer;
  private mid!: GPUBuffer;
  private ffnOut!: GPUBuffer;
  private maskBuf!: GPUBuffer;
  private packedBuf!: GPUBuffer;
  private states!: GPUBuffer;
  private h1!: GPUBuffer;
  private raw!: GPUBuffer;
  private logitsBuf!: GPUBuffer;
  private staging!: GPUBuffer;
  // Lazy: capture is a B=1 parity tool, so batch plans skip this large
  // buffer ((layers + 2) * length * hidden * 4) entirely.
  private capBuf?: GPUBuffer;
  private ownedBufs: GPUBuffer[] = [];
  gpuBytes = 0;

  private bgEmb!: GPUBindGroup;
  private bgEmbPlain!: GPUBindGroup;
  private layerBgs: GPUBindGroup[][] = [];
  private bgGather!: GPUBindGroup;
  private bgFc1!: GPUBindGroup;
  private bgFc2!: GPUBindGroup;
  private bgMaskL!: GPUBindGroup;

  constructor(
    device: GPUDevice,
    private spec: EncoderSpec,
    private tensors: Map<string, GPUBuffer>,
    private embeddingsLN: { weight: GPUBuffer; bias: GPUBuffer },
    private head: { fc1w: GPUBuffer; fc1b: GPUBuffer; fc2w: GPUBuffer; fc2b: GPUBuffer },
    private headTemperature: number,
    public readonly length: number,
    f16: boolean,
    public readonly markers = 16,
    private headHidden = 768,
    public readonly batch = 1,
  ) {
    this.device = device;
    this.f16 = f16;
    this.buildPipelines();
    this.buildBuffers();
    this.buildBindGroups();
  }

  private bytesOf(el: number): number {
    return el * (this.f16 ? 2 : 4);
  }

  private buf(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const b = this.device.createBuffer({ size, usage });
    this.ownedBufs.push(b);
    this.gpuBytes += size;
    return b;
  }

  private captureBuffer(): GPUBuffer {
    if (!this.capBuf) {
      this.capBuf = this.device.createBuffer({
        size: (this.spec.layers + 2) * this.length
          * this.spec.hiddenSize * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.ownedBufs.push(this.capBuf);
      this.gpuBytes += this.capBuf.size;
    }
    return this.capBuf;
  }

  destroy(): void {
    for (const b of this.ownedBufs) b.destroy();
    this.ownedBufs.length = 0;
  }

  private pipe(src: string, constants: Record<string, number>): GPUComputePipeline {
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: wgsl(src, this.f16) }),
        entryPoint: 'main',
        constants,
      },
    });
  }

  private buildPipelines(): void {
    const L = this.length;
    const M = L * this.batch;
    const K = this.markers * this.batch;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediateSize;
    const eps = this.spec.layerNormEps;
    this.pipeLn = [0, 1, 2].map((mode) => this.pipe(
      KERNELS.layernorm, { N: H, MODE: mode, EPS: eps }));
    const mm = KERNELS.matmul;
    this.pipeMmQkv = this.pipe(mm, { M, N: 3 * H, K: H, ACT: 0 });
    this.pipeMmAttn = this.pipe(mm, { M, N: H, K: H, ACT: 0 });
    this.pipeMmFfn1 = this.pipe(mm, { M, N: I, K: H, ACT: 2 });
    this.pipeMmFfn2 = this.pipe(mm, { M, N: H, K: I, ACT: 0 });
    this.pipeMmFc1 = this.pipe(mm, { M: K, N: this.headHidden, K: H, ACT: 1 });
    this.pipeMmFc2 = this.pipe(mm, { M: K, N: 1, K: this.headHidden, ACT: 0 });
    this.pipeAttn = this.pipe(KERNELS.attention, {
      L, H: this.spec.heads, D: H / this.spec.heads,
      SCALE: Math.sqrt(3 * (H / this.spec.heads)),
    });
    this.pipeGather = this.pipe(KERNELS.gather, { K: this.markers, D: H, L });
    this.pipeMaskL = this.pipe(KERNELS.masklogits, {
      K: this.markers, TEMP: this.headTemperature });
  }

  private buildBuffers(): void {
    const L = this.length * this.batch;
    const K = this.markers * this.batch;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediateSize;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const rows = (n: number, cols: number) => this.buf(this.bytesOf(n * cols), st);
    const src = st | GPUBufferUsage.COPY_SRC; // capture path reads x and tmp
    this.emb = rows(L, H);
    this.x = this.buf(this.bytesOf(L * H), src);
    this.tmp = this.buf(this.bytesOf(L * H), src);
    this.qkv = rows(L, 3 * H);
    this.ctx = rows(L, H);
    this.attnOut = rows(L, H);
    this.mid = rows(L, I);
    this.ffnOut = rows(L, H);
    this.states = rows(K, H);
    this.h1 = rows(K, this.headHidden);
    this.raw = rows(K, 1);
    this.maskBuf = this.buf(L * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.packedBuf = this.buf(3 * K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    // LN modes 0/1 never read b, but a writable out may not alias any other
    // binding in the same dispatch, so a dedicated dummy is required.
    this.dummyBuf = this.buf(4, GPUBufferUsage.STORAGE);
    this.logitsBuf = this.buf(
      K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.staging = this.device.createBuffer({
      size: K * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.ownedBufs.push(this.staging);
    this.gpuBytes += K * 4;
    const rel = relPosTable(
      this.length, this.spec.positionBuckets, this.spec.maxRelativePositions);
    this.relidxBuf = this.buf(rel.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this.relidxBuf, 0, rel);
    // Layer capture is a B=1 parity tool; allocated lazily by
    // captureBuffer() so batch plans skip it.
  }

  private relidxBuf!: GPUBuffer;
  private dummyBuf!: GPUBuffer;

  private buildBindGroups(): void {
    const dev = this.device;
    const t = this.tensors;
    const L = this.spec.layers;
    const mkBg = (layout: GPUBindGroupLayout, bufs: GPUBuffer[]) => dev.createBindGroup({
      layout,
      entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const mk = (pipe: GPUComputePipeline) =>
      (bufs: GPUBuffer[]) => mkBg(pipe.getBindGroupLayout(0), bufs);

    const bEmb = mk(this.pipeLn[1]);
    this.bgEmb = bEmb([this.emb, this.dummyBuf, this.embeddingsLN.weight,
      this.embeddingsLN.bias, this.maskBuf, this.x]);
    const bEmb0 = mk(this.pipeLn[0]);
    this.bgEmbPlain = bEmb0([this.emb, this.dummyBuf, this.embeddingsLN.weight,
      this.embeddingsLN.bias, this.maskBuf, this.tmp]);

    const qkvBg = mk(this.pipeMmQkv);
    const attnBg = mk(this.pipeAttn);
    const attnOutBg = mk(this.pipeMmAttn);
    const lnABg = mk(this.pipeLn[2]);
    const ffn1Bg = mk(this.pipeMmFfn1);
    const ffn2Bg = mk(this.pipeMmFfn2);
    const lnFBg = mk(this.pipeLn[2]);
    for (let l = 0; l < L; l += 1) {
      this.layerBgs.push([
        qkvBg([this.x, t.get(`layers.${l}.qkv.weight`)!,
          t.get(`layers.${l}.qkv.bias`)!, this.qkv]),
        attnBg([this.qkv, t.get(`layers.${l}.pos_key`)!,
          t.get(`layers.${l}.pos_query`)!, this.relidxBuf, this.maskBuf, this.ctx]),
        attnOutBg([this.ctx, t.get(`layers.${l}.attn_out.weight`)!,
          t.get(`layers.${l}.attn_out.bias`)!, this.attnOut]),
        lnABg([this.x, this.attnOut, t.get(`layers.${l}.attn_ln.weight`)!,
          t.get(`layers.${l}.attn_ln.bias`)!, this.maskBuf, this.tmp]),
        ffn1Bg([this.tmp, t.get(`layers.${l}.ffn_in.weight`)!,
          t.get(`layers.${l}.ffn_in.bias`)!, this.mid]),
        ffn2Bg([this.mid, t.get(`layers.${l}.ffn_out.weight`)!,
          t.get(`layers.${l}.ffn_out.bias`)!, this.ffnOut]),
        lnFBg([this.tmp, this.ffnOut, t.get(`layers.${l}.ffn_ln.weight`)!,
          t.get(`layers.${l}.ffn_ln.bias`)!, this.maskBuf, this.x]),
      ]);
    }

    this.bgGather = mk(this.pipeGather)([this.packedBuf, this.x, this.states]);
    this.bgFc1 = mk(this.pipeMmFc1)([this.states, this.head.fc1w, this.head.fc1b, this.h1]);
    this.bgFc2 = mk(this.pipeMmFc2)([this.h1, this.head.fc2w, this.head.fc2b, this.raw]);
    this.bgMaskL = mk(this.pipeMaskL)([this.raw, this.packedBuf, this.logitsBuf]);
  }

  // Encode the full pass into one command buffer. Rows at index >= seqLen are
  // masked everywhere downstream, so row-dispatch dimensions use seqLen
  // (capture keeps the full bucket for the parity tooling). capture: also copy
  // the pre-mask LN embeddings, masked embeddings and every layer output into
  // capBuf (f32 bytes, capture path is f32-only parity tooling). skip is a
  // debug knob for per-kernel cost attribution (indices into the per-layer
  // dispatch list). ts: optional timestamp-query resources; when set, each
  // compute pass writes begin/end GPU timestamps that are resolved into
  // ts.staging for the caller to map.
  encode(
    capture: boolean, skip?: Set<number>, seqLen = this.length,
    ts?: { querySet: GPUQuerySet; resolve: GPUBuffer; staging: GPUBuffer },
  ): GPUCommandBuffer {
    const enc = this.device.createCommandEncoder();
    let passIdx = 0;
    const beginPass = (): GPUComputePassEncoder => {
      const writes = ts ? {
        querySet: ts.querySet,
        beginningOfPassWriteIndex: passIdx * 2,
        endOfPassWriteIndex: passIdx * 2 + 1,
      } : undefined;
      passIdx += 1;
      return enc.beginComputePass(
        writes ? { timestampWrites: writes } : undefined);
    };
    const rows = capture ? this.length : seqLen;
    const capOff = (this.length * this.spec.hiddenSize * 4);
    const H16 = Math.ceil(this.spec.hiddenSize / 16);
    const R16 = Math.ceil(rows / 16);
    const I16 = Math.ceil(this.spec.intermediateSize / 16);
    const seq: [GPUComputePipeline, number, number, number][] = [
      [this.pipeMmQkv, 0, Math.ceil(3 * this.spec.hiddenSize / 16), R16],
      [this.pipeAttn, 1, this.spec.heads, rows],
      [this.pipeMmAttn, 2, H16, R16],
      [this.pipeLn[2], 3, rows, 1],
      [this.pipeMmFfn1, 4, I16, R16],
      [this.pipeMmFfn2, 5, H16, R16],
      [this.pipeLn[2], 6, rows, 1],
    ];
    // Non-capture forward: one compute pass for the whole graph. Dispatches
    // inside a pass are still program-ordered, so the seven pass boundaries
    // between embedding, layers and head are pure overhead. Capture and
    // timestamp profiling keep the multi-pass shape: buffer copies cannot
    // live inside a compute pass.
    if (!capture && !ts) {
      const pass = beginPass();
      pass.setPipeline(this.pipeLn[1]);
      pass.setBindGroup(0, this.bgEmb);
      pass.dispatchWorkgroups(rows);
      for (let l = 0; l < this.spec.layers; l += 1) {
        const bgs = this.layerBgs[l];
        for (const [pipe, bgi, dx, dy] of seq) {
          if (skip?.has(bgi)) continue;
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bgs[bgi]);
          pass.dispatchWorkgroups(dx, dy);
        }
      }
      pass.setPipeline(this.pipeGather);
      pass.setBindGroup(0, this.bgGather);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pipeMmFc1);
      pass.setBindGroup(0, this.bgFc1);
      pass.dispatchWorkgroups(
        Math.ceil(this.headHidden / 16), Math.ceil(this.markers * this.batch / 16));
      pass.setPipeline(this.pipeMmFc2);
      pass.setBindGroup(0, this.bgFc2);
      pass.dispatchWorkgroups(1, Math.ceil(this.markers * this.batch / 16));
      pass.setPipeline(this.pipeMaskL);
      pass.setBindGroup(0, this.bgMaskL);
      pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(
        this.logitsBuf, 0, this.staging, 0, this.markers * this.batch * 4);
      return enc.finish();
    }

    const pass = beginPass();
    if (capture) {
      pass.setPipeline(this.pipeLn[0]);
      pass.setBindGroup(0, this.bgEmbPlain);
      pass.dispatchWorkgroups(rows);
      pass.setPipeline(this.pipeLn[1]);
    } else {
      pass.setPipeline(this.pipeLn[1]);
    }
    pass.setBindGroup(0, this.bgEmb);
    pass.dispatchWorkgroups(rows);
    pass.end();

    if (capture) {
      enc.copyBufferToBuffer(this.tmp, 0, this.captureBuffer(), 0, capOff);
      enc.copyBufferToBuffer(this.x, 0, this.captureBuffer(), capOff, capOff);
    }

    for (let l = 0; l < this.spec.layers; l += 1) {
      const p2 = beginPass();
      const bgs = this.layerBgs[l];
      for (const [pipe, bgi, dx, dy] of seq) {
        if (skip?.has(bgi)) continue;
        p2.setPipeline(pipe);
        p2.setBindGroup(0, bgs[bgi]);
        p2.dispatchWorkgroups(dx, dy);
      }
      p2.end();
      if (capture) {
        enc.copyBufferToBuffer(
          this.x, 0, this.captureBuffer(), (l + 2) * capOff, capOff);
      }
    }

    const p3 = beginPass();
    p3.setPipeline(this.pipeGather);
    p3.setBindGroup(0, this.bgGather);
    p3.dispatchWorkgroups(1);
    p3.setPipeline(this.pipeMmFc1);
    p3.setBindGroup(0, this.bgFc1);
    p3.dispatchWorkgroups(
      Math.ceil(this.headHidden / 16), Math.ceil(this.markers * this.batch / 16));
    p3.setPipeline(this.pipeMmFc2);
    p3.setBindGroup(0, this.bgFc2);
    p3.dispatchWorkgroups(1, Math.ceil(this.markers * this.batch / 16));
    p3.setPipeline(this.pipeMaskL);
    p3.setBindGroup(0, this.bgMaskL);
    p3.dispatchWorkgroups(1);
    p3.end();
    enc.copyBufferToBuffer(
      this.logitsBuf, 0, this.staging, 0, this.markers * this.batch * 4);
    if (ts) {
      const bytes = passIdx * 16;
      enc.resolveQuerySet(ts.querySet, 0, passIdx * 2, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.staging, 0, bytes);
    }
    return enc.finish();
  }

  submit(capture: boolean, skip?: Set<number>, seqLen = this.length): void {
    this.device.queue.submit([this.encode(capture, skip, seqLen)]);
  }

  // GPU timestamp profiling (K5): one timed forward, per-pass milliseconds
  // keyed 'embed' | 'layer0..N-1' | 'head'. Null when the device lacks the
  // 'timestamp-query' feature; the caller then uses CPU-side timing.
  async kernelTimesMs(input: RunInput): Promise<Record<string, number> | null> {
    if (!this.device.features.has('timestamp-query')) return null;
    const passCount = this.spec.layers + 2;
    const bytes = passCount * 16;
    const ts = {
      querySet: this.device.createQuerySet({
        type: 'timestamp', count: passCount * 2 }),
      resolve: this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      staging: this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    this.upload(input);
    this.device.queue.submit([this.encode(false, undefined, this.length, ts)]);
    await ts.staging.mapAsync(GPUMapMode.READ);
    const ns = new BigUint64Array(ts.staging.getMappedRange().slice(0));
    ts.staging.unmap();
    const names = [
      'embed',
      ...Array.from({ length: this.spec.layers }, (_, l) => `layer${l}`),
      'head',
    ];
    const out: Record<string, number> = {};
    for (let i = 0; i < passCount; i += 1) {
      out[names[i]] = Number(ns[i * 2 + 1] - ns[i * 2]) / 1e6;
    }
    ts.querySet.destroy();
    ts.resolve.destroy();
    ts.staging.destroy();
    return out;
  }

  upload(input: RunInput): void {
    this.device.queue.writeBuffer(this.emb, 0, input.embeddings);
    this.device.queue.writeBuffer(this.maskBuf, 0, input.mask);
    this.device.queue.writeBuffer(this.packedBuf, 0, input.packedMarkers);
  }

  async readLogits(): Promise<Float32Array> {
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    return out;
  }

  async readCapture(): Promise<Float32Array> {
    const cap = this.captureBuffer();
    await cap.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(cap.getMappedRange().slice(0));
    cap.unmap();
    return out;
  }
}
