// Countdown set-judge as a WebGPU compute plan (K17). Mirrors the upstream
// interference_search/judge.py forward: Linear(nFeat -> d) on each number
// feature row, a learned class token prepended at row b*maxN, then
// `layers` pre-norm TransformerEncoderLayer blocks (packed in_proj
// q|k|v, 4 heads, out_proj, ReLU FFN d -> ffn -> d, biases everywhere,
// eps 1e-5) and out = LayerNorm + Linear(d -> 1) on the class row.
// Masked multi-head attention reuses mbattention.wgsl with WINDOW=0 — a
// standard full-attention kernel with key mask; no judge-specific kernel
// is needed. Batch (K16): B states packed as B*maxN rows, one padded
// 8-token sequence per state; logits decode to B raw values, sigmoid is
// applied on the CPU like upstream's score().

import { KERNELS, wgsl } from '../kernels/index.ts';

export interface SetJudgeSpec {
  layers: number;
  hiddenSize: number;
  heads: number;
  ffn: number;
  nFeat: number;
  maxN: number;
  normEps: number;
}

export class SetJudgePlan {
  private device: GPUDevice;
  private pipeLn!: GPUComputePipeline;
  private pipeMmInp!: GPUComputePipeline;
  private pipeMmQkv!: GPUComputePipeline;
  private pipeAttn!: GPUComputePipeline;
  private pipeMmAttn!: GPUComputePipeline;
  private pipeMmLin1!: GPUComputePipeline;
  private pipeMmLin2!: GPUComputePipeline;
  private pipeMmOut!: GPUComputePipeline;
  private pipeAdd0!: GPUComputePipeline;

  private feats!: GPUBuffer;
  private x!: GPUBuffer;
  private normed!: GPUBuffer;
  private qkv!: GPUBuffer;
  private ctx!: GPUBuffer;
  private attnOut!: GPUBuffer;
  private mid!: GPUBuffer;
  private ffnOut!: GPUBuffer;
  private clsRows!: GPUBuffer;
  private kNormed!: GPUBuffer;
  private raw!: GPUBuffer;
  private maskBuf!: GPUBuffer;
  private staging!: GPUBuffer;
  gpuBytes = 0;

  private bgInp!: GPUBindGroup;
  private layerBgs: GPUBindGroup[][] = [];
  private bgClsLn!: GPUBindGroup;
  private bgOut!: GPUBindGroup;

  constructor(
    device: GPUDevice,
    private spec: SetJudgeSpec,
    private tensors: Map<string, GPUBuffer>,
    private f16: boolean,
    public readonly batch = 1,
  ) {
    this.device = device;
    this.buildPipelines();
    this.buildBuffers();
    this.buildBindGroups();
  }

  private bytesOf(el: number): number {
    return el * (this.f16 ? 2 : 4);
  }

  private buf(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const b = this.device.createBuffer({ size, usage });
    this.gpuBytes += size;
    return b;
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
    const H = this.spec.hiddenSize;
    const S = this.spec.maxN * this.batch;
    const headDim = H / this.spec.heads;
    this.pipeLn = this.pipe(KERNELS.layernorm, {
      N: H, MODE: 0, EPS: this.spec.normEps });
    const mm = KERNELS.matmul;
    this.pipeMmInp = this.pipe(mm, { M: S, N: H, K: this.spec.nFeat, ACT: 0 });
    this.pipeMmQkv = this.pipe(mm, { M: S, N: 3 * H, K: H, ACT: 0 });
    this.pipeMmAttn = this.pipe(mm, { M: S, N: H, K: H, ACT: 0 });
    this.pipeMmLin1 = this.pipe(mm, { M: S, N: this.spec.ffn, K: H, ACT: 1 });
    this.pipeMmLin2 = this.pipe(mm, { M: S, N: H, K: this.spec.ffn, ACT: 0 });
    this.pipeMmOut = this.pipe(mm, { M: this.batch, N: 1, K: H, ACT: 0 });
    this.pipeAttn = this.pipe(KERNELS.mbattention, {
      L: this.spec.maxN, H: this.spec.heads, D: headDim,
      SCALE: headDim ** -0.5, WINDOW: 0 });
    this.pipeAdd0 = this.pipe(KERNELS.add, {
      TOTAL: S * H, N: H, MODE: 0, L: this.spec.maxN });
  }

  private buildBuffers(): void {
    const H = this.spec.hiddenSize;
    const S = this.spec.maxN * this.batch;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      | GPUBufferUsage.COPY_SRC;
    const rows = (n: number, cols: number) => this.buf(this.bytesOf(n * cols), st);
    this.feats = rows(S, this.spec.nFeat);
    this.x = rows(S, H);
    this.normed = rows(S, H);
    this.qkv = rows(S, 3 * H);
    this.ctx = rows(S, H);
    this.attnOut = rows(S, H);
    this.mid = rows(S, this.spec.ffn);
    this.ffnOut = rows(S, H);
    this.clsRows = this.buf(this.bytesOf(this.batch * H), st);
    this.kNormed = rows(this.batch, H);
    this.raw = this.buf(
      Math.max(4, this.bytesOf(this.batch)),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.maskBuf = this.buf(S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const mapBytes = Math.max(4, Math.ceil(this.bytesOf(this.batch) / 4) * 4);
    this.staging = this.device.createBuffer({
      size: mapBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.gpuBytes += mapBytes;
  }

  private buildBindGroups(): void {
    const dev = this.device;
    const t = this.tensors;
    const mk = (pipe: GPUComputePipeline) =>
      (bufs: GPUBuffer[]) => dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
      });
    const mkLn = mk(this.pipeLn);
    this.bgInp = mk(this.pipeMmInp)([this.feats, t.get('inp.weight')!,
      t.get('inp.bias')!, this.x]);
    for (let i = 0; i < this.spec.layers; i += 1) {
      const p = `layers.${i}`;
      this.layerBgs.push([
        mkLn([this.x, this.feats, t.get(`${p}.norm1.weight`)!,
          t.get(`${p}.norm1.bias`)!, this.maskBuf, this.normed]),
        mk(this.pipeMmQkv)([this.normed, t.get(`${p}.in_proj.weight`)!,
          t.get(`${p}.in_proj.bias`)!, this.qkv]),
        mk(this.pipeAttn)([this.qkv, this.maskBuf, this.ctx]),
        mk(this.pipeMmAttn)([this.ctx, t.get(`${p}.out_proj.weight`)!,
          t.get(`${p}.out_proj.bias`)!, this.attnOut]),
        mk(this.pipeAdd0)([this.x, this.attnOut]),
        mkLn([this.x, this.feats, t.get(`${p}.norm2.weight`)!,
          t.get(`${p}.norm2.bias`)!, this.maskBuf, this.normed]),
        mk(this.pipeMmLin1)([this.normed, t.get(`${p}.linear1.weight`)!,
          t.get(`${p}.linear1.bias`)!, this.mid]),
        mk(this.pipeMmLin2)([this.mid, t.get(`${p}.linear2.weight`)!,
          t.get(`${p}.linear2.bias`)!, this.ffnOut]),
        mk(this.pipeAdd0)([this.x, this.ffnOut]),
      ]);
    }
    this.bgClsLn = mkLn([this.clsRows, this.feats,
      t.get('out.norm.weight')!, t.get('out.norm.bias')!, this.maskBuf,
      this.kNormed]);
    this.bgOut = mk(this.pipeMmOut)([this.kNormed,
      t.get('out.fc.weight')!, t.get('out.fc.bias')!, this.raw]);
  }

  // feats holds batch * maxN rows of nFeat values; row b*maxN is the class
  // slot (content ignored), rows b*maxN+1..b*maxN+7 the number features.
  upload(
    feats: Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>,
    mask: Float32Array<ArrayBuffer>,
  ): void {
    this.device.queue.writeBuffer(this.feats, 0, feats);
    this.device.queue.writeBuffer(this.maskBuf, 0, mask);
  }

  encode(): GPUCommandBuffer {
    const H = this.spec.hiddenSize;
    const S = this.spec.maxN * this.batch;
    const elt = this.f16 ? 2 : 4;
    const R16 = Math.ceil(S / 16);
    const enc = this.device.createCommandEncoder();
    const layerOps = () =>
      ([
        [this.pipeLn, 0, S, 1],
        [this.pipeMmQkv, 1, Math.ceil(3 * H / 16), R16],
        [this.pipeAttn, 2, this.spec.heads, S],
        [this.pipeMmAttn, 3, Math.ceil(H / 16), R16],
        [this.pipeAdd0, 4, Math.ceil(S * H / 64), 1],
        [this.pipeLn, 5, S, 1],
        [this.pipeMmLin1, 6, Math.ceil(this.spec.ffn / 16), R16],
        [this.pipeMmLin2, 7, Math.ceil(H / 16), R16],
        [this.pipeAdd0, 8, Math.ceil(S * H / 64), 1],
      ] as [GPUComputePipeline, number, number, number][]);

    const p1 = enc.beginComputePass();
    p1.setPipeline(this.pipeMmInp);
    p1.setBindGroup(0, this.bgInp);
    p1.dispatchWorkgroups(Math.ceil(H / 16), R16);
    p1.end();
    // The class token is a parameter, not a projection: write it into the
    // slot row of every sequence after the input matmul.
    for (let b = 0; b < this.batch; b += 1) {
      enc.copyBufferToBuffer(
        this.tensors.get('cls')!, 0, this.x, b * this.spec.maxN * H * elt,
        H * elt);
    }
    const p2 = enc.beginComputePass();
    for (const bgs of this.layerBgs) {
      for (const [pipe, bgi, dx, dy] of layerOps()) {
        p2.setPipeline(pipe);
        p2.setBindGroup(0, bgs[bgi]);
        p2.dispatchWorkgroups(dx, dy);
      }
    }
    p2.end();
    for (let b = 0; b < this.batch; b += 1) {
      enc.copyBufferToBuffer(
        this.x, b * this.spec.maxN * H * elt, this.clsRows, b * H * elt,
        H * elt);
    }
    const p3 = enc.beginComputePass();
    p3.setPipeline(this.pipeLn);
    p3.setBindGroup(0, this.bgClsLn);
    p3.dispatchWorkgroups(this.batch, 1);
    p3.setPipeline(this.pipeMmOut);
    p3.setBindGroup(0, this.bgOut);
    p3.dispatchWorkgroups(1, Math.ceil(this.batch / 16));
    p3.end();
    enc.copyBufferToBuffer(this.raw, 0, this.staging, 0,
      Math.ceil(this.bytesOf(this.batch) / 4) * 4);
    return enc.finish();
  }

  submit(): void {
    this.device.queue.submit([this.encode()]);
  }

  // Raw logits in manifest dtype (f16 callers decode Uint16).
  async readLogits(): Promise<Float32Array | Uint16Array> {
    await this.staging.mapAsync(GPUMapMode.READ);
    const bytes = this.staging.getMappedRange().slice(0);
    this.staging.unmap();
    return this.f16 ? new Uint16Array(bytes) : new Float32Array(bytes);
  }

  // Debug readback: copies an intermediate buffer after the last submit
  // and returns it as raw bytes (decode by caller).
  async debugRead(name: 'feats' | 'x' | 'normed' | 'qkv' | 'ctx' | 'attnOut'
    | 'mid' | 'ffnOut' | 'clsRows' | 'kNormed' | 'raw' | 'maskBuf',
  ): Promise<ArrayBuffer> {
    const src = this[name];
    const tmp = this.device.createBuffer({
      size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, tmp, 0, src.size);
    this.device.queue.submit([enc.finish()]);
    await tmp.mapAsync(GPUMapMode.READ);
    const out = tmp.getMappedRange().slice(0);
    tmp.unmap();
    tmp.destroy();
    return out;
  }
}
