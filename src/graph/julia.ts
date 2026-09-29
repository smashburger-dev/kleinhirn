// ModernBERT encoder + Julia decision head as a WebGPU compute plan (K8).
// Per bucket (L): all buffers, pipelines and bind groups are created at load;
// a call uploads CPU-gathered embeddings/mask/markers, encodes one command
// buffer, submits, and reads back K option logits.
// Batch (K16): with batch > 1 row buffers hold B sequences packed as B*L
// global rows, markers B blocks of 3*K, the type row B rows selected by
// per-sequence qtype, and the readback B*K logits. Position-aware kernels
// (mbattention, rope, gather, masklogits, add MODE 1) resolve the sequence
// from the row index. Capture stays a B=1 tool. The layout mirrors
// convert/julia_manifest_forward.py: embedding LayerNorm (no bias), 22
// pre-norm layers with Identity attn_norm at layer 0, fused Wqkv without
// bias, RoPE on q/k, sliding-window attention |i-j| <= 64 except every third
// layer (index % 3 == 0 is global), GeGLU FFN, final_norm, then +type_emb,
// two pre-norm TransformerEncoderLayer blocks with biases, marker gather,
// scorer LayerNorm -> Linear -> exact-erf GELU -> Linear(384->1), -1e4 fill.

import { KERNELS, wgsl } from '../kernels/index.ts';
import type { DispatchTimestamps, ProfileGranularity } from './deberta.ts';

export interface JuliaSpec {
  layers: number;
  hiddenSize: number;
  heads: number;
  intermediate: number;
  normEps: number;
  ropeTheta: number;
  localAttention: number;
  globalEvery: number;
  headLayers: number;
  headFfn: number;
  options: number;
}

export interface JuliaRunInput {
  embeddings: Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>;
  mask: Float32Array<ArrayBuffer>;         // L
  packedMarkers: Uint32Array<ArrayBuffer>; // 3*K: idx, mask f32 bits, groups
  qtype: number;
}

function ropeTable(
  length: number, theta: number, headDim: number,
): Float32Array<ArrayBuffer> {
  const half = headDim / 2;
  const t = new Float32Array(length * 2 * headDim);
  for (let i = 0; i < length; i += 1) {
    for (let d = 0; d < half; d += 1) {
      const f = i * theta ** (-(2 * d) / headDim);
      t[i * 2 * headDim + d] = Math.cos(f);
      t[i * 2 * headDim + headDim + d] = Math.sin(f);
    }
  }
  return t;
}

export class JuliaPlan {
  private device: GPUDevice;
  private f16: boolean;
  private pipeLn!: GPUComputePipeline;
  private pipeMmQkv!: GPUComputePipeline;
  private pipeMmAttn!: GPUComputePipeline;
  private pipeRope!: GPUComputePipeline;
  private pipeAttnG!: GPUComputePipeline;
  private pipeAttnW!: GPUComputePipeline;
  private pipeAdd0!: GPUComputePipeline;
  private pipeAdd1!: GPUComputePipeline;
  private pipeMmIn!: GPUComputePipeline;
  private pipeGeglu!: GPUComputePipeline;
  private pipeMmFfn!: GPUComputePipeline;
  private pipeMmLin1!: GPUComputePipeline;
  private pipeMmLin2!: GPUComputePipeline;
  private pipeGather!: GPUComputePipeline;
  private pipeMmFc1!: GPUComputePipeline;
  private pipeMmFc2!: GPUComputePipeline;
  private pipeMaskL!: GPUComputePipeline;

  private emb!: GPUBuffer;
  private x!: GPUBuffer;
  private normed!: GPUBuffer;
  private qkv!: GPUBuffer;
  private ctx!: GPUBuffer;
  private attnOut!: GPUBuffer;
  private mid!: GPUBuffer;
  private gate!: GPUBuffer;
  private ffnOut!: GPUBuffer;
  private tmp!: GPUBuffer;
  private maskBuf!: GPUBuffer;
  private packedBuf!: GPUBuffer;
  private cossinBuf!: GPUBuffer;
  private typeRow!: GPUBuffer;
  private zeroBuf!: GPUBuffer;
  private dummyBuf!: GPUBuffer;
  private states!: GPUBuffer;
  private kNormed!: GPUBuffer;
  private h1!: GPUBuffer;
  private raw!: GPUBuffer;
  private logitsBuf!: GPUBuffer;
  private staging!: GPUBuffer;
  // Lazy: capture is a B=1 parity tool, so batch plans skip this large
  // buffer ((layers + 5) * length * hidden * 4) entirely.
  private capBuf?: GPUBuffer;
  private ownedBufs: GPUBuffer[] = [];
  gpuBytes = 0;

  private bgEmb!: GPUBindGroup;
  private layerBgs: GPUBindGroup[][] = [];
  private bgFinal!: GPUBindGroup;
  private bgType!: GPUBindGroup;
  private headBgs: GPUBindGroup[][] = [];
  private bgGather!: GPUBindGroup;
  private bgLnK!: GPUBindGroup;
  private bgFc1!: GPUBindGroup;
  private bgFc2!: GPUBindGroup;
  private bgMaskL!: GPUBindGroup;

  constructor(
    device: GPUDevice,
    private spec: JuliaSpec,
    private tensors: Map<string, GPUBuffer>,
    public readonly length: number,
    f16: boolean,
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
        size: (this.spec.layers + 5) * this.length
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
    const K = this.spec.options;
    const KB = K * this.batch;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediate;
    const eps = this.spec.normEps;
    this.pipeLn = this.pipe(KERNELS.layernorm, { N: H, MODE: 0, EPS: eps });
    const mm = KERNELS.matmul;
    this.pipeMmQkv = this.pipe(mm, { M, N: 3 * H, K: H, ACT: 0 });
    this.pipeMmAttn = this.pipe(mm, { M, N: H, K: H, ACT: 0 });
    this.pipeMmIn = this.pipe(mm, { M, N: 2 * I, K: H, ACT: 0 });
    this.pipeMmFfn = this.pipe(mm, { M, N: H, K: I, ACT: 0 });
    this.pipeMmLin1 = this.pipe(mm, { M, N: this.spec.headFfn, K: H, ACT: 1 });
    this.pipeMmLin2 = this.pipe(mm, { M, N: H, K: this.spec.headFfn, ACT: 0 });
    this.pipeMmFc1 = this.pipe(mm, { M: KB, N: H, K: H, ACT: 2 });
    this.pipeMmFc2 = this.pipe(mm, { M: KB, N: 1, K: H, ACT: 0 });
    this.pipeRope = this.pipe(KERNELS.rope, {
      L, H: this.spec.heads, D: H / this.spec.heads });
    this.pipeAttnG = this.pipe(KERNELS.mbattention, {
      L, H: this.spec.heads, D: H / this.spec.heads,
      SCALE: (H / this.spec.heads) ** -0.5, WINDOW: 0 });
    this.pipeAttnW = this.pipe(KERNELS.mbattention, {
      L, H: this.spec.heads, D: H / this.spec.heads,
      SCALE: (H / this.spec.heads) ** -0.5, WINDOW: this.spec.localAttention });
    this.pipeAdd0 = this.pipe(KERNELS.add, { TOTAL: M * H, N: H, MODE: 0, L });
    this.pipeAdd1 = this.pipe(KERNELS.add, { TOTAL: M * H, N: H, MODE: 1, L });
    this.pipeGeglu = this.pipe(KERNELS.geglu, { I });
    this.pipeGather = this.pipe(KERNELS.gather, { K, D: H, L });
    this.pipeMaskL = this.pipe(KERNELS.masklogits, { K, TEMP: 1.0 });
  }

  private buildBuffers(): void {
    const L = this.length * this.batch;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediate;
    const K = this.spec.options * this.batch;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const rows = (n: number, cols: number) => this.buf(this.bytesOf(n * cols), st);
    const src = st | GPUBufferUsage.COPY_SRC;
    this.emb = rows(L, H);
    this.x = this.buf(this.bytesOf(L * H), src);
    this.normed = rows(L, H);
    this.qkv = rows(L, 3 * H);
    this.ctx = rows(L, H);
    this.attnOut = rows(L, H);
    this.mid = rows(L, 2 * I);
    this.gate = rows(L, I);
    this.ffnOut = rows(L, H);
    this.tmp = this.buf(this.bytesOf(L * H), src);
    this.maskBuf = this.buf(L * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.packedBuf = this.buf(3 * K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const cs = ropeTable(this.length, this.spec.ropeTheta, H / this.spec.heads);
    this.cossinBuf = this.buf(
      cs.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this.cossinBuf, 0, cs);
    // One type-embedding row per sequence, selected by that row's qtype.
    this.typeRow = rows(this.batch, H);
    // Zero bias shared by every bias-free LN / matmul (max width 2*I).
    this.zeroBuf = this.buf(this.bytesOf(2 * I), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(
      this.zeroBuf, 0, new Uint8Array(this.bytesOf(2 * I)));
    this.dummyBuf = this.buf(4, GPUBufferUsage.STORAGE);
    this.states = rows(K, H);
    this.kNormed = rows(K, H);
    this.h1 = rows(K, H);
    this.raw = rows(K, 1);
    this.logitsBuf = this.buf(
      K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.staging = this.device.createBuffer({
      size: K * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.ownedBufs.push(this.staging);
    this.gpuBytes += K * 4;
    // Capture layout (f32 parity tooling): emb | layer0..L-1 | final | typed
    // | head0 | head1 == layers + 5 rows of L x H. B=1 tool, one sequence;
    // allocated lazily by captureBuffer().
  }

  private typeEmbBuf!: GPUBuffer;

  private buildBindGroups(): void {
    const dev = this.device;
    const t = this.tensors;
    const mkBg = (layout: GPUBindGroupLayout, bufs: GPUBuffer[]) => dev.createBindGroup({
      layout,
      entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const mk = (pipe: GPUComputePipeline) =>
      (bufs: GPUBuffer[]) => mkBg(pipe.getBindGroupLayout(0), bufs);
    const mkLn = mk(this.pipeLn);
    const mkQkv = mk(this.pipeMmQkv);
    const mkAttnOut = mk(this.pipeMmAttn);
    const mkRope = mk(this.pipeRope);
    const mkAttnG = mk(this.pipeAttnG);
    const mkAttnW = mk(this.pipeAttnW);
    const mkAdd0 = mk(this.pipeAdd0);
    const mkIn = mk(this.pipeMmIn);
    const mkGeglu = mk(this.pipeGeglu);
    const mkFfn = mk(this.pipeMmFfn);
    const mkLin1 = mk(this.pipeMmLin1);
    const mkLin2 = mk(this.pipeMmLin2);

    this.bgEmb = mkLn([this.emb, this.dummyBuf,
      t.get('embeddings.norm.weight')!, this.zeroBuf, this.maskBuf, this.x]);

    for (let l = 0; l < this.spec.layers; l += 1) {
      const p = `layers.${l}`;
      const global = l % this.spec.globalEvery === 0;
      const lnA = t.get(`${p}.attn_norm.weight`);
      this.layerBgs.push([
        // l == 0 has Identity attn_norm: qkv reads x directly (no LN
        // dispatch); the LN bind group stays unused for that layer.
        lnA
          ? mkLn([this.x, this.dummyBuf, lnA, this.zeroBuf, this.maskBuf,
            this.normed])
          : mkLn([this.x, this.dummyBuf, lnA ?? this.zeroBuf, this.zeroBuf,
            this.maskBuf, this.normed]),
        mkQkv([lnA ? this.normed : this.x,
          t.get(`${p}.wqkv.weight`)!, this.zeroBuf, this.qkv]),
        mkRope([this.qkv, this.cossinBuf]),
        (global ? mkAttnG : mkAttnW)([this.qkv, this.maskBuf, this.ctx]),
        mkAttnOut([this.ctx, t.get(`${p}.attn_out.weight`)!, this.zeroBuf,
          this.attnOut]),
        mkAdd0([this.x, this.attnOut]),
        mkLn([this.x, this.dummyBuf, t.get(`${p}.mlp_norm.weight`)!,
          this.zeroBuf, this.maskBuf, this.normed]),
        mkIn([this.normed, t.get(`${p}.mlp_in.weight`)!, this.zeroBuf,
          this.mid]),
        mkGeglu([this.mid, this.gate]),
        mkFfn([this.gate, t.get(`${p}.mlp_out.weight`)!, this.zeroBuf,
          this.ffnOut]),
        mkAdd0([this.x, this.ffnOut]),
      ]);
    }

    // final_norm and everything downstream work on tmp: binding x as both
    // LN input and output in one dispatch would violate usage rules, so the
    // head stream lives in tmp from final_norm on.
    this.bgFinal = mkLn([this.x, this.dummyBuf, t.get('final_norm.weight')!,
      this.zeroBuf, this.maskBuf, this.tmp]);
    this.bgType = mk(this.pipeAdd1)([this.tmp, this.typeRow]);

    const typeEmb = t.get('type_emb.weight')!;
    this.typeEmbBuf = typeEmb;
    for (let i = 0; i < this.spec.headLayers; i += 1) {
      const p = `head.${i}`;
      this.headBgs.push([
        mkLn([this.tmp, this.dummyBuf, t.get(`${p}.norm1.weight`)!,
          t.get(`${p}.norm1.bias`)!, this.maskBuf, this.normed]),
        mkQkv([this.normed, t.get(`${p}.in_proj.weight`)!,
          t.get(`${p}.in_proj.bias`)!, this.qkv]),
        mkAttnG([this.qkv, this.maskBuf, this.ctx]),
        mkAttnOut([this.ctx, t.get(`${p}.out_proj.weight`)!,
          t.get(`${p}.out_proj.bias`)!, this.attnOut]),
        mkAdd0([this.tmp, this.attnOut]),
        mkLn([this.tmp, this.dummyBuf, t.get(`${p}.norm2.weight`)!,
          t.get(`${p}.norm2.bias`)!, this.maskBuf, this.normed]),
        mkLin1([this.normed, t.get(`${p}.linear1.weight`)!,
          t.get(`${p}.linear1.bias`)!, this.mid]),
        mkLin2([this.mid, t.get(`${p}.linear2.weight`)!,
          t.get(`${p}.linear2.bias`)!, this.ffnOut]),
        mkAdd0([this.tmp, this.ffnOut]),
      ]);
    }

    this.bgGather = mk(this.pipeGather)([this.packedBuf, this.tmp, this.states]);
    this.bgLnK = mkLn([this.states, this.dummyBuf,
      t.get('scorer.norm.weight')!, t.get('scorer.norm.bias')!,
      this.maskBuf, this.kNormed]);
    this.bgFc1 = mk(this.pipeMmFc1)([this.kNormed,
      t.get('scorer.fc1.weight')!, t.get('scorer.fc1.bias')!, this.h1]);
    this.bgFc2 = mk(this.pipeMmFc2)([this.h1,
      t.get('scorer.fc2.weight')!, t.get('scorer.fc2.bias')!, this.raw]);
    this.bgMaskL = mk(this.pipeMaskL)([this.raw, this.packedBuf, this.logitsBuf]);
  }

  // qtype selects the type_emb row copied into typeRow before the typed
  // add; batch calls pass one qtype per sequence (array of batch entries).
  encode(
    capture: boolean, seqLen: number, qtype: number | number[],
    ts?: { querySet: GPUQuerySet; resolve: GPUBuffer; staging: GPUBuffer },
    dispatchNames?: string[],
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
    const capOff = this.length * this.spec.hiddenSize * 4;
    const H16 = Math.ceil(this.spec.hiddenSize / 16);
    const R16 = Math.ceil(rows / 16);
    const I16 = Math.ceil((2 * this.spec.intermediate) / 16);
    const F16 = Math.ceil(this.spec.headFfn / 16);
    const nAdd = Math.ceil(
      (this.length * this.batch * this.spec.hiddenSize) / 64);
    const H = this.spec.hiddenSize;
    const elt = this.f16 ? 2 : 4;
    // Per-layer dispatch list: [pipeline, bindgroup index, dx, dy].
    const layerOps = (l: number): [GPUComputePipeline, number, number, number, string][] => [
      [this.pipeLn, 0, rows, 1, 'lnA'],
      [this.pipeMmQkv, 1, Math.ceil(3 * H / 16), R16, 'qkv'],
      [this.pipeRope, 2, rows, 2 * this.spec.heads, 'rope'],
      [l % this.spec.globalEvery === 0 ? this.pipeAttnG : this.pipeAttnW,
        3, this.spec.heads, rows, 'attn'],
      [this.pipeMmAttn, 4, H16, R16, 'attnOut'],
      [this.pipeAdd0, 5, nAdd, 1, 'addA'],
      [this.pipeLn, 6, rows, 1, 'lnF'],
      [this.pipeMmIn, 7, I16, R16, 'ffnIn'],
      [this.pipeGeglu, 8, rows, 1, 'geglu'],
      [this.pipeMmFfn, 9, H16, R16, 'ffnOut'],
      [this.pipeAdd0, 10, nAdd, 1, 'addF'],
    ];
    const runLayers = (pass: GPUComputePassEncoder): void => {
      for (let l = 0; l < this.spec.layers; l += 1) {
        const bgs = this.layerBgs[l];
        for (const [pipe, bgi, dx, dy] of layerOps(l)) {
          if (bgi === 0 && l === 0) continue; // Identity attn_norm at layer 0
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bgs[bgi]);
          pass.dispatchWorkgroups(dx, dy);
        }
      }
    };
    const headOps = (i: number): [GPUComputePipeline, number, number, number, string][] => [
      [this.pipeLn, 0, rows, 1, 'lnA'],
      [this.pipeMmQkv, 1, Math.ceil(3 * H / 16), R16, 'qkv'],
      [this.pipeAttnG, 2, this.spec.heads, rows, 'attn'],
      [this.pipeMmAttn, 3, H16, R16, 'attnOut'],
      [this.pipeAdd0, 4, nAdd, 1, 'addA'],
      [this.pipeLn, 5, rows, 1, 'lnF'],
      [this.pipeMmLin1, 6, F16, R16, 'lin1'],
      [this.pipeMmLin2, 7, H16, R16, 'lin2'],
      [this.pipeAdd0, 8, nAdd, 1, 'addF'],
    ];
    const runHead = (pass: GPUComputePassEncoder): void => {
      for (let i = 0; i < this.spec.headLayers; i += 1) {
        const bgs = this.headBgs[i];
        for (const [pipe, bgi, dx, dy] of headOps(i)) {
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bgs[bgi]);
          pass.dispatchWorkgroups(dx, dy);
        }
      }
    };
    const scorerOps = (): [GPUComputePipeline, GPUBindGroup, number, number, string][] => {
      const KB = this.spec.options * this.batch;
      const K16 = Math.ceil(KB / 16);
      return [
        [this.pipeGather, this.bgGather, 1, 1, 'gather'],
        [this.pipeLn, this.bgLnK, KB, 1, 'ln'],
        [this.pipeMmFc1, this.bgFc1, Math.ceil(H / 16), K16, 'fc1'],
        [this.pipeMmFc2, this.bgFc2, 1, K16, 'fc2'],
        [this.pipeMaskL, this.bgMaskL, 1, 1, 'maskl'],
      ];
    };
    const runScorer = (pass: GPUComputePassEncoder): void => {
      for (const [pipe, bg, dx, dy] of scorerOps()) {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dx, dy);
      }
    };

    const qtypes = Array.isArray(qtype) ? qtype : [qtype];
    for (let b = 0; b < this.batch; b += 1) {
      enc.copyBufferToBuffer(this.typeEmbBuf,
        (qtypes[b] ?? 0) * H * elt, this.typeRow, b * H * elt, H * elt);
    }

    if (ts && dispatchNames) {
      // K27 dispatch profile: same dispatches in the same order, each in its
      // own timed compute pass.
      const step = (
        pipe: GPUComputePipeline, bg: GPUBindGroup, dx: number, dy: number, name: string,
      ): void => {
        dispatchNames.push(name);
        const pass = beginPass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dx, dy);
        pass.end();
      };
      step(this.pipeLn, this.bgEmb, rows, 1, 'embed');
      for (let l = 0; l < this.spec.layers; l += 1) {
        for (const [pipe, bgi, dx, dy, op] of layerOps(l)) {
          if (bgi === 0 && l === 0) continue;
          step(pipe, this.layerBgs[l][bgi], dx, dy, `L${l}.${op}`);
        }
      }
      step(this.pipeLn, this.bgFinal, rows, 1, 'final');
      step(this.pipeAdd1, this.bgType, nAdd, 1, 'type');
      for (let i = 0; i < this.spec.headLayers; i += 1) {
        for (const [pipe, bgi, dx, dy, op] of headOps(i)) {
          step(pipe, this.headBgs[i][bgi], dx, dy, `head${i}.${op}`);
        }
      }
      for (const [pipe, bg, dx, dy, op] of scorerOps()) {
        step(pipe, bg, dx, dy, `scorer.${op}`);
      }
      enc.copyBufferToBuffer(
        this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4);
      enc.resolveQuerySet(ts.querySet, 0, passIdx * 2, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.staging, 0, passIdx * 16);
      return enc.finish();
    }

    if (!capture && !ts) {
      const pass = beginPass();
      pass.setPipeline(this.pipeLn);
      pass.setBindGroup(0, this.bgEmb);
      pass.dispatchWorkgroups(rows);
      runLayers(pass);
      pass.setPipeline(this.pipeLn);
      pass.setBindGroup(0, this.bgFinal);
      pass.dispatchWorkgroups(rows);
      pass.setPipeline(this.pipeAdd1);
      pass.setBindGroup(0, this.bgType);
      pass.dispatchWorkgroups(nAdd);
      runHead(pass);
      runScorer(pass);
      pass.end();
      enc.copyBufferToBuffer(
        this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4);
      return enc.finish();
    }

    // Multi-pass (capture / timestamp): copies cannot live inside a pass.
    const p0 = beginPass();
    p0.setPipeline(this.pipeLn);
    p0.setBindGroup(0, this.bgEmb);
    p0.dispatchWorkgroups(rows);
    p0.end();
    if (capture) {
      enc.copyBufferToBuffer(this.x, 0, this.captureBuffer(), 0, capOff);
    }
    for (let l = 0; l < this.spec.layers; l += 1) {
      const p = beginPass();
      const bgs = this.layerBgs[l];
      for (const [pipe, bgi, dx, dy] of layerOps(l)) {
        if (bgi === 0 && l === 0) continue;
        p.setPipeline(pipe);
        p.setBindGroup(0, bgs[bgi]);
        p.dispatchWorkgroups(dx, dy);
      }
      p.end();
      if (capture) {
        enc.copyBufferToBuffer(this.x, 0, this.captureBuffer(), (l + 1) * capOff, capOff);
      }
    }
    const pf = beginPass();
    pf.setPipeline(this.pipeLn);
    pf.setBindGroup(0, this.bgFinal);
    pf.dispatchWorkgroups(rows);
    pf.end();
    if (capture) {
      enc.copyBufferToBuffer(
        this.tmp, 0, this.captureBuffer(), (this.spec.layers + 1) * capOff, capOff);
    }
    const pt = beginPass();
    pt.setPipeline(this.pipeAdd1);
    pt.setBindGroup(0, this.bgType);
    pt.dispatchWorkgroups(nAdd);
    pt.end();
    if (capture) {
      enc.copyBufferToBuffer(
        this.tmp, 0, this.captureBuffer(), (this.spec.layers + 2) * capOff, capOff);
    }
    for (let i = 0; i < this.spec.headLayers; i += 1) {
      const p = beginPass();
      const bgs = this.headBgs[i];
      for (const [pipe, bgi, dx, dy] of headOps(i)) {
        p.setPipeline(pipe);
        p.setBindGroup(0, bgs[bgi]);
        p.dispatchWorkgroups(dx, dy);
      }
      p.end();
      if (capture) {
        enc.copyBufferToBuffer(
          this.tmp, 0, this.captureBuffer(), (this.spec.layers + 3 + i) * capOff,
          capOff);
      }
    }
    const p9 = beginPass();
    runScorer(p9);
    p9.end();
    enc.copyBufferToBuffer(
      this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4);
    if (ts) {
      const bytes = passIdx * 16;
      enc.resolveQuerySet(ts.querySet, 0, passIdx * 2, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.staging, 0, bytes);
    }
    return enc.finish();
  }

  submit(capture: boolean, seqLen: number, qtype: number | number[]): void {
    this.device.queue.submit([this.encode(capture, seqLen, qtype)]);
  }

  // GPU timestamp profiling. 'pass' (default): per-pass milliseconds on the
  // full bucket. 'dispatch' (K27): every dispatch in its own timed pass at the
  // item's real seqLen, keys 'embed' | 'L<l>.<op>' | 'final' | 'type' |
  // 'head<i>.<op>' | 'scorer.<op>', plus the logits of that forward.
  async kernelTimesMs(
    input: JuliaRunInput, granularity: ProfileGranularity = 'pass', seqLen = this.length,
  ): Promise<Record<string, number> | null> {
    const r = await this.profileForward(input, granularity, seqLen);
    return r && r.times;
  }

  async profileForward(
    input: JuliaRunInput, granularity: ProfileGranularity = 'pass', seqLen = this.length,
  ): Promise<{ times: Record<string, number>; logits: Float32Array } | null> {
    if (!this.device.features.has('timestamp-query')) return null;
    const dispatch = granularity === 'dispatch';
    // 5 scorer dispatches, 11 per layer minus the skipped layer-0 norm, 9 per
    // head layer, embed + final + type.
    const dispatchCount = 3 + this.spec.layers * 11 - 1 + this.spec.headLayers * 9 + 5;
    const passCount = dispatch
      ? dispatchCount : this.spec.layers + this.spec.headLayers + 4;
    const ts = this.timestampResources(passCount);
    this.upload(input);
    const names: string[] = dispatch ? [] : [
      'embed',
      ...Array.from({ length: this.spec.layers }, (_, l) => `layer${l}`),
      'final', 'typed',
      ...Array.from({ length: this.spec.headLayers }, (_, i) => `head${i}`),
      'scorer',
    ];
    this.device.queue.submit([this.encode(
      false, dispatch ? seqLen : this.length, input.qtype, ts,
      dispatch ? names : undefined)]);
    if (names.length !== passCount) {
      throw new Error(`profile: ${names.length} timed passes, expected ${passCount}`);
    }
    await ts.staging.mapAsync(GPUMapMode.READ);
    const ns = new BigUint64Array(ts.staging.getMappedRange().slice(0));
    ts.staging.unmap();
    const times: Record<string, number> = {};
    for (let i = 0; i < passCount; i += 1) {
      times[names[i]] = Number(ns[i * 2 + 1] - ns[i * 2]) / 1e6;
    }
    return { times, logits: await this.readLogits() };
  }

  private tsResources?: DispatchTimestamps & { count: number };

  private timestampResources(count: number): DispatchTimestamps {
    if (this.tsResources && this.tsResources.count === count) return this.tsResources;
    if (this.tsResources) {
      this.tsResources.querySet.destroy();
      this.tsResources.resolve.destroy();
      this.tsResources.staging.destroy();
    }
    const bytes = count * 16;
    this.tsResources = {
      count,
      querySet: this.device.createQuerySet({ type: 'timestamp', count: count * 2 }),
      resolve: this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      staging: this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    return this.tsResources;
  }

  upload(input: JuliaRunInput): void {
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
