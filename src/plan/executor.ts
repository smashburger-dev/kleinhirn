// WebGPU executor for a Plan (K28 design section 3). Buffers, pipelines and
// bind groups are created once at construction; a call uploads the inputs,
// encodes one command buffer from the prepared per-segment lists, submits,
// and reads back the logits. Exact one-call-at-a-time discipline lives in
// the engines (index.ts, julia.ts).
// Pass structure: the normal path is one compute pass for the whole graph.
// Capture and pass-level profiling use one pass per segment (buffer copies
// cannot live inside a pass); dispatch-level profiling uses one pass per
// dispatch. Capture stays a B=1 f32 parity tool.

import { MINIMUM_LIMITS } from '../device.ts';
import { halfBitsToFloat32 } from '../half.ts';
import { KERNELS, wgsl } from '../kernels/index.ts';
import type { BufferDecl, Dim, KernelName, Op, Plan } from './ir.ts';

export type ProfileGranularity = 'pass' | 'dispatch';

export interface DispatchTimestamps {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  staging: GPUBuffer;
}

export interface RunInput {
  embeddings: Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>;
  mask: Float32Array<ArrayBuffer>;         // L
  packedMarkers?: Uint32Array<ArrayBuffer>; // 3*K: idx, mask f32 bits, groups
  typeIds?: Uint32Array<ArrayBuffer>;       // one u32 per row (absolute-position families)
  // Julia: type row per sequence (one entry per batch element in a batch).
  qtype?: number | number[];
}

export interface EncodeOptions {
  seqLen?: number;           // rows to dispatch (capture uses the full bucket)
  skip?: Set<number>;        // debug: op indices to leave out of skippable segments
  qtype?: number | number[];
  ts?: DispatchTimestamps;
  dispatchNames?: string[];  // set: dispatch-level profile, names are appended
}

interface Prepared {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  dispatch: [Dim, Dim];
  name: string;
  alts: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; dispatch: [Dim, Dim]; maxRows: number }[];
}

interface PreparedSegment {
  name: string;
  prefix: string;
  ops: Prepared[];
  captureOps: Prepared[];
  capture: { buffer: GPUBuffer; slot: number }[];
  skippable: boolean;
}

const USAGE: Record<BufferDecl['usage'], () => GPUBufferUsageFlags> = {
  rw: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  rwSrc: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  storage: () => GPUBufferUsage.STORAGE,
  logits: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
};

const constantsKey = (c: Record<string, number>): string =>
  Object.keys(c).sort().map((k) => `${k}=${c[k]}`).join(',');

export class PlanExecutor {
  readonly length: number;
  readonly markers: number;
  readonly batch: number;
  gpuBytes = 0;

  private buffers = new Map<string, GPUBuffer>();
  private ownedBufs: GPUBuffer[] = [];
  private staging!: GPUBuffer;
  // Lazy: capture is a B=1 parity tool, so plans that never capture skip
  // this large buffer entirely.
  private capBuf?: GPUBuffer;
  private segments: PreparedSegment[] = [];
  private typeTable?: GPUBuffer;
  private typeRow?: GPUBuffer;

  constructor(
    private device: GPUDevice,
    private plan: Plan,
    private tensors: Map<string, GPUBuffer>,
  ) {
    this.length = plan.length;
    this.markers = plan.markers;
    this.batch = plan.batch;
    this.buildBuffers();
    this.buildSegments();
    if (plan.rowSelect) {
      this.typeTable = this.resolve(plan.rowSelect.table);
      this.typeRow = this.resolve(plan.rowSelect.dst);
    }
  }

  private track(b: GPUBuffer, bytes: number): GPUBuffer {
    this.ownedBufs.push(b);
    this.gpuBytes += bytes;
    return b;
  }

  private resolve(id: string): GPUBuffer {
    if (id.startsWith('w:')) {
      const b = this.tensors.get(id.slice(2));
      if (!b) throw new Error(`weight tensor ${id.slice(2)} missing`);
      return b;
    }
    const b = this.buffers.get(id);
    if (!b) throw new Error(`plan buffer ${id} missing`);
    return b;
  }

  private buildBuffers(): void {
    for (const d of this.plan.buffers) {
      const b = this.track(
        this.device.createBuffer({ size: d.bytes, usage: USAGE[d.usage]() }), d.bytes);
      this.buffers.set(d.id, b);
      if (d.init) this.device.queue.writeBuffer(b, 0, d.init as GPUAllowSharedBufferSource);
    }
    const bytes = this.plan.output.bytes;
    this.staging = this.track(this.device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), bytes);
  }

  private buildSegments(): void {
    const pipes = new Map<string, GPUComputePipeline>();
    const pipeFor = (op: Pick<Op, 'kernel' | 'constants'>): GPUComputePipeline => {
      const key = `${op.kernel}|${constantsKey(op.constants)}`;
      let p = pipes.get(key);
      if (!p) {
        p = this.device.createComputePipeline({
          layout: 'auto',
          compute: {
            module: this.device.createShaderModule({
              code: wgsl(KERNELS[op.kernel as KernelName], this.plan.f16) }),
            entryPoint: 'main',
            constants: op.constants,
          },
        });
        pipes.set(key, p);
      }
      return p;
    };
    // The layout comes from the pipeline that runs the bind group.
    const groupFor = (pipeline: GPUComputePipeline, op: Op): GPUBindGroup => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: op.bind.map((id, i) => ({
        binding: i, resource: { buffer: this.resolve(id) } })),
    });
    const prepare = (op: Op): Prepared => {
      const pipeline = pipeFor(op);
      return {
        pipeline, bindGroup: groupFor(pipeline, op), dispatch: op.dispatch, name: op.name,
        alts: (op.alts ?? []).map((a) => {
          const p = pipeFor({ kernel: a.kernel, constants: op.constants });
          return { pipeline: p, bindGroup: groupFor(p, op), dispatch: a.dispatch, maxRows: a.maxRows };
        }),
      };
    };
    for (const seg of this.plan.segments) {
      this.segments.push({
        name: seg.name, prefix: seg.prefix,
        ops: seg.ops.map(prepare),
        captureOps: (seg.captureOps ?? []).map(prepare),
        capture: (seg.capture ?? []).map((c) => ({
          buffer: this.resolve(c.buffer), slot: c.slot })),
        skippable: !!seg.skippable,
      });
    }
  }

  // Capture is an f32 B=1 parity tool: the copies are f32 sized and
  // readCapture reads f32. Anything else, or a capture buffer over maxBufferSize,
  // throws; the engines call this before the upload.
  assertCapturable(): void {
    if (this.plan.f16) throw new Error('capture needs an f32 plan: the capture copies and readCapture are f32');
    if (this.plan.batch !== 1) throw new Error(`capture needs a plan of batch 1, this one has ${this.plan.batch}`);
    const bytes = this.plan.captureSlots * this.plan.captureSlotBytes;
    const limit = this.device.limits?.maxBufferSize ?? MINIMUM_LIMITS.maxBufferSize;
    if (bytes > limit) throw new Error(`capture buffer needs ${bytes} B, maxBufferSize is ${limit}`);
  }

  private captureBuffer(): GPUBuffer {
    if (!this.capBuf) {
      const bytes = this.plan.captureSlots * this.plan.captureSlotBytes;
      this.capBuf = this.track(this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), bytes);
    }
    return this.capBuf;
  }

  destroy(): void {
    for (const b of this.ownedBufs) b.destroy();
    this.ownedBufs.length = 0;
  }

  private dim(d: Dim, rows: number): number {
    return d === 'rows' ? rows : d === 'rows8' ? Math.ceil(rows / 8) : d === 'rows16' ? Math.ceil(rows / 16) : d === 'rows32' ? Math.ceil(rows / 32) : d;
  }

  private run(pass: GPUComputePassEncoder, op: Prepared, rows: number): void {
    const o = op.alts.find((a) => rows <= a.maxRows) ?? op;
    pass.setPipeline(o.pipeline);
    pass.setBindGroup(0, o.bindGroup);
    pass.dispatchWorkgroups(this.dim(o.dispatch[0], rows), this.dim(o.dispatch[1], rows));
  }

  // Encode the full pass into one command buffer. Rows at index >= seqLen are
  // masked everywhere downstream, so row-dispatch dimensions use seqLen
  // (capture keeps the full bucket for the parity tooling). ts: optional
  // timestamp-query resources; each compute pass then writes begin/end GPU
  // timestamps that are resolved into ts.staging for the caller to map.
  encode(capture: boolean, o: EncodeOptions = {}): GPUCommandBuffer {
    const { ts, skip, dispatchNames } = o;
    if (capture) this.assertCapturable();
    const enc = this.device.createCommandEncoder();
    let passIdx = 0;
    const beginPass = (): GPUComputePassEncoder => {
      const writes = ts ? {
        querySet: ts.querySet,
        beginningOfPassWriteIndex: passIdx * 2,
        endOfPassWriteIndex: passIdx * 2 + 1,
      } : undefined;
      passIdx += 1;
      return enc.beginComputePass(writes ? { timestampWrites: writes } : undefined);
    };
    const rows = capture ? this.length : (o.seqLen ?? this.length);
    const kept = (seg: PreparedSegment): Prepared[] => seg.skippable && skip
      ? seg.ops.filter((_, i) => !skip.has(i)) : seg.ops;

    // Julia: one type-embedding row per sequence, selected by that
    // sequence's qtype, copied before the first pass in every mode.
    const rs = this.plan.rowSelect;
    if (rs) {
      const qtypes = Array.isArray(o.qtype) ? o.qtype : [o.qtype ?? 0];
      for (let b = 0; b < this.batch; b += 1) {
        enc.copyBufferToBuffer(this.typeTable as GPUBuffer,
          (qtypes[b] ?? 0) * rs.rowBytes, this.typeRow as GPUBuffer,
          b * rs.rowBytes, rs.rowBytes);
      }
    }

    if (dispatchNames) {
      if (!ts) throw new Error('dispatch profile needs timestamp resources');
      // Same dispatches in the same order, each in its own timed pass.
      for (const seg of this.segments) {
        for (const op of seg.ops) {
          dispatchNames.push(`${seg.prefix}${op.name}`);
          const pass = beginPass();
          this.run(pass, op, rows);
          pass.end();
        }
      }
    } else if (!capture && !ts) {
      // Dispatches inside a pass are still program-ordered, so the pass
      // boundaries between segments are pure overhead here.
      const pass = beginPass();
      for (const seg of this.segments) {
        for (const op of kept(seg)) this.run(pass, op, rows);
      }
      pass.end();
    } else {
      const capOff = this.plan.captureSlotBytes;
      for (const seg of this.segments) {
        const pass = beginPass();
        if (capture) for (const op of seg.captureOps) this.run(pass, op, rows);
        for (const op of kept(seg)) this.run(pass, op, rows);
        pass.end();
        if (capture) {
          for (const c of seg.capture) {
            enc.copyBufferToBuffer(c.buffer, 0, this.captureBuffer(), c.slot * capOff, capOff);
          }
        }
      }
    }
    enc.copyBufferToBuffer(
      this.resolve(this.plan.output.buffer), 0, this.staging, 0, this.plan.output.bytes);
    if (ts) {
      enc.resolveQuerySet(ts.querySet, 0, passIdx * 2, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.staging, 0, passIdx * 16);
    }
    return enc.finish();
  }

  submit(capture: boolean, o: EncodeOptions = {}): void {
    this.device.queue.submit([this.encode(capture, o)]);
  }

  // GPU timestamp profiling (K5, K27): one timed forward. 'pass' (default)
  // keys the segment names on the full bucket; 'dispatch' times each
  // dispatch at the item's real seqLen, keys '<prefix><op>', and also
  // returns the logits of that forward. Null when the device lacks
  // 'timestamp-query'; the caller then uses CPU-side timing.
  async kernelTimesMs(
    input: RunInput, granularity: ProfileGranularity = 'pass', seqLen = this.length,
  ): Promise<Record<string, number> | null> {
    const r = await this.profileForward(input, granularity, seqLen);
    return r && r.times;
  }

  async profileForward(
    input: RunInput, granularity: ProfileGranularity = 'pass', seqLen = this.length,
  ): Promise<{ times: Record<string, number>; logits: Float32Array } | null> {
    if (!this.device.features.has('timestamp-query')) return null;
    const dispatch = granularity === 'dispatch';
    const passCount = dispatch
      ? this.segments.reduce((n, s) => n + s.ops.length, 0)
      : this.segments.length;
    const ts = this.timestampResources(passCount);
    this.upload(input);
    const names: string[] = dispatch ? [] : this.segments.map((s) => s.name);
    this.device.queue.submit([this.encode(false, {
      seqLen: dispatch ? seqLen : this.length, qtype: input.qtype, ts,
      dispatchNames: dispatch ? names : undefined })]);
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

  upload(input: RunInput): void {
    const i = this.plan.inputs;
    this.device.queue.writeBuffer(this.resolve(i.embeddings), 0, input.embeddings);
    this.device.queue.writeBuffer(this.resolve(i.mask), 0, input.mask);
    if (this.plan.buffers.some((b) => b.id === 'kinfo')) {
      // First and last valid key of every sequence (K27): attention kernels skip tiles outside.
      const L = this.plan.length;
      const info = new Uint32Array(2 * this.plan.batch);
      for (let b = 0; b < this.plan.batch; b += 1) {
        let first = L;
        let last = 0;
        for (let j = 0; j < L; j += 1) {
          if (input.mask[b * L + j] > 0.5) { if (first === L) first = j; last = j; }
        }
        info[2 * b] = first;
        info[2 * b + 1] = last;
      }
      this.device.queue.writeBuffer(this.resolve('kinfo'), 0, info);
    }
    if (i.markers) {
      if (!input.packedMarkers) throw new Error('plan needs packedMarkers');
      this.device.queue.writeBuffer(this.resolve(i.markers), 0, input.packedMarkers);
    }
    if (i.typeIds) {
      if (!input.typeIds) throw new Error('plan needs typeIds');
      this.device.queue.writeBuffer(this.resolve(i.typeIds), 0, input.typeIds);
    }
  }

  // The output as f32: rows * cols values. A 'storage' output of an f16 plan
  // is converted from binary16; an f32 plan copies the bits as they are.
  // One mapAsync per call, like readLogits.
  async readOutput(): Promise<Float32Array> {
    const o = this.plan.output;
    await this.staging.mapAsync(GPUMapMode.READ);
    const raw = this.staging.getMappedRange().slice(0);
    this.staging.unmap();
    const n = o.rows * o.cols;
    if (o.dtype === 'storage' && this.plan.f16) {
      return halfBitsToFloat32(new Uint16Array(raw, 0, n));
    }
    return new Float32Array(raw, 0, n);
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
