// Public API: Kleinhirn.load / classify / info / dispose, plus the internal
// runPrepared used by "nur Modell" benchmarking and parity checks.
// Exactly one GPU call is in flight at a time; further calls queue.

import { getDevice, type KhDevice } from './device.ts';
import { fetchManifest, loadWeights, type LoadedWeights } from './weights.ts';
import { HfTokenizer } from './tokenizer/tokenizer.ts';
import { BucketOverflowError, prepareTasks, type SchemaInput } from './tokenizer/schema.ts';
import { EncoderPlan, type EncoderSpec } from './graph/deberta.ts';
import { JuliaEngine } from './julia.ts';
import { WasmClient } from './wasm-client.ts';
import {
  LruCache, MAX_BATCH, batchStride, dedupMerge, nextBatchSize,
  schemaMergeKey, sortByKey,
} from './cache.ts';

export { BucketOverflowError };

// Entry point honouring the backend option / fallback chain. Returns
// Kleinhirn (DeBERTa) or JuliaEngine (modernbert-julia arch) for the GPU
// paths and a worker-hosted WasmClient otherwise; all expose the run/info/
// dispose surface.
export async function loadEngine(
  options: LoadOptions,
): Promise<Kleinhirn | JuliaEngine | WasmClient> {
  const backend = options.backend ?? 'auto';
  if (backend === 'wasm') {
    const mf = await fetchManifest(options.manifestUrl);
    if (isJuliaManifest(mf)) {
      throw new Error('julia-1 needs WebGPU; the wasm fallback covers DeBERTa only');
    }
    return WasmClient.load(options);
  }
  const gpuMissing = typeof navigator === 'undefined' || !navigator.gpu;
  if (!gpuMissing) {
    const manifest = await fetchManifest(options.manifestUrl);
    if (isJuliaManifest(manifest)) {
      return JuliaEngine.load(options, manifest);
    }
    try {
      return await Kleinhirn.load(options);
    } catch (err) {
      if (backend === 'webgpu') throw err;
    }
  } else {
    const mf = await fetchManifest(options.manifestUrl);
    if (isJuliaManifest(mf)) {
      throw new Error('julia-1 needs WebGPU; the wasm fallback covers DeBERTa only');
    }
  }
  return WasmClient.load(options);
}

function isJuliaManifest(mf: { encoder?: Record<string, unknown> }): boolean {
  return mf.encoder?.arch === 'modernbert-julia';
}

declare const __KH_BUILD_ID__: string;

export interface BucketSpec {
  length: number;
  // Label markers the bucket routes through the head (default 16).
  // Wide classification (>16 labels, e.g. banking77's 72) needs a bucket
  // with markers >= labels; all buffers stay far under the 128 MiB
  // binding limit, so no raised device limit is required.
  markers?: number;
}

export interface LoadOptions {
  manifestUrl: string;
  buckets?: Array<number | BucketSpec>;
  precision?: 'auto' | 'f16' | 'f32';
  // 'minimum' requests only the WebGPU spec-required limits (the standard
  // test, guarantees mobile-class adapters); 'default' leaves limit
  // negotiation to the adapter.
  limits?: 'minimum' | 'default';
  // Fallback chain (docs/PLAN.md K5): 'auto' = WebGPU f16/f32 then WASM-SIMD
  // in a Worker when navigator.gpu is absent or no adapter arrives;
  // 'webgpu' forces the GPU path; 'wasm' forces the CPU path.
  backend?: 'auto' | 'webgpu' | 'wasm';
  // Merge-keyed result cache size in entries (K16); 0 disables the cache.
  cacheSize?: number;
}

export interface ClassifyLabel {
  label: string;
  probability: number;
  logit: number;
}

export interface ClassifyResult {
  tasks: { task: string; labels: ClassifyLabel[] }[];
  timings: { tokenizeMs: number; gpuMs: number; totalMs: number };
}

export interface PreparedResult {
  logits: Float32Array;
  probabilities: Float32Array;
  // f32-only parity output: [embLn | embMasked | layer_0..N-1], each L x hidden.
  captureData?: Float32Array;
}

const DEFAULT_MARKERS = 16;
// Stride-quantized batch plans are lazily built per (stride, markers, B);
// the cap evicts the least recently inserted plan when exceeded.
const MAX_BATCH_PLANS = 8;

export class Kleinhirn {
  private plans = new Map<number, EncoderPlan>();
  private batchPlans = new Map<string, EncoderPlan>();
  private queue: Promise<unknown> = Promise.resolve();
  private downloadBytes = 0;
  private tokenizerBytes = 0;
  private gpuBytes = 0;
  private loadTiming: Record<string, number> = {};
  private cache = new LruCache<PreparedResult>(0);
  private embLN!: { weight: GPUBuffer; bias: GPUBuffer };
  private head!: { fc1w: GPUBuffer; fc1b: GPUBuffer; fc2w: GPUBuffer; fc2b: GPUBuffer };
  private headHidden = 768;

  private constructor(
    private kh: KhDevice,
    private weights: LoadedWeights,
    private tokenizer: HfTokenizer,
    private spec: EncoderSpec,
    private temperature: number,
    public readonly precision: 'f16' | 'f32',
  ) {}

  static async load(options: LoadOptions): Promise<Kleinhirn> {
    const preferF16 = options.precision !== 'f32';
    const base = options.manifestUrl.slice(0, options.manifestUrl.lastIndexOf('/') + 1);
    // Device, manifest, shards and the tokenizer all resolve independently;
    // overlapping them removes the tokenizer's parse from the critical path.
    const tMf = performance.now();
    const manifestPromise = fetchManifest(options.manifestUrl);
    const kh = await getDevice(preferF16, options.limits !== 'default');
    const manifest = await manifestPromise;
    const manifestMs = performance.now() - tMf;
    const tokPromise = (async () =>
      (await fetch(base + manifest.tokenizer)).arrayBuffer())();
    const weights = await loadWeights(kh.device, options.manifestUrl, manifest);
    const first = weights.manifest.tensors.find((t) => t.name.endsWith('qkv.weight'));
    const dtype = first?.dtype ?? 'f32';
    if (dtype === 'f16' && !kh.hasF16) {
      throw new Error('f16 manifest but adapter lacks shader-f16; use the f32 manifest');
    }
    if (options.precision && options.precision !== 'auto' && options.precision !== dtype) {
      throw new Error(`precision ${options.precision} requested but manifest is ${dtype}`);
    }
    const precision = dtype as 'f16' | 'f32';
    const tTok = performance.now();
    const tokBuf = await tokPromise;
    const tokenizer = new HfTokenizer(
      JSON.parse(new TextDecoder().decode(tokBuf)));
    const tokenizerMs = performance.now() - tTok;
    const spec = weights.manifest.encoder as unknown as EncoderSpec;
    const engine = new Kleinhirn(
      kh, weights, tokenizer, spec, weights.manifest.head.temperature, precision);
    engine.downloadBytes = weights.downloadBytes + tokBuf.byteLength;
    engine.tokenizerBytes = tokBuf.byteLength;
    engine.gpuBytes = weights.gpuBytes;
    engine.loadTiming = {
      ...weights.timing, manifestMs, tokenizerMs: tokenizerMs, planMs: 0 };
    const t = weights.tensors;
    const embLN = {
      weight: t.get('embeddings.LayerNorm.weight')!,
      bias: t.get('embeddings.LayerNorm.bias')!,
    };
    const head = {
      fc1w: t.get('head.fc1.weight')!, fc1b: t.get('head.fc1.bias')!,
      fc2w: t.get('head.fc2.weight')!, fc2b: t.get('head.fc2.bias')!,
    };
    const headHidden = Number(weights.manifest.head.hiddenSize) || 768;
    engine.embLN = embLN;
    engine.head = head;
    engine.headHidden = headHidden;
    engine.cache = new LruCache(options.cacheSize ?? 256);
    const tPlan = performance.now();
    for (const bucket of options.buckets ?? [128]) {
      const spec2 = typeof bucket === 'number'
        ? { length: bucket, markers: DEFAULT_MARKERS }
        : { length: bucket.length, markers: bucket.markers ?? DEFAULT_MARKERS };
      const plan = new EncoderPlan(
        kh.device, spec, t, embLN, head, engine.temperature, spec2.length,
        precision === 'f16', spec2.markers, headHidden);
      engine.gpuBytes += plan.gpuBytes;
      engine.plans.set(spec2.length, plan);
    }
    engine.loadTiming.planMs = performance.now() - tPlan;
    return engine;
  }

  // Smallest length bucket that fits seqLen and routes at least `markers`
  // label markers. Wide requests (>16 labels) need a bucket declared with
  // a matching markers budget; a plain length match is not enough.
  private pickBucket(seqLen: number, markers = 1): EncoderPlan {
    const fits = [...this.plans.values()]
      .filter((p) => seqLen <= p.length && markers <= p.markers)
      .sort((a, b) => a.length - b.length || a.markers - b.markers);
    if (!fits.length) {
      throw new BucketOverflowError(
        `seqLen ${seqLen} or ${markers} markers exceed loaded buckets`);
    }
    return fits[0];
  }

  private embeddingRows(
    input: SchemaInput, plan: EncoderPlan,
  ): Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer> {
    const L = plan.length;
    const hidden = this.spec.hiddenSize;
    if (this.weights.embeddings instanceof Float32Array) {
      const rows = new Float32Array(L * hidden);
      const emb = this.weights.embeddings;
      for (let i = 0; i < input.seqLen; i += 1) {
        const off = input.inputIds[i] * hidden;
        rows.set(emb.subarray(off, off + hidden), i * hidden);
      }
      return rows;
    }
    const rows = new Uint16Array(L * hidden);
    const emb = this.weights.embeddings;
    for (let i = 0; i < input.seqLen; i += 1) {
      const off = input.inputIds[i] * hidden;
      rows.set(emb.subarray(off, off + hidden), i * hidden);
    }
    return rows;
  }

  private packedMarkers(
    input: SchemaInput, markers: number,
  ): Uint32Array<ArrayBuffer> {
    const packed = new Uint32Array(3 * markers);
    packed.set(input.markerIndices.subarray(0, markers), 0);
    packed.set(new Uint32Array(input.markerMask.buffer).subarray(0, markers), markers);
    packed.set(input.markerGroups.subarray(0, markers), 2 * markers);
    return packed;
  }

  private maskOf(input: SchemaInput, length: number): Float32Array<ArrayBuffer> {
    const mask = new Float32Array(length);
    mask.fill(1, 0, input.seqLen);
    return mask;
  }

  // Serialized single-flight runner. capture is for the f32 parity path only.
  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch(() => {});
    return run;
  }

  async runPrepared(
    input: SchemaInput, capture = false, bucket?: number,
  ): Promise<PreparedResult> {
    const nMarkers = input.markerMask.reduce(
      (n, v) => n + (v > 0.5 ? 1 : 0), 0);
    const plan = bucket === undefined
      ? this.pickBucket(input.seqLen, nMarkers)
      : this.plans.get(bucket) as EncoderPlan;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (input.seqLen > plan.length || nMarkers > plan.markers) {
      throw new BucketOverflowError(
        `seqLen ${input.seqLen}/${nMarkers} markers exceeds bucket ${plan.length}/${plan.markers}`);
    }
    const key = schemaMergeKey(input);
    const hit = capture ? undefined : this.cache.get(key);
    if (hit) return { logits: hit.logits, probabilities: hit.probabilities };
    return this.enqueue(async () => {
      plan.upload({
        embeddings: this.embeddingRows(input, plan),
        mask: this.maskOf(input, plan.length),
        packedMarkers: this.packedMarkers(input, plan.markers),
      });
      plan.submit(capture, undefined, input.seqLen);
      const logits = await plan.readLogits();
      const probabilities = groupSoftmax(
        logits, input.markerGroups, input.markerMask);
      if (!capture) this.cache.set(key, { logits, probabilities });
      const captureData = capture ? await plan.readCapture() : undefined;
      return { logits, probabilities, captureData };
    });
  }

  // Lazily built batch plan for row stride at batch size B; stride is the
  // row block each sequence occupies, not a weight bucket. The widest
  // buffer is checked against the binding limit up front:
  // B*stride*max(3H, I) and B*K*max(H, headHidden) elements must stay
  // under it. Plans are capped: distinct strides would otherwise grow
  // GPU memory without bound.
  private batchPlan(
    stride: number, markers: number, batch: number,
  ): EncoderPlan {
    const key = `${stride}:${markers}:${batch}`;
    let p = this.batchPlans.get(key);
    if (p) return p;
    const elt = this.precision === 'f16' ? 2 : 4;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediateSize;
    const limit = this.kh.device.limits.maxStorageBufferBindingSize;
    const worst = Math.max(
      batch * stride * Math.max(3 * H, I),
      batch * markers * Math.max(H, this.headHidden)) * elt;
    if (worst > limit) {
      throw new BucketOverflowError(
        `batch ${batch} x stride ${stride} needs ${worst} B > ${limit} B binding limit`);
    }
    if (this.batchPlans.size >= MAX_BATCH_PLANS) {
      const oldest = this.batchPlans.keys().next().value as string;
      this.batchPlans.get(oldest)?.destroy();
      this.batchPlans.delete(oldest);
    }
    p = new EncoderPlan(
      this.kh.device, this.spec, this.weights.tensors, this.embLN, this.head,
      this.temperature, stride, this.precision === 'f16',
      markers, this.headHidden, batch);
    this.gpuBytes += p.gpuBytes;
    this.batchPlans.set(key, p);
    return p;
  }

  // All-zero padding row: seqLen 0 masks every position, so the row feeds
  // only masked work and its logits slot is discarded by the caller.
  private padInput(): SchemaInput {
    return {
      inputIds: new Int32Array(0), attentionMask: new Int32Array(0),
      markerIndices: new Int32Array(0), markerMask: new Float32Array(0),
      markerGroups: new Int32Array(0), seqLen: 0,
    };
  }

  private batchEmbeddingRows(
    inputs: SchemaInput[], plan: EncoderPlan,
  ): Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer> {
    const hidden = this.spec.hiddenSize;
    const emb = this.weights.embeddings;
    const out = emb instanceof Float32Array
      ? new Float32Array(plan.length * plan.batch * hidden)
      : new Uint16Array(plan.length * plan.batch * hidden);
    for (const [b, input] of inputs.entries()) {
      const base = b * plan.length * hidden;
      for (let i = 0; i < input.seqLen; i += 1) {
        const off = input.inputIds[i] * hidden;
        out.set(emb.subarray(off, off + hidden), base + i * hidden);
      }
    }
    return out;
  }

  private batchMask(
    inputs: SchemaInput[], plan: EncoderPlan,
  ): Float32Array<ArrayBuffer> {
    const mask = new Float32Array(plan.length * plan.batch);
    for (const [b, input] of inputs.entries()) {
      mask.fill(1, b * plan.length, b * plan.length + input.seqLen);
    }
    return mask;
  }

  private batchPackedMarkers(
    inputs: SchemaInput[], plan: EncoderPlan,
  ): Uint32Array<ArrayBuffer> {
    const packed = new Uint32Array(inputs.length * 3 * plan.markers);
    for (const [b, input] of inputs.entries()) {
      packed.set(this.packedMarkers(input, plan.markers), b * 3 * plan.markers);
    }
    return packed;
  }

  // One GPU batch pass over `inputs` (1..MAX_BATCH rows after padding).
  // The bucket covers the widest marker count; the batch plan's row
  // stride follows the chunk's longest sequence (quantized by
  // batchStride), so short sequences do not pay for the full bucket.
  private async runBatchChunk(
    inputs: SchemaInput[], bucket?: number,
  ): Promise<PreparedResult[]> {
    const maxSeq = Math.max(...inputs.map((i) => i.seqLen));
    const maxMarkers = Math.max(...inputs.map(
      (i) => i.markerMask.reduce((n, v) => n + (v > 0.5 ? 1 : 0), 0)));
    const plan = bucket === undefined
      ? this.pickBucket(maxSeq, maxMarkers)
      : this.plans.get(bucket) as EncoderPlan;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (maxSeq > plan.length || maxMarkers > plan.markers) {
      throw new BucketOverflowError(
        `batch exceeds bucket ${plan.length}/${plan.markers}`);
    }
    const n = nextBatchSize(inputs.length);
    // B=1 runs on the bucket plan like the single path and dispatches
    // only the real rows; B>1 packs at the quantized stride.
    const stride = n === 1
      ? maxSeq
      : Math.min(plan.length, batchStride(maxSeq));
    let bPlan: EncoderPlan;
    try {
      bPlan = n === 1 ? plan : this.batchPlan(stride, plan.markers, n);
    } catch (e) {
      // A batch this size exceeds the binding limit (e.g. base f32 L1024
      // B16); halving the chunk fits, so retry instead of failing.
      if (!(e instanceof BucketOverflowError) || inputs.length < 2) throw e;
      const mid = Math.ceil(inputs.length / 2);
      const a = await this.runBatchChunk(inputs.slice(0, mid), bucket);
      const b = await this.runBatchChunk(inputs.slice(mid), bucket);
      return [...a, ...b];
    }
    const padded = inputs.length === bPlan.batch
      ? inputs
      : [...inputs, ...Array.from(
        { length: bPlan.batch - inputs.length }, () => this.padInput())];
    return this.enqueue(async () => {
      bPlan.upload({
        embeddings: this.batchEmbeddingRows(padded, bPlan),
        mask: this.batchMask(padded, bPlan),
        packedMarkers: this.batchPackedMarkers(padded, bPlan),
      });
      bPlan.submit(false, undefined, stride * bPlan.batch);
      const all = await bPlan.readLogits();
      return inputs.map((input, i) => {
        const logits = all.slice(
          i * bPlan.markers, (i + 1) * bPlan.markers);
        const res = {
          logits,
          probabilities: groupSoftmax(
            logits, input.markerGroups, input.markerMask),
        };
        this.cache.set(schemaMergeKey(input), res);
        return res;
      });
    });
  }

  // Batch runner (K16): cache hits skip the GPU, identical merge keys in
  // one call are deduplicated, and unique rows are chunked into groups of
  // at most MAX_BATCH before padding up to the next supported batch size.
  async runPreparedBatch(
    inputs: SchemaInput[], bucket?: number,
  ): Promise<PreparedResult[]> {
    const results: (PreparedResult | undefined)[] = new Array(inputs.length);
    const pending: SchemaInput[] = [];
    const pendingIdx: number[] = [];
    for (const [i, input] of inputs.entries()) {
      const hit = this.cache.get(schemaMergeKey(input));
      if (hit) {
        results[i] = { logits: hit.logits, probabilities: hit.probabilities };
      } else {
        pending.push(input);
        pendingIdx.push(i);
      }
    }
    const { unique, slot } = dedupMerge(pending, schemaMergeKey);
    const sorted = sortByKey(unique, slot, (i) => i.seqLen);
    const uniqueResults: (PreparedResult | undefined)[] = [];
    for (let start = 0; start < sorted.unique.length; start += MAX_BATCH) {
      const chunkResults = await this.runBatchChunk(
        sorted.unique.slice(start, start + MAX_BATCH), bucket);
      uniqueResults.push(...chunkResults);
    }
    for (const [i] of pending.entries()) {
      results[pendingIdx[i]] = uniqueResults[sorted.slot[i]];
    }
    return results as PreparedResult[];
  }

  async classify(
    text: string,
    tasks: { task: string; labels: string[] }[],
  ): Promise<ClassifyResult> {
    const t0 = performance.now();
    const tuples = tasks.map((t): [string, string[]] => [t.task, t.labels]);
    const nMarkers = tuples.reduce((n, [, labels]) => n + labels.length, 0);
    const input = prepareTasks(
      this.tokenizer, text, tuples, this.maxBucket(nMarkers), nMarkers);
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const { logits, probabilities } = await this.runPrepared(input);
    const gpuMs = performance.now() - g0;
    return this.toResult(
      tasks, input, logits, probabilities,
      { tokenizeMs, gpuMs, totalMs: performance.now() - t0 });
  }

  // Batch classification (K16): every item carries text plus the same
  // task/label schema shape as classify. Items share the GPU batch in
  // groups of up to 16 (deduplicated by merge key, cached, padded to
  // batch sizes 1/4/8/16); order is preserved.
  async classifyBatch(
    items: { text: string; tasks: { task: string; labels: string[] }[] }[],
  ): Promise<ClassifyResult[]> {
    const t0 = performance.now();
    const prepared = items.map((item) => {
      const tuples = item.tasks.map((t): [string, string[]] => [t.task, t.labels]);
      const nMarkers = tuples.reduce((n, [, labels]) => n + labels.length, 0);
      return prepareTasks(
        this.tokenizer, item.text, tuples,
        this.maxBucket(nMarkers), nMarkers);
    });
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const outputs = await this.runPreparedBatch(prepared);
    const gpuMs = performance.now() - g0;
    const totalMs = performance.now() - t0;
    // gpuMs is the shared batch wall time; a cache-hit row reports the same
    // aggregate, which matches the per-request accounting of classify().
    const timings = { tokenizeMs, gpuMs, totalMs };
    return items.map((item, i) => this.toResult(
      item.tasks, prepared[i],
      outputs[i].logits, outputs[i].probabilities, timings));
  }

  private toResult(
    tasks: { task: string; labels: string[] }[],
    input: SchemaInput,
    logits: Float32Array,
    probabilities: Float32Array,
    timings: { tokenizeMs: number; gpuMs: number; totalMs: number },
  ): ClassifyResult {
    return {
      tasks: tasks.map(({ task, labels }, g) => ({
        task,
        labels: labels.map((label, i) => {
          const k = indexOfGroupMarker(input.markerGroups, g, i);
          return k < 0
            ? { label, probability: 0, logit: -1e4 }
            : { label, probability: probabilities[k], logit: logits[k] };
        }),
      })),
      timings,
    };
  }

  // Largest sequence bound among plans that can route `markers` labels.
  private maxBucket(markers = 1): number {
    const lens = [...this.plans.values()]
      .filter((p) => p.markers >= markers)
      .map((p) => p.length);
    if (!lens.length) {
      throw new BucketOverflowError(`no bucket routes ${markers} markers`);
    }
    return Math.max(...lens);
  }

  // K5 GPU timestamp profiling: one timed forward on the seqLen bucket,
  // per-pass milliseconds or null when 'timestamp-query' is unavailable.
  async profile(input: SchemaInput): Promise<Record<string, number> | null> {
    const nMarkers = input.markerMask.reduce(
      (n, v) => n + (v > 0.5 ? 1 : 0), 0);
    const plan = this.pickBucket(input.seqLen, nMarkers);
    return this.enqueue(() => plan.kernelTimesMs({
      embeddings: this.embeddingRows(input, plan),
      mask: this.maskOf(input, plan.length),
      packedMarkers: this.packedMarkers(input, plan.markers),
    }));
  }

  info(): Record<string, unknown> {
    return {
      precision: this.precision,
      buildId: typeof __KH_BUILD_ID__ === 'string' ? __KH_BUILD_ID__ : 'dev',
      adapter: this.kh.adapterInfo,
      limitsMode: this.kh.limitsMode,
      timestamps: this.kh.hasTimestamps,
      buckets: [...this.plans.keys()].sort((a, b) => a - b),
      gpuBytes: this.gpuBytes,
      downloadBytes: this.downloadBytes,
      tokenizerBytes: this.tokenizerBytes,
      weightBytes: this.downloadBytes - this.tokenizerBytes,
      loadTiming: this.loadTiming,
    };
  }

  dispose(): void {
    this.kh.device.destroy();
  }
}

// Softmax per marker group over valid markers; invalid markers get 0.
export function groupSoftmax(
  logits: Float32Array,
  groups: Int32Array,
  mask: Float32Array,
): Float32Array {
  const out = new Float32Array(logits.length);
  const byGroup = new Map<number, number[]>();
  for (let k = 0; k < logits.length; k += 1) {
    if (mask[k] <= 0.5) continue;
    const g = groups[k];
    if (!byGroup.has(g)) byGroup.set(g, []);
    (byGroup.get(g) as number[]).push(k);
  }
  for (const idxs of byGroup.values()) {
    let mx = -Infinity;
    for (const k of idxs) mx = Math.max(mx, logits[k]);
    let sum = 0;
    for (const k of idxs) sum += Math.exp(logits[k] - mx);
    for (const k of idxs) out[k] = Math.exp(logits[k] - mx) / sum;
  }
  return out;
}

function indexOfGroupMarker(groups: Int32Array, group: number, ordinal: number): number {
  let seen = 0;
  for (let k = 0; k < groups.length; k += 1) {
    if (groups[k] === group) {
      if (seen === ordinal) return k;
      seen += 1;
    }
  }
  return -1;
}
