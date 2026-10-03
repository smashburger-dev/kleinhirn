// Public API: Kleinhirn.load / classify / info / dispose, plus the internal
// runPrepared used by "nur Modell" benchmarking and parity checks.
// Exactly one GPU call is in flight at a time; further calls queue.

import { getDevice, scopedCall, scopedSync, type KhDevice } from './device.ts';
import { fetchManifest, loadWeights, type LoadedWeights } from './weights.ts';
import { HfTokenizer } from './tokenizer/tokenizer.ts';
import { BucketOverflowError, prepareTasks, type SchemaInput } from './tokenizer/schema.ts';
import { buildPlan } from './plan/build.ts';
import { assertPlan, fitBatchPlan } from './plan/check.ts';
import type { Plan } from './plan/ir.ts';
import { PlanExecutor, type ProfileGranularity } from './plan/executor.ts';
import { specFromGlinerManifest, type EncoderSpec } from './plan/spec.ts';
import { JuliaEngine } from './julia.ts';
import { assertFinite, tokenRowOffset } from './tasks.ts';
import { WasmClient } from './wasm-client.ts';
import { EncoderModel } from './encoder.ts';
import {
  LruCache, MAX_BATCH, batchStride, dedupMerge, nextBatchSize,
  schemaMergeKey, sortByKey,
} from './cache.ts';

export { BucketOverflowError };
export { EncoderModel };
export type {
  EncoderInput, EncoderLoadOptions, EncoderOutput, RerankResult, RunOptions, TextClassification,
  TokenSpan, ZeroShotResult,
} from './encoder.ts';
export { JsonTokenizer } from './tokenizer/hf/index.ts';

// Entry point honouring the backend option / fallback chain. Returns
// Kleinhirn (DeBERTa) or JuliaEngine (modernbert-julia arch) for the GPU
// paths and a worker-hosted WasmClient otherwise; all expose the run/info/
// dispose surface. Manifests of format kleinhirn-weights-2 give an
// EncoderModel (WebGPU only).
export async function loadEngine(
  options: LoadOptions,
): Promise<Kleinhirn | JuliaEngine | WasmClient | EncoderModel> {
  const backend = options.backend ?? 'auto';
  if (backend === 'wasm') {
    const mf = await fetchManifest(options.manifestUrl);
    if (isJuliaManifest(mf)) {
      throw new Error('julia-1 needs WebGPU; the wasm fallback covers DeBERTa only');
    }
    if (isEncoderManifest(mf)) {
      throw new Error('kleinhirn-weights-2 models need WebGPU; the wasm backend does not run them yet');
    }
    return WasmClient.load(options);
  }
  const gpuMissing = typeof navigator === 'undefined' || !navigator.gpu;
  if (!gpuMissing) {
    const manifest = await fetchManifest(options.manifestUrl);
    if (isJuliaManifest(manifest)) {
      return JuliaEngine.load(options, manifest);
    }
    if (isEncoderManifest(manifest)) {
      return EncoderModel.load({
        manifestUrl: options.manifestUrl, precision: options.precision, limits: options.limits,
        buckets: options.buckets?.map((b) => (typeof b === 'number' ? b : b.length)),
      });
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
    if (isEncoderManifest(mf)) {
      throw new Error('kleinhirn-weights-2 models need WebGPU; the wasm backend does not run them yet');
    }
  }
  return WasmClient.load(options);
}

function isJuliaManifest(mf: { encoder?: Record<string, unknown> }): boolean {
  return mf.encoder?.arch === 'modernbert-julia';
}

function isEncoderManifest(mf: { format?: string }): boolean {
  return mf.format === 'kleinhirn-weights-2';
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
  private plans = new Map<number, PlanExecutor>();
  private batchPlans = new Map<string, PlanExecutor>();
  // `${stride}:${markers}:${asked batch}` -> the batch size that fits the device (1: none, use the bucket plan)
  private batchFit = new Map<string, number>();
  private queue: Promise<unknown> = Promise.resolve();
  private downloadBytes = 0;
  private tokenizerBytes = 0;
  private weightGpuBytes = 0;
  private loadTiming: Record<string, number> = {};
  private cache = new LruCache<PreparedResult>(0);
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
    engine.weightGpuBytes = weights.gpuBytes;
    engine.loadTiming = {
      ...weights.timing, manifestMs, tokenizerMs: tokenizerMs, planMs: 0 };
    engine.headHidden = Number(weights.manifest.head.hiddenSize) || 768;
    engine.cache = new LruCache(options.cacheSize ?? 256);
    const tPlan = performance.now();
    await scopedSync(kh, () => {
      for (const bucket of options.buckets ?? [128]) {
        const spec2 = typeof bucket === 'number'
          ? { length: bucket, markers: DEFAULT_MARKERS }
          : { length: bucket.length, markers: bucket.markers ?? DEFAULT_MARKERS };
        const bucketPlan = engine.planFor(spec2.length, spec2.markers, 1);
        assertPlan(bucketPlan, kh.device.limits, (name) => weights.tensors.get(name)?.size);
        engine.plans.set(spec2.length, new PlanExecutor(kh.device, bucketPlan, weights.tensors));
      }
    });
    engine.loadTiming.planMs = performance.now() - tPlan;
    return engine;
  }

  private planFor(length: number, markers: number, batch: number): Plan {
    const { spec, head } = specFromGlinerManifest(
      this.spec, { temperature: this.temperature, hiddenSize: this.headHidden }, markers);
    return buildPlan(spec, head, {
      length, batch, markers, f16: this.precision === 'f16' });
  }

  // Smallest length bucket that fits seqLen and routes at least `markers`
  // label markers. Wide requests (>16 labels) need a bucket declared with
  // a matching markers budget; a plain length match is not enough.
  private pickBucket(seqLen: number, markers = 1): PlanExecutor {
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
    input: SchemaInput, plan: PlanExecutor,
  ): Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer> {
    const L = plan.length;
    const hidden = this.spec.hiddenSize;
    if (this.weights.embeddings instanceof Float32Array) {
      const rows = new Float32Array(L * hidden);
      const emb = this.weights.embeddings;
      for (let i = 0; i < input.seqLen; i += 1) {
        const off = tokenRowOffset(input.inputIds[i], hidden, emb.length);
        rows.set(emb.subarray(off, off + hidden), i * hidden);
      }
      return rows;
    }
    const rows = new Uint16Array(L * hidden);
    const emb = this.weights.embeddings;
    for (let i = 0; i < input.seqLen; i += 1) {
      const off = tokenRowOffset(input.inputIds[i], hidden, emb.length);
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
      : this.plans.get(bucket) as PlanExecutor;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (input.seqLen > plan.length || nMarkers > plan.markers) {
      throw new BucketOverflowError(
        `seqLen ${input.seqLen}/${nMarkers} markers exceeds bucket ${plan.length}/${plan.markers}`);
    }
    if (capture) plan.assertCapturable();
    const key = schemaMergeKey(input);
    const hit = capture ? undefined : this.cache.get(key);
    if (hit) return copyResult(hit);
    return this.enqueue(() => scopedCall(this.kh, () => {
      plan.upload({
        embeddings: this.embeddingRows(input, plan),
        mask: this.maskOf(input, plan.length),
        packedMarkers: this.packedMarkers(input, plan.markers),
      });
      plan.submit(capture, { seqLen: input.seqLen });
    }, async () => {
      const logits = await plan.readLogits();
      assertFinite(logits, 'logits');
      const probabilities = groupSoftmax(
        logits, input.markerGroups, input.markerMask);
      if (!capture) this.cache.set(key, copyResult({ logits, probabilities }));
      const captureData = capture ? await plan.readCapture() : undefined;
      return { logits, probabilities, captureData };
    }));
  }

  // Lazily built batch plan for row stride at batch size B; stride is the
  // row block each sequence occupies, not a weight bucket. It is the plan of
  // the largest size up to B that fits the device limits (checkPlan), or
  // undefined when none does and the rows run on the bucket plan. Plans are
  // capped: distinct strides would otherwise grow GPU memory without bound.
  private batchPlan(
    stride: number, markers: number, batch: number,
  ): PlanExecutor | undefined {
    const asked = `${stride}:${markers}:${batch}`;
    const memo = this.batchFit.get(asked);
    if (memo === 1) return undefined;
    const known = this.batchPlans.get(`${stride}:${markers}:${memo ?? batch}`);
    if (known) return known;
    const plan = memo === undefined
      ? fitBatchPlan((b) => this.planFor(stride, markers, b), this.kh.device.limits, batch)
      : this.planFor(stride, markers, memo);
    this.batchFit.set(asked, plan ? plan.batch : 1);
    if (!plan) return undefined;
    if (this.batchPlans.size >= MAX_BATCH_PLANS) {
      const oldest = this.batchPlans.keys().next().value as string;
      this.batchPlans.get(oldest)?.destroy();
      this.batchPlans.delete(oldest);
    }
    const p = new PlanExecutor(this.kh.device, plan, this.weights.tensors);
    this.batchPlans.set(`${stride}:${markers}:${plan.batch}`, p);
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
    inputs: SchemaInput[], plan: PlanExecutor,
  ): Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer> {
    const hidden = this.spec.hiddenSize;
    const emb = this.weights.embeddings;
    const out = emb instanceof Float32Array
      ? new Float32Array(plan.length * plan.batch * hidden)
      : new Uint16Array(plan.length * plan.batch * hidden);
    for (const [b, input] of inputs.entries()) {
      const base = b * plan.length * hidden;
      for (let i = 0; i < input.seqLen; i += 1) {
        const off = tokenRowOffset(input.inputIds[i], hidden, emb.length);
        out.set(emb.subarray(off, off + hidden), base + i * hidden);
      }
    }
    return out;
  }

  private batchMask(
    inputs: SchemaInput[], plan: PlanExecutor,
  ): Float32Array<ArrayBuffer> {
    const mask = new Float32Array(plan.length * plan.batch);
    for (const [b, input] of inputs.entries()) {
      mask.fill(1, b * plan.length, b * plan.length + input.seqLen);
    }
    return mask;
  }

  private batchPackedMarkers(
    inputs: SchemaInput[], plan: PlanExecutor,
  ): Uint32Array<ArrayBuffer> {
    const packed = new Uint32Array(inputs.length * 3 * plan.markers);
    for (const [b, input] of inputs.entries()) {
      packed.set(this.packedMarkers(input, plan.markers), b * 3 * plan.markers);
    }
    return packed;
  }

  // One GPU batch chunk over `inputs` (1..MAX_BATCH rows). The bucket covers
  // the widest marker count; the batch plan's row stride follows the chunk's
  // longest sequence (quantized by batchStride), so short sequences do not
  // pay for the full bucket. Choosing, building and evicting the batch plan
  // happen inside the queued function, next to upload, submit and readback.
  // The plan that fits the device may be smaller than the chunk (or the
  // bucket plan, B1): the rows then run in pieces of that plan's batch size,
  // one after the other in the same queued function (a nested enqueue would
  // wait on itself).
  private async runBatchChunk(
    inputs: SchemaInput[], bucket?: number,
  ): Promise<PreparedResult[]> {
    const maxSeq = Math.max(...inputs.map((i) => i.seqLen));
    const maxMarkers = Math.max(...inputs.map(
      (i) => i.markerMask.reduce((n, v) => n + (v > 0.5 ? 1 : 0), 0)));
    const plan = bucket === undefined
      ? this.pickBucket(maxSeq, maxMarkers)
      : this.plans.get(bucket) as PlanExecutor;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (maxSeq > plan.length || maxMarkers > plan.markers) {
      throw new BucketOverflowError(
        `batch exceeds bucket ${plan.length}/${plan.markers}`);
    }
    const n = nextBatchSize(inputs.length);
    // B=1 runs on the bucket plan like the single path and dispatches
    // only the real rows; B>1 packs at the quantized stride.
    const stride = Math.min(plan.length, batchStride(maxSeq));
    return this.enqueue(async () => {
      const out: PreparedResult[] = [];
      let bPlan: PlanExecutor | undefined;
      while (out.length < inputs.length) {
        const piece = await scopedCall(this.kh, () => {
          bPlan ??= (n === 1 ? undefined : this.batchPlan(stride, plan.markers, n)) ?? plan;
          const rows = inputs.slice(out.length, out.length + bPlan.batch);
          this.submitPiece(rows, bPlan);
          return { rows, plan: bPlan };
        }, (p) => this.readPiece(p.rows, p.plan));
        out.push(...piece);
      }
      return out;
    });
  }

  // Upload and submit of one piece (the synchronous part of a scoped call).
  private submitPiece(inputs: SchemaInput[], bPlan: PlanExecutor): void {
    const padded = inputs.length === bPlan.batch
      ? inputs
      : [...inputs, ...Array.from(
        { length: bPlan.batch - inputs.length }, () => this.padInput())];
    const rows = bPlan.batch === 1
      ? Math.max(...inputs.map((i) => i.seqLen)) : bPlan.length * bPlan.batch;
    bPlan.upload({
      embeddings: this.batchEmbeddingRows(padded, bPlan),
      mask: this.batchMask(padded, bPlan),
      packedMarkers: this.batchPackedMarkers(padded, bPlan),
    });
    bPlan.submit(false, { seqLen: rows });
  }

  private async readPiece(inputs: SchemaInput[], bPlan: PlanExecutor): Promise<PreparedResult[]> {
    const all = await bPlan.readLogits();
    return inputs.map((input, i) => {
      const logits = all.slice(
        i * bPlan.markers, (i + 1) * bPlan.markers);
      assertFinite(logits, 'logits');
      const res = {
        logits,
        probabilities: groupSoftmax(
          logits, input.markerGroups, input.markerMask),
      };
      this.cache.set(schemaMergeKey(input), copyResult(res));
      return res;
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
        results[i] = copyResult(hit);
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
    // The first row of a slot gets the computed arrays, duplicates their own copies.
    const handedOut = new Set<number>();
    for (const [i] of pending.entries()) {
      const r = uniqueResults[sorted.slot[i]] as PreparedResult;
      results[pendingIdx[i]] = handedOut.has(sorted.slot[i]) ? copyResult(r) : r;
      handedOut.add(sorted.slot[i]);
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

  // K5 GPU timestamp profiling: one timed forward. granularity 'pass'
  // (default): per-pass milliseconds on the seqLen bucket. 'dispatch': every
  // dispatch in its own timed pass at the item's real seqLen (K27). Null
  // when 'timestamp-query' is unavailable.
  async profile(
    input: SchemaInput, options: { granularity?: ProfileGranularity } = {},
  ): Promise<Record<string, number> | null> {
    const r = await this.profileDetailed(input, options);
    return r && r.times;
  }

  // Same as profile() plus the logits of the profiled forward (parity check
  // of the dispatch path against the normal one).
  async profileDetailed(
    input: SchemaInput, options: { granularity?: ProfileGranularity } = {},
  ): Promise<{ times: Record<string, number>; logits: Float32Array } | null> {
    const nMarkers = input.markerMask.reduce(
      (n, v) => n + (v > 0.5 ? 1 : 0), 0);
    const plan = this.pickBucket(input.seqLen, nMarkers);
    return this.enqueue(() => plan.profileForward({
      embeddings: this.embeddingRows(input, plan),
      mask: this.maskOf(input, plan.length),
      packedMarkers: this.packedMarkers(input, plan.markers),
    }, options.granularity ?? 'pass', input.seqLen));
  }

  // Weights plus every live plan (a capture buffer counts once it exists);
  // an evicted batch plan is gone from the map and from the sum.
  private liveGpuBytes(): number {
    let n = this.weightGpuBytes;
    for (const p of this.plans.values()) n += p.gpuBytes;
    for (const p of this.batchPlans.values()) n += p.gpuBytes;
    return n;
  }

  info(): Record<string, unknown> {
    return {
      precision: this.precision,
      buildId: typeof __KH_BUILD_ID__ === 'string' ? __KH_BUILD_ID__ : 'dev',
      adapter: this.kh.adapterInfo,
      limitsMode: this.kh.limitsMode,
      timestamps: this.kh.hasTimestamps,
      buckets: [...this.plans.keys()].sort((a, b) => a - b),
      gpuBytes: this.liveGpuBytes(),
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

// Cache entries and caller results never share arrays: a caller that edits
// its logits cannot change what a later identical call returns.
function copyResult(r: PreparedResult): PreparedResult {
  return { logits: r.logits.slice(), probabilities: r.probabilities.slice() };
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
