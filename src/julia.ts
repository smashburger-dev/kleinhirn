// Julia 1 (ModernBERT + decision head) on the WebGPU path (K8). Same
// load/classify/info shape as Kleinhirn; routing happens in index.ts on
// manifest.encoder.arch === 'modernbert-julia'. Embeddings stay on the CPU
// (256k x 384 exceeds the 128 MiB binding limit); sequence assembly follows
// julia/data.py via prepareDecision.

import { getDevice, type KhDevice } from './device.ts';
import { fetchManifest, loadWeights, type LoadedWeights, type Manifest } from './weights.ts';
import { BpeTokenizer } from './tokenizer/bpe.ts';
import { BucketOverflowError } from './tokenizer/schema.ts';
import {
  prepareDecision, type JuliaInput, type JuliaRequest,
} from './tokenizer/julia-input.ts';
import { JuliaPlan, type JuliaSpec } from './graph/julia.ts';
import type { LoadOptions } from './index.ts';
import {
  LruCache, MAX_BATCH, batchStride, dedupMerge, juliaMergeKey,
  nextBatchSize, sortByKey,
} from './cache.ts';

export interface JuliaPreparedInput {
  inputIds: Int32Array;
  markers: number[];
  qtype: number;
  seqLen: number;
}

export interface JuliaPreparedResult {
  logits: Float32Array;
  probabilities: Float32Array;
  // f32-only parity output: [emb | layer0..L-1 | final | typed | head0 |
  // head1], each L x hidden.
  captureData?: Float32Array;
}

export interface DecisionResult {
  logits: Float32Array;
  probabilities: Float32Array;
  choice: number;
  timings: { tokenizeMs: number; gpuMs: number; totalMs: number };
}

function softmaxPrefix(logits: Float32Array, valid: number): Float32Array {
  const out = new Float32Array(valid);
  let mx = -Infinity;
  for (let k = 0; k < valid; k += 1) mx = Math.max(mx, logits[k]);
  let sum = 0;
  for (let k = 0; k < valid; k += 1) sum += Math.exp(logits[k] - mx);
  for (let k = 0; k < valid; k += 1) out[k] = Math.exp(logits[k] - mx) / sum;
  return out;
}

// Stride-quantized batch plans are lazily built per (stride, B); the
// cap evicts the least recently inserted plan when exceeded.
const MAX_BATCH_PLANS = 8;

export class JuliaEngine {
  private plans = new Map<number, JuliaPlan>();
  private batchPlans = new Map<string, JuliaPlan>();
  private queue: Promise<unknown> = Promise.resolve();
  private downloadBytes = 0;
  private tokenizerBytes = 0;
  private gpuBytes = 0;
  private loadTiming: Record<string, number> = {};
  private cache = new LruCache<JuliaPreparedResult>(0);

  private constructor(
    private kh: KhDevice,
    private weights: LoadedWeights,
    private tokenizer: BpeTokenizer,
    private spec: JuliaSpec,
    public readonly precision: 'f16' | 'f32',
  ) {}

  static async load(options: LoadOptions, prefetched?: Manifest): Promise<JuliaEngine> {
    const preferF16 = options.precision !== 'f32';
    const base = options.manifestUrl.slice(0, options.manifestUrl.lastIndexOf('/') + 1);
    const tMf = performance.now();
    const manifestPromise = prefetched
      ? Promise.resolve(prefetched)
      : fetchManifest(options.manifestUrl);
    const kh = await getDevice(preferF16, options.limits !== 'default');
    const manifest = await manifestPromise;
    const manifestMs = performance.now() - tMf;
    const tokPromise = (async () =>
      (await fetch(base + manifest.tokenizer)).arrayBuffer())();
    const weights = await loadWeights(kh.device, options.manifestUrl, manifest);
    const first = weights.manifest.tensors.find((t) => t.name.endsWith('wqkv.weight'));
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
    const tokenizer = new BpeTokenizer(
      JSON.parse(new TextDecoder().decode(tokBuf)));
    const tokenizerMs = performance.now() - tTok;
    const spec = manifest.encoder as unknown as JuliaSpec;
    const engine = new JuliaEngine(kh, weights, tokenizer, spec, precision);
    engine.downloadBytes = weights.downloadBytes + tokBuf.byteLength;
    engine.tokenizerBytes = tokBuf.byteLength;
    engine.gpuBytes = weights.gpuBytes;
    engine.loadTiming = {
      ...weights.timing, manifestMs, tokenizerMs, planMs: 0 };
    const tPlan = performance.now();
    for (const bucket of options.buckets ?? [512]) {
      const length = typeof bucket === 'number' ? bucket : bucket.length;
      const plan = new JuliaPlan(
        kh.device, spec, weights.tensors, length, precision === 'f16');
      engine.gpuBytes += plan.gpuBytes;
      engine.plans.set(length, plan);
    }
    engine.cache = new LruCache(options.cacheSize ?? 256);
    engine.loadTiming.planMs = performance.now() - tPlan;
    return engine;
  }

  private pickBucket(seqLen: number): JuliaPlan {
    const fits = [...this.plans.keys()].sort((a, b) => a - b)
      .find((b) => seqLen <= b);
    if (fits === undefined) {
      throw new BucketOverflowError(`seqLen ${seqLen} exceeds largest bucket`);
    }
    return this.plans.get(fits) as JuliaPlan;
  }

  private embeddingRows(
    input: JuliaPreparedInput, plan: JuliaPlan,
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

  private packedMarkers(input: JuliaPreparedInput): Uint32Array<ArrayBuffer> {
    const K = this.spec.options;
    const packed = new Uint32Array(3 * K);
    for (let i = 0; i < input.markers.length && i < K; i += 1) {
      packed[i] = input.markers[i];
    }
    const mask = new Float32Array(K);
    mask.fill(1, 0, Math.min(input.markers.length, K));
    packed.set(new Uint32Array(mask.buffer), K);
    return packed;
  }

  private maskOf(input: JuliaPreparedInput, length: number): Float32Array<ArrayBuffer> {
    const mask = new Float32Array(length);
    mask.fill(1, 0, input.seqLen);
    return mask;
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch(() => {});
    return run;
  }

  async runPrepared(
    input: JuliaPreparedInput, capture = false, bucket?: number,
  ): Promise<JuliaPreparedResult> {
    const plan = bucket === undefined
      ? this.pickBucket(input.seqLen)
      : this.plans.get(bucket) as JuliaPlan;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (input.seqLen > plan.length) {
      throw new BucketOverflowError(
        `seqLen ${input.seqLen} exceeds bucket ${plan.length}`);
    }
    const key = juliaMergeKey(input);
    const hit = capture ? undefined : this.cache.get(key);
    if (hit) return { logits: hit.logits, probabilities: hit.probabilities };
    return this.enqueue(async () => {
      plan.upload({
        embeddings: this.embeddingRows(input, plan),
        mask: this.maskOf(input, plan.length),
        packedMarkers: this.packedMarkers(input),
        qtype: input.qtype,
      });
      plan.submit(capture, input.seqLen, input.qtype);
      const logits = await plan.readLogits();
      const probabilities = softmaxPrefix(
        logits, Math.min(input.markers.length, this.spec.options));
      if (!capture) this.cache.set(key, { logits, probabilities });
      const captureData = capture ? await plan.readCapture() : undefined;
      return { logits, probabilities, captureData };
    });
  }

  // Lazily built batch plan for row stride at batch size B; stride is
  // the row block each sequence occupies, not a weight bucket. The
  // widest buffer B*stride*max(3H, 2I) elements is checked against the
  // binding limit before allocation. Plans are capped so distinct
  // strides cannot grow GPU memory without bound.
  private batchPlan(stride: number, batch: number): JuliaPlan {
    const key = `${stride}:${batch}`;
    let p = this.batchPlans.get(key);
    if (p) return p;
    const elt = this.precision === 'f16' ? 2 : 4;
    const H = this.spec.hiddenSize;
    const I = this.spec.intermediate;
    const limit = this.kh.device.limits.maxStorageBufferBindingSize;
    const worst = batch * stride * Math.max(3 * H, 2 * I) * elt;
    if (worst > limit) {
      throw new BucketOverflowError(
        `batch ${batch} x stride ${stride} needs ${worst} B > ${limit} B binding limit`);
    }
    if (this.batchPlans.size >= MAX_BATCH_PLANS) {
      const oldest = this.batchPlans.keys().next().value as string;
      this.batchPlans.get(oldest)?.destroy();
      this.batchPlans.delete(oldest);
    }
    p = new JuliaPlan(
      this.kh.device, this.spec, this.weights.tensors, stride,
      this.precision === 'f16', batch);
    this.gpuBytes += p.gpuBytes;
    this.batchPlans.set(key, p);
    return p;
  }

  private padInput(): JuliaPreparedInput {
    return {
      inputIds: new Int32Array(0), markers: [], qtype: 0, seqLen: 0,
    };
  }

  private batchEmbeddingRows(
    inputs: JuliaPreparedInput[], plan: JuliaPlan,
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
    inputs: JuliaPreparedInput[], plan: JuliaPlan,
  ): Float32Array<ArrayBuffer> {
    const mask = new Float32Array(plan.length * plan.batch);
    for (const [b, input] of inputs.entries()) {
      mask.fill(1, b * plan.length, b * plan.length + input.seqLen);
    }
    return mask;
  }

  private batchPackedMarkers(
    inputs: JuliaPreparedInput[],
  ): Uint32Array<ArrayBuffer> {
    const packed = new Uint32Array(inputs.length * 3 * this.spec.options);
    for (const [b, input] of inputs.entries()) {
      packed.set(this.packedMarkers(input), b * 3 * this.spec.options);
    }
    return packed;
  }

  // One GPU batch pass over `inputs` (1..MAX_BATCH rows after padding).
  private async runBatchChunk(
    inputs: JuliaPreparedInput[], bucket?: number,
  ): Promise<JuliaPreparedResult[]> {
    const maxSeq = Math.max(...inputs.map((i) => i.seqLen));
    const plan = bucket === undefined
      ? this.pickBucket(maxSeq)
      : this.plans.get(bucket) as JuliaPlan;
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (maxSeq > plan.length) {
      throw new BucketOverflowError(
        `batch seqLen ${maxSeq} exceeds bucket ${plan.length}`);
    }
    const n = nextBatchSize(inputs.length);
    // B=1 runs on the bucket plan like the single path and dispatches
    // only the real rows; B>1 packs at the quantized stride.
    const stride = n === 1
      ? maxSeq
      : Math.min(plan.length, batchStride(maxSeq));
    let bPlan: JuliaPlan;
    try {
      bPlan = n === 1 ? plan : this.batchPlan(stride, n);
    } catch (e) {
      // A batch this size exceeds the binding limit (e.g. f32 L1024 B16);
      // halving the chunk fits, so retry recursively instead of failing.
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
        packedMarkers: this.batchPackedMarkers(padded),
        qtype: 0,
      });
      bPlan.submit(false, stride * bPlan.batch,
        padded.map((p) => p.qtype));
      const all = await bPlan.readLogits();
      const K = this.spec.options;
      return inputs.map((input, i) => {
        const logits = all.slice(i * K, (i + 1) * K);
        const res = {
          logits,
          probabilities: softmaxPrefix(
            logits, Math.min(input.markers.length, K)),
        };
        this.cache.set(juliaMergeKey(input), res);
        return res;
      });
    });
  }

  // Batch runner (K16): cache hits skip the GPU, identical merge keys are
  // deduplicated inside the call, uniques chunk into groups of at most
  // MAX_BATCH and pad to batch sizes 1/4/8/16. Order is preserved.
  async runPreparedBatch(
    inputs: JuliaPreparedInput[], bucket?: number,
  ): Promise<JuliaPreparedResult[]> {
    const results: (JuliaPreparedResult | undefined)[] = new Array(inputs.length);
    const pending: JuliaPreparedInput[] = [];
    const pendingIdx: number[] = [];
    for (const [i, input] of inputs.entries()) {
      const hit = this.cache.get(juliaMergeKey(input));
      if (hit) {
        results[i] = { logits: hit.logits, probabilities: hit.probabilities };
      } else {
        pending.push(input);
        pendingIdx.push(i);
      }
    }
    const { unique, slot } = dedupMerge(pending, juliaMergeKey);
    const sorted = sortByKey(unique, slot, (i) => i.seqLen);
    const uniqueResults: (JuliaPreparedResult | undefined)[] = [];
    for (let start = 0; start < sorted.unique.length; start += MAX_BATCH) {
      const chunkResults = await this.runBatchChunk(
        sorted.unique.slice(start, start + MAX_BATCH), bucket);
      uniqueResults.push(...chunkResults);
    }
    for (const [i] of pending.entries()) {
      results[pendingIdx[i]] = uniqueResults[sorted.slot[i]];
    }
    return results as JuliaPreparedResult[];
  }

  // Strict-mode input assembly for protocol-faithful evaluation (K9):
  // throws on marker injection, >48-token options, or any truncation,
  // matching julia/data.py:sequence(strict=True).
  prepare(row: JuliaRequest, maxLength = 1024, headLength = 512,
    strict = false): JuliaInput {
    return prepareDecision(this.tokenizer, row, maxLength, headLength, strict);
  }

  async decide(row: JuliaRequest): Promise<DecisionResult> {
    const t0 = performance.now();
    const input = prepareDecision(this.tokenizer, row, this.maxBucket(), 256);
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const { logits, probabilities } = await this.runPrepared({
      inputIds: input.inputIds, markers: input.markers,
      qtype: input.qtype, seqLen: input.seqLen,
    });
    const gpuMs = performance.now() - g0;
    let choice = 0;
    for (let k = 1; k < input.markers.length; k += 1) {
      if (logits[k] > logits[choice]) choice = k;
    }
    return {
      logits, probabilities, choice,
      timings: { tokenizeMs, gpuMs, totalMs: performance.now() - t0 },
    };
  }

  // Batch decisions (K16): rows tokenize independently, then share GPU
  // batches in groups of up to 16 with merge-key dedup and the result
  // cache; order and choice selection match decide() row for row.
  async decideBatch(rows: JuliaRequest[]): Promise<DecisionResult[]> {
    const t0 = performance.now();
    const inputs = rows.map((row) => {
      const input = prepareDecision(this.tokenizer, row, this.maxBucket(), 256);
      return {
        inputIds: input.inputIds, markers: input.markers,
        qtype: input.qtype, seqLen: input.seqLen,
      };
    });
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const outputs = await this.runPreparedBatch(inputs);
    const gpuMs = performance.now() - g0;
    const totalMs = performance.now() - t0;
    const timings = { tokenizeMs, gpuMs, totalMs };
    return rows.map((_, i) => {
      const { logits, probabilities } = outputs[i];
      let choice = 0;
      for (let k = 1; k < inputs[i].markers.length; k += 1) {
        if (logits[k] > logits[choice]) choice = k;
      }
      return { logits, probabilities, choice, timings };
    });
  }

  private maxBucket(): number {
    return Math.max(...this.plans.keys());
  }

  async profile(input: JuliaPreparedInput): Promise<Record<string, number> | null> {
    const plan = this.pickBucket(input.seqLen);
    return this.enqueue(() => plan.kernelTimesMs({
      embeddings: this.embeddingRows(input, plan),
      mask: this.maskOf(input, plan.length),
      packedMarkers: this.packedMarkers(input),
      qtype: input.qtype,
    }));
  }

  info(): Record<string, unknown> {
    return {
      precision: this.precision,
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
