// EncoderModel (K28.4, K28.3): encoders in the manifest format
// kleinhirn-weights-2 (BERT family: sequence classification, NLI, reranking,
// token classification, embeddings). runIds and runIdsBatch take token ids
// and return f32 output; classify, zeroShot, rerank, embed and tokenClassify
// take text (tokenizer.json next to the manifest) and go through them. Same
// discipline as Kleinhirn in index.ts: one plan per bucket, exactly one
// call in flight, one output read per call. The plans run behind the Backend
// seam (src/backend.ts).

import { BATCH_SIZES, MAX_BATCH, batchStride, nextBatchSize } from './cache.ts';
import { f32ManifestUrl, loadBackend, type Backend, type PlanRunner } from './backend.ts';
import type { Transport } from './wasm-backend.ts';
import { getDevice } from './device.ts';
import { buildPlan } from './plan/build.ts';
import type { Plan } from './plan/ir.ts';
import { positionRows, type HeadSpec, type ModelSpec } from './plan/spec.ts';
import { JsonTokenizer, type TruncationStrategy } from './tokenizer/hf/index.ts';
import { BucketOverflowError } from './tokenizer/schema.ts';
import { aggregateSimple, argmaxRows, assertFinite, completeLabels, l2normalize, softmax, tokenRowOffset, type EntitySpan } from './tasks.ts';
import { fetchManifest, resolveManifest } from './weights.ts';

declare const __KH_BUILD_ID__: string;

export interface EncoderLoadOptions {
  manifestUrl: string;
  precision?: 'auto' | 'f16' | 'f32';
  // Bucket lengths (rows per sequence). Default [128, 512].
  buckets?: number[];
  // 'minimum' requests only the WebGPU spec-required limits.
  limits?: 'minimum' | 'default';
  // Token limit of the text methods. Default: the model's position rows (and
  // the Sentence-Transformers max_seq_length for embeddings), at most the
  // largest bucket.
  maxLength?: number;
  // 'webgpu' (default) or 'wasm': the plan executor of src/wasm/plan.ts in a worker, f32
  // manifest (an .../f16/... URL maps to its f32 sibling), R2.
  backend?: 'webgpu' | 'wasm';
  // Threads of the WASM path (R8): 'auto' is the core count, at most 8; more than one needs a cross-origin
  // isolated page (SharedArrayBuffer), else the path runs one thread and info() says why.
  threads?: number | 'auto';
  // Build of the WASM executor (R8 hc6): 'auto' (default) takes the relaxed-SIMD build where the
  // browser validates it, 'plain' and 'relaxed' force one.
  wasmBuild?: 'auto' | 'plain' | 'relaxed';
  // Work split of the threaded WASM path (R8 hc8 search; default DEFAULT_SPLIT of src/plan/wasm.ts).
  wasmSplit?: { rb?: number; cb?: number; qb?: number; minWork?: number };
  // Batch plans on the threaded WASM path (R8 hc9, default false: slower than single rows).
  wasmBatchPlans?: boolean;
}

export interface EncoderInput {
  inputIds: ArrayLike<number>;
  typeIds?: ArrayLike<number>; // default all 0
}

// Sequence classification, NLI, reranking: rows 1, cols classes. Token
// classification: rows seqLen, cols classes. Embeddings: rows 1, cols hidden
// (or the width after the last Dense step), before the L2 norm.
export interface EncoderOutput {
  data: Float32Array;
  rows: number;
  cols: number;
  seqLen: number;
  // capture runs only: [embedding, layer 0 .. N-1], captureSlotElements
  // f32 values each (bucket length x hidden)
  capture?: Float32Array;
  captureSlotElements?: number;
}

export interface RunOptions {
  // Force a loaded bucket instead of the smallest that fits.
  bucket?: number;
  // f32 parity tool: also return the layer states (single calls only).
  capture?: boolean;
}

// Text methods (K28.3). logits are the raw engine output for the class row.
export interface TextClassification {
  label: string;
  index: number;
  // probability of the label (softmax for single-label heads, sigmoid of its logit for
  // multi-label heads); the raw logit for regression heads (problem_type regression, or one class)
  score: number;
  // per class in class order: softmax probabilities, sigmoid probabilities (multi-label,
  // independent, they do not sum to 1) or the raw logits (regression)
  scores: Float32Array;
  logits: Float32Array;
}

export interface ZeroShotResult {
  label: string;
  index: number;
  // softmax over the entailment logits of all labels
  score: number;
  scores: Float32Array;
  // entailment logit per label, in the order of the labels argument
  logits: Float32Array;
}

export interface RerankResult {
  // raw logit per passage, in input order
  scores: number[];
  // passage indices, best first (ties keep the lower index first)
  order: number[];
}

export interface TokenSpan extends EntitySpan {
  text: string;
}

const RERANK_CHUNK = 16;

const DEFAULT_BUCKETS = [128, 512];
// Stride-quantized batch plans are built lazily per (stride, B); the cap
// evicts the oldest plan when exceeded.
const MAX_BATCH_PLANS = 8;

// The file is read as bytes so its size counts as download; the parse error
// (or the failed fetch, with no bytes) travels as text.
async function fetchTokenizer(url: string): Promise<{ bytes: number; tokenizer?: JsonTokenizer; error?: string }> {
  let buf: ArrayBuffer;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    buf = await res.arrayBuffer();
  } catch (e) {
    return { bytes: 0, error: String(e instanceof Error ? e.message : e) };
  }
  try {
    return { bytes: buf.byteLength, tokenizer: JsonTokenizer.fromString(new TextDecoder().decode(buf)) };
  } catch (e) {
    return { bytes: buf.byteLength, error: String(e instanceof Error ? e.message : e) };
  }
}

export class EncoderModel {
  private plans = new Map<number, PlanRunner>();
  private batchPlans = new Map<string, PlanRunner>();
  // `${stride}:${asked batch}` -> the batch size that fits the device (1: none, use the bucket plan)
  private batchFit = new Map<string, number>();
  private queue: Promise<unknown> = Promise.resolve();
  private downloadBytes = 0;
  private loadTiming: Record<string, number> = {};
  private tokenizer: JsonTokenizer | null = null;
  // Why the tokenizer is missing: a tokenizer.json with a component the text
  // path does not implement yet (ByteLevel, Unigram, Metaspace: K28.3b) must not
  // block the id path (runIds, runIdsBatch); the text methods rethrow it.
  private tokenizerError: string | null = null;
  private precisionNote: string | undefined;
  private recommendedPrecision: 'f32' | undefined;
  private labels: string[] = [];
  private textMaxLength = 0;

  private constructor(
    private backend: Backend,
    private embeddings: Float32Array | Uint16Array,
    readonly spec: ModelSpec,
    readonly head: HeadSpec,
    readonly task: string,
    readonly precision: 'f16' | 'f32',
  ) {}

  // deps.wasmTransport: the executor host of the WASM path without a worker (Node parity run).
  static async load(
    options: EncoderLoadOptions, deps: { wasmTransport?: () => Transport } = {},
  ): Promise<EncoderModel> {
    const wasm = options.backend === 'wasm';
    if (wasm && options.precision === 'f16') throw new Error('the WASM path runs the f32 manifest');
    const preferF16 = options.precision !== 'f32';
    const tMf = performance.now();
    const url = wasm ? f32ManifestUrl(options.manifestUrl) : options.manifestUrl;
    const manifestPromise = fetchManifest(url);
    const kh = wasm ? undefined : await getDevice(preferF16, options.limits !== 'default');
    const resolved = await resolveManifest(url, wasm ? 'f32' : options.precision, await manifestPromise);
    const manifest = resolved.manifest;
    const manifestUrl = resolved.url;
    const manifestMs = performance.now() - tMf;
    if (manifest.format !== 'kleinhirn-weights-2' || !manifest.spec) {
      throw new Error(`EncoderModel needs kleinhirn-weights-2, manifest is ${manifest.format}`);
    }
    // Every tensor of a kleinhirn-weights-2 manifest has the manifest's dtype. The first
    // tensor is the word table in all rows (ModernBERT has no layers.0.qkv).
    const dtype = manifest.tensors[0]?.dtype ?? 'f32';
    if (kh && dtype === 'f16' && !kh.hasF16) {
      throw new Error('f16 manifest but adapter lacks shader-f16; use the f32 manifest');
    }
    if (wasm && dtype !== 'f32') throw new Error(`the WASM path needs an f32 manifest, ${manifestUrl} holds ${dtype}`);
    if (options.precision && options.precision !== 'auto' && options.precision !== dtype) {
      throw new Error(`precision ${options.precision} requested but manifest is ${dtype}`);
    }
    const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
    const tokenizerPromise = manifest.tokenizer ? fetchTokenizer(base + manifest.tokenizer) : undefined;
    const loaded = await loadBackend(manifestUrl, manifest, kh, deps.wasmTransport, { build: options.wasmBuild, threads: options.threads, split: options.wasmSplit,
      batchPlans: options.wasmBatchPlans });
    const { backend, embeddings } = loaded;
    const spec = manifest.spec as ModelSpec;
    const model = new EncoderModel(
      backend, embeddings, spec, manifest.head as unknown as HeadSpec,
      manifest.task ?? '', dtype as 'f16' | 'f32');
    model.precisionNote = resolved.note;
    model.recommendedPrecision = manifest.recommendedPrecision;
    const tok = await tokenizerPromise;
    if (tok?.error !== undefined) model.tokenizerError = tok.error;
    else if (tok) model.tokenizer = tok.tokenizer as JsonTokenizer;
    const head = model.head;
    model.labels = head.type === 'classify' || head.type === 'token'
      ? completeLabels(manifest.labels, head.classes)
      : Object.entries(manifest.labels ?? {})
        .sort((a, b) => Number(a[0]) - Number(b[0])).map(([, name]) => name);
    const st = manifest.sentenceTransformers as { maxSeqLength?: number } | undefined;
    const largest = Math.max(...(options.buckets ?? DEFAULT_BUCKETS));
    // The manifest's maxLength (K28.5) already holds the position rows after the
    // offset, the tokenizer's model_max_length and the max_seq_length; manifests
    // of K28.2 to K28.4 lack it and use the rule those versions wrote.
    model.textMaxLength = options.maxLength ?? Math.min(
      manifest.maxLength ?? Math.min(
        spec.embed.maxPositions - spec.embed.positionOffset, st?.maxSeqLength ?? Infinity),
      largest);
    model.downloadBytes = loaded.downloadBytes + (tok?.bytes ?? 0);
    model.loadTiming = { ...loaded.timing, manifestMs, planMs: 0 };
    const tPlan = performance.now();
    await backend.prepare(() => {
      for (const length of options.buckets ?? DEFAULT_BUCKETS) {
        const bucket = model.planFor(length, 1);
        backend.check(bucket);
        model.plans.set(length, backend.runner(bucket));
      }
    });
    model.loadTiming.planMs = performance.now() - tPlan;
    return model;
  }

  private planFor(length: number, batch: number): Plan {
    return buildPlan(this.spec, this.head, {
      length, batch, markers: 0, f16: this.precision === 'f16' });
  }

  private makePlan(length: number, batch: number): PlanRunner {
    return this.backend.runner(this.planFor(length, batch));
  }

  private pickBucket(seqLen: number): PlanRunner {
    const fits = [...this.plans.values()]
      .filter((p) => seqLen <= p.length)
      .sort((a, b) => a.length - b.length);
    if (!fits.length) {
      throw new BucketOverflowError(`seqLen ${seqLen} exceeds the loaded buckets`);
    }
    return fits[0];
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch(() => {});
    return run;
  }

  // Word rows of sequence b at row b * length, width embeddingSize; rows
  // past a sequence's end stay zero.
  private wordRows(
    inputs: EncoderInput[], length: number,
  ): Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer> {
    const width = this.spec.embeddingSize;
    const emb = this.embeddings;
    const out = emb instanceof Float32Array
      ? new Float32Array(length * inputs.length * width)
      : new Uint16Array(length * inputs.length * width);
    for (const [b, input] of inputs.entries()) {
      for (let i = 0; i < input.inputIds.length; i += 1) {
        const off = tokenRowOffset(input.inputIds[i], width, emb.length);
        out.set(emb.subarray(off, off + width), (b * length + i) * width);
      }
    }
    return out;
  }

  private maskAndTypes(
    inputs: EncoderInput[], length: number,
  ): { mask: Float32Array<ArrayBuffer>; typeIds: Uint32Array<ArrayBuffer> } {
    const mask = new Float32Array(length * inputs.length);
    const typeIds = new Uint32Array(length * inputs.length);
    for (const [b, input] of inputs.entries()) {
      mask.fill(1, b * length, b * length + input.inputIds.length);
      // DistilBERT has no type table: type ids are ignored, as in transformers.
      if (input.typeIds && this.spec.embed.typeVocab > 0) {
        for (let i = 0; i < input.inputIds.length; i += 1) {
          const t = input.typeIds[i] ?? 0;
          if (!Number.isInteger(t) || t < 0 || t >= this.spec.embed.typeVocab) {
            throw new Error(`token type ${t} is not an integer inside the ${this.spec.embed.typeVocab} type rows`);
          }
          typeIds[b * length + i] = t;
        }
      }
      // RoBERTa and XLM-R: the position row of each id in the high 16 bits (embln POSIDS, review
      // R09); rows past the input keep the pad row.
      const { padId, maxPositions } = this.spec.embed;
      if (padId !== undefined) {
        const rows = positionRows(padId, input.inputIds);
        for (let i = 0; i < length; i += 1) {
          const p = i < rows.length ? rows[i] : padId;
          if (p >= maxPositions || p > 0xffff) throw new Error(`position row ${p} of id ${i} is outside the ${maxPositions} position rows`);
          typeIds[b * length + i] |= p << 16;
        }
      }
    }
    return { mask, typeIds };
  }

  // Output columns: classes, or the embedding width after the last Dense step (a Norm step
  // keeps the width of the step before it, as in the plan's output.cols).
  private get cols(): number {
    const h = this.head;
    if (h.type === 'classify' || h.type === 'token') return h.classes;
    if (h.type === 'embed') {
      let width = this.spec.hidden;
      for (const step of h.steps) if (step.op === 'dense') width = step.out;
      return width;
    }
    throw new Error(`head ${h.type} is not an encoder head`);
  }

  // Sequence b of a call: its slice of the plan output.
  private shape(data: Float32Array, b: number, seqLen: number, plan: PlanRunner): EncoderOutput {
    const cols = this.cols;
    if (this.head.type === 'token') {
      const start = b * plan.length * cols;
      return { data: data.slice(start, start + seqLen * cols), rows: seqLen, cols, seqLen };
    }
    return { data: data.slice(b * cols, (b + 1) * cols), rows: 1, cols, seqLen };
  }

  // A bucket only sizes buffers; an input must still fit the position table.
  private checkPositions(seqLen: number): void {
    const { maxPositions, positionOffset } = this.spec.embed;
    if (seqLen > maxPositions - positionOffset) {
      throw new Error(`input of ${seqLen} tokens exceeds the ${maxPositions} position rows `
        + `minus offset ${positionOffset} (${maxPositions - positionOffset} tokens)`);
    }
  }

  async runIds(input: EncoderInput, options: RunOptions = {}): Promise<EncoderOutput> {
    const seqLen = input.inputIds.length;
    if (seqLen === 0) throw new Error('empty input');
    this.checkPositions(seqLen);
    const plan = options.bucket === undefined
      ? this.pickBucket(seqLen) : this.plans.get(options.bucket);
    if (!plan) throw new Error(`bucket ${options.bucket} not loaded`);
    if (seqLen > plan.length) {
      throw new BucketOverflowError(`seqLen ${seqLen} exceeds bucket ${plan.length}`);
    }
    if (options.capture) plan.assertCapturable();
    return this.enqueue(() => this.backend.call(() => {
      const { mask, typeIds } = this.maskAndTypes([input], plan.length);
      plan.upload({ embeddings: this.wordRows([input], plan.length), mask, typeIds });
      plan.run(seqLen, !!options.capture);
    }, async () => {
      const data = await plan.readOutput();
      const result = this.shape(data, 0, seqLen, plan);
      assertFinite(result.data, 'engine output');
      if (options.capture) {
        result.capture = await plan.readCapture();
        result.captureSlotElements = plan.length * this.spec.hidden;
      }
      return result;
    }));
  }

  // Lazily built batch plan for row stride and batch size B: the plan of the
  // largest size up to B that fits the device limits (checkPlan), or
  // undefined when none does and the rows run on the bucket plan. Plans are capped.
  private batchPlan(stride: number, batch: number): PlanRunner | undefined {
    const asked = `${stride}:${batch}`;
    const memo = this.batchFit.get(asked);
    if (memo === 1) return undefined;
    const known = this.batchPlans.get(`${stride}:${memo ?? batch}`);
    if (known) return known;
    const plan = memo === undefined
      ? this.backend.fitBatch((b) => this.planFor(stride, b), batch)
      : this.planFor(stride, memo);
    this.batchFit.set(asked, plan ? plan.batch : 1);
    if (!plan) return undefined;
    if (this.batchPlans.size >= MAX_BATCH_PLANS) {
      const oldest = this.batchPlans.keys().next().value as string;
      this.batchPlans.get(oldest)?.destroy();
      this.batchPlans.delete(oldest);
    }
    const p = this.backend.runner(plan);
    this.batchPlans.set(`${stride}:${plan.batch}`, p);
    return p;
  }

  // One chunk of up to MAX_BATCH rows. Choosing, building and evicting the
  // batch plan happen inside the queued function, next to upload, submit and
  // readback: a plan chosen for a call cannot be evicted before that call runs.
  // The plan that fits the device may be smaller than the chunk (or the bucket
  // plan, B1): the rows then run in pieces of that plan's batch size, one after
  // the other in the same queued function (a nested enqueue would wait on itself).
  private async runChunk(inputs: EncoderInput[], bucket?: number): Promise<EncoderOutput[]> {
    const maxSeq = Math.max(...inputs.map((i) => i.inputIds.length));
    if (maxSeq === 0) throw new Error('empty input');
    this.checkPositions(maxSeq);
    const plan = bucket === undefined ? this.pickBucket(maxSeq) : this.plans.get(bucket);
    if (!plan) throw new Error(`bucket ${bucket} not loaded`);
    if (maxSeq > plan.length) {
      throw new BucketOverflowError(`batch exceeds bucket ${plan.length}`);
    }
    const n = nextBatchSize(inputs.length);
    // B=1 runs on the bucket plan like runIds and dispatches only the real
    // rows; B>1 packs at the quantized stride.
    const stride = Math.min(plan.length, batchStride(maxSeq));
    return this.enqueue(async () => {
      const out: EncoderOutput[] = [];
      let bPlan: PlanRunner | undefined;
      while (out.length < inputs.length) {
        const piece = await this.backend.call(() => {
          bPlan ??= (n === 1 ? undefined : this.batchPlan(stride, n)) ?? plan;
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
  private submitPiece(inputs: EncoderInput[], bPlan: PlanRunner): void {
    // Zero-length rows pad up to the batch size; their outputs are dropped.
    const padded = [...inputs, ...Array.from(
      { length: bPlan.batch - inputs.length }, () => ({ inputIds: [] as number[] }))];
    const rows = bPlan.batch === 1
      ? Math.max(...inputs.map((i) => i.inputIds.length)) : bPlan.length * bPlan.batch;
    const { mask, typeIds } = this.maskAndTypes(padded, bPlan.length);
    bPlan.upload({ embeddings: this.wordRows(padded, bPlan.length), mask, typeIds });
    bPlan.run(rows);
  }

  private async readPiece(inputs: EncoderInput[], bPlan: PlanRunner): Promise<EncoderOutput[]> {
    const all = await bPlan.readOutput();
    return inputs.map((input, i) => {
      const result = this.shape(all, i, input.inputIds.length, bPlan);
      assertFinite(result.data, 'engine output');
      return result;
    });
  }

  // Batch call: results in input order. Rows are sorted by length and run
  // in chunks of at most 16, padded up to the next batch size in {1, 4, 8, 16}.
  async runIdsBatch(inputs: EncoderInput[], options: { bucket?: number } = {}): Promise<EncoderOutput[]> {
    for (const [i, input] of inputs.entries()) {
      if (input.inputIds.length === 0) throw new Error(`empty input (input ${i})`);
    }
    const order = inputs.map((_, i) => i)
      .sort((a, b) => inputs[a].inputIds.length - inputs[b].inputIds.length || a - b);
    const results: EncoderOutput[] = new Array(inputs.length);
    for (let start = 0; start < order.length; start += MAX_BATCH) {
      const idx = order.slice(start, start + MAX_BATCH);
      const out = await this.runChunk(idx.map((i) => inputs[i]), options.bucket);
      idx.forEach((i, k) => { results[i] = out[k]; });
    }
    return results;
  }

  private requireTokenizer(): JsonTokenizer {
    if (this.tokenizerError) throw new Error(`text input unavailable: ${this.tokenizerError}`);
    if (!this.tokenizer) throw new Error('this model has no tokenizer.json in its manifest');
    return this.tokenizer;
  }

  private textInput(
    text: string, pair: string | null, truncation: TruncationStrategy = 'longest_first',
  ): EncoderInput {
    const e = this.requireTokenizer().encode(text, pair, { maxLength: this.textMaxLength, truncation });
    return { inputIds: e.ids, typeIds: e.typeIds };
  }

  private requireHead(...types: string[]): void {
    if (!types.includes(this.head.type)) {
      throw new Error(`needs a ${types.join(' or ')} head, this model has ${this.head.type}`);
    }
  }

  // Sequence classification, also NLI and reranking with one text pair. The score
  // semantics follow the head's problem: single-label softmax, multi-label sigmoid per
  // logit (the label is the largest logit), regression the raw logits.
  async classify(text: string, pair?: string): Promise<TextClassification> {
    this.requireHead('classify');
    const out = await this.runIds(this.textInput(text, pair ?? null));
    const logits = out.data;
    const head = this.head as Extract<HeadSpec, { type: 'classify' }>;
    const problem = head.problem ?? (head.classes === 1 ? 'regression' : 'single');
    if (problem === 'regression') {
      const index = argmaxRows(logits, logits.length)[0];
      return { label: this.labels[index] ?? String(index), index, score: logits[index], scores: logits, logits };
    }
    let scores: Float32Array<ArrayBuffer>;
    if (problem === 'multi') scores = Float32Array.from(logits, (v) => 1 / (1 + Math.exp(-v)));
    else scores = softmax(logits);
    const index = argmaxRows(problem === 'multi' ? logits : scores, scores.length)[0];
    return { label: this.labels[index] ?? String(index), index, score: scores[index], scores, logits };
  }

  // NLI zero-shot: one (text, hypothesis) pair per label in one batch call,
  // softmax over the entailment logits.
  async zeroShot(
    text: string, labels: string[], options: { template?: string } = {},
  ): Promise<ZeroShotResult> {
    this.requireHead('classify');
    const entail = this.labels.findIndex((l) => l.toLowerCase().startsWith('entail'));
    if (entail < 0) throw new Error(`no entailment label in [${this.labels.join(', ')}]`);
    if (!labels.length) throw new Error('zeroShot needs at least one label');
    const template = options.template ?? 'This example is {}.';
    const outs = await this.runIdsBatch(labels.map(
      (label) => this.textInput(text, template.replace('{}', () => label), 'only_first')));
    const logits = Float32Array.from(outs, (o) => o.data[entail]);
    const scores = softmax(logits);
    const index = argmaxRows(scores, scores.length)[0];
    return { label: labels[index], index, score: scores[index], scores, logits };
  }

  // Reranker with one logit per pair; batches of up to 16 pairs.
  async rerank(query: string, passages: string[]): Promise<RerankResult> {
    this.requireHead('classify');
    if (this.cols !== 1) throw new Error(`rerank needs a one-logit head, this model has ${this.cols}`);
    const scores: number[] = [];
    for (let i = 0; i < passages.length; i += RERANK_CHUNK) {
      const outs = await this.runIdsBatch(passages.slice(i, i + RERANK_CHUNK)
        .map((p) => this.textInput(query, p)));
      for (const o of outs) scores.push(o.data[0]);
    }
    const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
    return { scores, order };
  }

  // Sentence embeddings, truncated to the model's max_seq_length, L2 norm in
  // JS when the manifest says normalize.
  async embed(texts: string[], options: { prompt?: string } = {}): Promise<Float32Array[]> {
    this.requireHead('embed');
    const prompt = options.prompt ?? '';
    const outs = await this.runIdsBatch(texts.map((t) => this.textInput(prompt + t, null)));
    const normalize = (this.head as { normalize?: boolean }).normalize === true;
    return outs.map((o) => (normalize ? l2normalize(o.data) : o.data));
  }

  // Entity spans of a token classification model (aggregation "simple").
  // Softmax per token whatever problem_type says, like the HF
  // token-classification pipeline; multi-label token models (Setur/BRAGD)
  // decode their own features from runIds.
  async tokenClassify(text: string): Promise<TokenSpan[]> {
    this.requireHead('token');
    const e = this.requireTokenizer().encode(text, null, { maxLength: this.textMaxLength });
    const out = await this.runIds({ inputIds: e.ids, typeIds: e.typeIds });
    const cols = out.cols;
    const probs = new Float32Array(out.rows * cols);
    for (let r = 0; r < out.rows; r += 1) {
      probs.set(softmax(out.data.subarray(r * cols, (r + 1) * cols)), r * cols);
    }
    return aggregateSimple(probs, this.labels, e.offsets, e.specialTokensMask)
      .map((g) => ({ ...g, text: text.slice(g.start, g.end) }));
  }

  // Weights plus every live plan (a capture buffer counts once it exists);
  // an evicted batch plan is gone from the map and from the sum.
  private liveGpuBytes(): number {
    let n = this.backend.weightBytes;
    for (const p of this.plans.values()) n += p.bytes;
    for (const p of this.batchPlans.values()) n += p.bytes;
    return n;
  }

  info(): Record<string, unknown> {
    return {
      precision: this.precision,
      ...(this.recommendedPrecision ? { recommendedPrecision: this.recommendedPrecision } : {}),
      ...(this.precisionNote ? { precisionNote: this.precisionNote } : {}),
      buildId: typeof __KH_BUILD_ID__ === 'string' ? __KH_BUILD_ID__ : 'dev',
      ...this.backend.info(),
      family: this.spec.family,
      task: this.task,
      buckets: [...this.plans.keys()].sort((a, b) => a - b),
      textMaxLength: this.textMaxLength,
      hasTokenizer: this.tokenizer !== null,
      batchSizes: [...BATCH_SIZES],
      gpuBytes: this.backend.kind === 'webgpu' ? this.liveGpuBytes() : null,
      downloadBytes: this.downloadBytes,
      loadTiming: this.loadTiming,
    };
  }

  dispose(): void {
    this.backend.dispose();
  }
}
