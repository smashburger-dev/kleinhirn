// WASM-SIMD fallback engine: same graph as the WebGPU path through the
// AssemblyScript module in src/wasm/deberta.wasm (f32 weights only).
// Used directly by the parity and bench runners; index.ts hosts it in a
// Worker for the automatic fallback chain (docs/PLAN.md K5).

import { fetchManifest, fetchShardBytes, type Manifest } from './weights.ts';
import { relPosTable } from './plan/build.ts';
import type { EncoderSpec } from './plan/spec.ts';
import { HfTokenizer } from './tokenizer/tokenizer.ts';
import {
  BucketOverflowError, prepareTasks, type SchemaInput,
} from './tokenizer/schema.ts';
import type { ClassifyResult, LoadOptions, PreparedResult } from './index.ts';
import { LruCache, dedupMerge, schemaMergeKey } from './cache.ts';
import wasmUrl from './wasm/deberta.wasm?url';

const K_MAX = 16;
const HEAD_HID = 768;
const BUCKETS = [128, 256, 512, 1024];
const MAX_BUCKET = BUCKETS[BUCKETS.length - 1];

export interface WasmExports {
  memory: WebAssembly.Memory;
  allocWeights(bytes: number): number;
  init(bucketLen: number, eps: number, temp: number): void;
  setShared(eW: number, eB: number, f1w: number, f1b: number, f2w: number, f2b: number): void;
  setLayer(
    l: number,
    qW: number, qB: number, pK: number, pQ: number,
    aW: number, aB: number, aLw: number, aLb: number,
    iW: number, iB: number, oW: number, oB: number,
    fLw: number, fLb: number,
  ): void;
  embPtr(): number;
  maskPtr(): number;
  packedPtr(): number;
  relidxPtr(): number;
  logitsPtr(): number;
  forward(seqLen: number, L: number): void;
}

// Same group softmax as index.ts, kept local so the worker chunk never
// pulls the WebGPU modules.
function groupSoftmax(
  logits: Float32Array, groups: Int32Array, mask: Float32Array,
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

export class WasmKleinhirn {
  private ex!: WasmExports;
  private spec!: EncoderSpec;
  private embeddings!: Float32Array;
  private loadTiming: Record<string, number> = {};
  private downloadBytes = 0;
  private cache = new LruCache<PreparedResult>(0);

  private constructor(
    private manifest: Manifest,
    private tokenizer: HfTokenizer,
  ) {}

  static async load(options: LoadOptions): Promise<WasmKleinhirn> {
    // f32 weights only: an f16 manifest URL maps to its f32 sibling.
    const manifestUrl = options.manifestUrl.replace(/\/f16\//, '/f32/');
    const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
    const t0 = performance.now();
    const labeled = async <T>(name: string, p: Promise<T>): Promise<T> => {
      try { return await p; } catch (e) {
        throw new Error(`wasm load ${name}: ${e instanceof Error ? e.message : e}`);
      }
    };
    const manifestP = labeled('manifest', fetchManifest(manifestUrl));
    const wasmP = labeled('wasm-bytes', fetch(wasmUrl).then((r) => r.arrayBuffer()));
    const manifest = await manifestP;
    const [fetched, wasmBytes, tokBuf] = await Promise.all([
      labeled('shards', fetchShardBytes(manifestUrl, manifest)),
      wasmP,
      labeled('tokenizer-fetch',
        (async () => (await fetch(base + manifest.tokenizer)).arrayBuffer())()),
    ]);
    const tTok = performance.now();
    let tokenizer: HfTokenizer;
    try {
      tokenizer = new HfTokenizer(JSON.parse(new TextDecoder().decode(tokBuf)));
    } catch (e) {
      throw new Error(
        `wasm load tokenizer-parse (${tokBuf.byteLength} B):`
        + ` ${e instanceof Error ? e.message : e}`);
    }
    const tokenizerMs = performance.now() - tTok;
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: {
        abort: (msg: number, file: number, line: number, col: number) => {
          throw new Error(`wasm abort at ${file}:${line}:${col} (msg ${msg})`);
        },
      },
    });
    const ex = instance.exports as unknown as WasmExports;
    const spec = manifest.encoder as unknown as EncoderSpec;
    const hidden = spec.hiddenSize;

    const kh = new WasmKleinhirn(manifest, tokenizer);
    kh.ex = ex;
    kh.spec = spec;
    kh.downloadBytes = fetched.downloadBytes + tokBuf.byteLength + wasmBytes.byteLength;

    // Pack every tensor contiguously into the wasm weights arena, 16-byte
    // aligned; the host registers pointers via setShared/setLayer.
    const entries = manifest.tensors.filter((t) => t.name !== 'embeddings.word.weight');
    let total = 0;
    const offs = new Map<string, number>();
    for (const t of entries) {
      offs.set(t.name, total);
      total += (t.byteLength + 15) & ~15;
    }
    const tUp = performance.now();
    const wbase = ex.allocWeights(total);
    ex.init(MAX_BUCKET, spec.layerNormEps, manifest.head.temperature);
    const mem = () => ex.memory.buffer;
    // Matmul weights are stored transposed [K,N] so the SIMD lane runs along
    // output columns (deberta.ts documents the layout).
    const matmulWeight = (name: string) =>
      name === 'head.fc1.weight' || name === 'head.fc2.weight'
      || /^layers\.\d+\.(qkv|attn_out|ffn_in|ffn_out)\.weight$/.test(name);
    for (const t of entries) {
      const dst = new Uint8Array(mem(), wbase + offs.get(t.name)!, t.byteLength);
      const src = new Uint8Array(fetched.shardBytes[t.shard], t.offset, t.byteLength);
      if (matmulWeight(t.name)) {
        const [N, Kdim] = t.shape;
        const inF = new Float32Array(src.buffer, src.byteOffset, t.byteLength / 4);
        const outF = new Float32Array(dst.buffer, dst.byteOffset, t.byteLength / 4);
        for (let n = 0; n < N; n += 1) {
          for (let k = 0; k < Kdim; k += 1) {
            outF[k * N + n] = inF[n * Kdim + k];
          }
        }
      } else {
        dst.set(src);
      }
    }
    const at = (name: string) => wbase + offs.get(name)!;
    ex.setShared(
      at('embeddings.LayerNorm.weight'), at('embeddings.LayerNorm.bias'),
      at('head.fc1.weight'), at('head.fc1.bias'),
      at('head.fc2.weight'), at('head.fc2.bias'),
    );
    for (let l = 0; l < spec.layers; l += 1) {
      const p = `layers.${l}.`;
      ex.setLayer(
        l,
        at(`${p}qkv.weight`), at(`${p}qkv.bias`), at(`${p}pos_key`), at(`${p}pos_query`),
        at(`${p}attn_out.weight`), at(`${p}attn_out.bias`),
        at(`${p}attn_ln.weight`), at(`${p}attn_ln.bias`),
        at(`${p}ffn_in.weight`), at(`${p}ffn_in.bias`),
        at(`${p}ffn_out.weight`), at(`${p}ffn_out.bias`),
        at(`${p}ffn_ln.weight`), at(`${p}ffn_ln.bias`),
      );
    }
    new Uint32Array(mem(), ex.relidxPtr(), MAX_BUCKET * MAX_BUCKET).set(
      relPosTable(MAX_BUCKET, spec.positionBuckets, spec.maxRelativePositions));
    const uploadMs = performance.now() - tUp;

    // Embedding rows stay on the CPU (keepOnCpu parts joined in row order).
    const tJoin = performance.now();
    const embParts = manifest.tensors
      .filter((t) => t.keepOnCpu && t.name === 'embeddings.word.weight')
      .sort((a, b) => (a.rowStart ?? 0) - (b.rowStart ?? 0));
    let joined = 0;
    for (const t of embParts) joined += t.byteLength;
    const embBytes = new Uint8Array(joined);
    let off = 0;
    for (const t of embParts) {
      embBytes.set(new Uint8Array(fetched.shardBytes[t.shard], t.offset, t.byteLength), off);
      off += t.byteLength;
    }
    kh.embeddings = new Float32Array(embBytes.buffer);
    kh.cache = new LruCache(options.cacheSize ?? 256);
    const embedJoinMs = performance.now() - tJoin;

    kh.loadTiming = {
      loadMs: performance.now() - t0,
      fetchMs: fetched.fetchMs,
      bodyMs: fetched.bodyMs,
      sha256Ms: fetched.sha256Ms,
      uploadMs,
      embedJoinMs,
      tokenizerMs,
      planMs: 0,
    };
    return kh;
  }

  private pickBucket(seqLen: number): number {
    const fits = BUCKETS.find((b) => seqLen <= b);
    if (fits === undefined) {
      throw new BucketOverflowError(`seqLen ${seqLen} exceeds largest bucket`);
    }
    return fits;
  }

  private packedMarkers(input: SchemaInput): Uint32Array {
    const packed = new Uint32Array(3 * K_MAX);
    packed.set(input.markerIndices.subarray(0, K_MAX), 0);
    packed.set(new Uint32Array(input.markerMask.buffer).subarray(0, K_MAX), K_MAX);
    packed.set(input.markerGroups.subarray(0, K_MAX), 2 * K_MAX);
    return packed;
  }

  async runPrepared(input: SchemaInput, capture = false, bucket?: number): Promise<PreparedResult> {
    const L = bucket ?? this.pickBucket(input.seqLen);
    if (input.seqLen > L) {
      throw new BucketOverflowError(`seqLen ${input.seqLen} exceeds bucket ${L}`);
    }
    const key = schemaMergeKey(input);
    const hit = capture ? undefined : this.cache.get(key);
    if (hit) return { logits: hit.logits, probabilities: hit.probabilities };
    const ex = this.ex;
    const hidden = this.spec.hiddenSize;
    const embOut = new Float32Array(ex.memory.buffer, ex.embPtr(), MAX_BUCKET * hidden);
    for (let i = 0; i < input.seqLen; i += 1) {
      const off = input.inputIds[i] * hidden;
      embOut.set(this.embeddings.subarray(off, off + hidden), i * hidden);
    }
    const mask = new Float32Array(ex.memory.buffer, ex.maskPtr(), MAX_BUCKET);
    mask.fill(0);
    mask.fill(1, 0, input.seqLen);
    new Uint32Array(ex.memory.buffer, ex.packedPtr(), 3 * K_MAX)
      .set(this.packedMarkers(input));
    ex.forward(input.seqLen, L);
    const logits = new Float32Array(ex.memory.buffer, ex.logitsPtr(), K_MAX).slice();
    const probabilities = groupSoftmax(logits, input.markerGroups, input.markerMask);
    // Layer capture is f32 GPU parity tooling; the WASM gate needs logits only.
    void capture;
    this.cache.set(schemaMergeKey(input), { logits, probabilities });
    return { logits, probabilities };
  }

  // WASM batch (K16): loop over the single-input path; identical merge
  // keys inside one call still run only once (dedup + result cache).
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
    const uniqueResults: PreparedResult[] = [];
    for (const input of unique) {
      uniqueResults.push(await this.runPrepared(input, false, bucket));
    }
    for (const [i] of pending.entries()) {
      results[pendingIdx[i]] = uniqueResults[slot[i]];
    }
    return results as PreparedResult[];
  }

  async classify(
    text: string,
    tasks: { task: string; labels: string[] }[],
  ): Promise<ClassifyResult> {
    const t0 = performance.now();
    const tuples = tasks.map((t): [string, string[]] => [t.task, t.labels]);
    const input = prepareTasks(this.tokenizer, text, tuples, MAX_BUCKET, K_MAX);
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const { logits, probabilities } = await this.runPrepared(input);
    const gpuMs = performance.now() - g0;
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
      timings: { tokenizeMs, gpuMs, totalMs: performance.now() - t0 },
    };
  }

  // WASM batch API (K16): same shape as the WebGPU classifyBatch, realized
  // as a loop over the single path with merge-key dedup and the cache.
  async classifyBatch(
    items: { text: string; tasks: { task: string; labels: string[] }[] }[],
  ): Promise<ClassifyResult[]> {
    const t0 = performance.now();
    const prepared = items.map((item) => {
      const tuples = item.tasks.map((t): [string, string[]] => [t.task, t.labels]);
      return prepareTasks(this.tokenizer, item.text, tuples, MAX_BUCKET, K_MAX);
    });
    const tokenizeMs = performance.now() - t0;
    const g0 = performance.now();
    const outputs = await this.runPreparedBatch(prepared);
    const gpuMs = performance.now() - g0;
    const totalMs = performance.now() - t0;
    return items.map((item, i) => ({
      tasks: item.tasks.map(({ task, labels }, g) => ({
        task,
        labels: labels.map((label, li) => {
          const k = indexOfGroupMarker(prepared[i].markerGroups, g, li);
          return k < 0
            ? { label, probability: 0, logit: -1e4 }
            : {
              label,
              probability: outputs[i].probabilities[k],
              logit: outputs[i].logits[k],
            };
        }),
      })),
      timings: { tokenizeMs, gpuMs, totalMs },
    }));
  }

  info(): Record<string, unknown> {
    return {
      backend: 'wasm-simd',
      precision: 'f32',
      limitsMode: 'minimum',
      buckets: BUCKETS,
      downloadBytes: this.downloadBytes,
      weightBytes: this.downloadBytes,
      loadTiming: this.loadTiming,
    };
  }

  dispose(): void {
    // The wasm module is GC-owned; nothing to release explicitly.
  }
}
