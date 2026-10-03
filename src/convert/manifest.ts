/// <reference lib="esnext.float16" />
// Manifest and shard writer of the converter core (format
// kleinhirn-weights-2). Shard limit, alignment and the row chunks of the word
// embedding table follow convert/export_weights.py. Pure functions over
// tensors; file input and output live in the CLI, so the same code runs in
// the browser (K28 design section 7, variant B).

import type { HeadSpec, ModelSpec } from '../plan/spec.ts';
import type { NamePlan, RelProjection } from './names.ts';
import { elementCount, readF32, type StFile } from './safetensors.ts';

export const SHARD_LIMIT = 95_000_000;
export const ALIGN = 256;
export const EMBED_CHUNK_ROWS = 60_000;
export const FORMAT = 'kleinhirn-weights-2';

export interface NamedTensor {
  name: string;
  shape: number[];
  data: Float32Array;
}

export interface ManifestMeta {
  source: { repo: string; revision: string; checkpointSha256: string };
  spec: ModelSpec;
  head: HeadSpec;
  task: string;
  labels: Record<string, string>;
  tokenizer: string;
  // Longest token sequence of the model (see manifestMaxLength in src/plan/hf.ts).
  maxLength: number;
  sentenceTransformers?: unknown;
}

export interface TensorEntry {
  name: string;
  dtype: 'f32' | 'f16';
  shape: number[];
  shard: number;
  offset: number;
  byteLength: number;
  keepOnCpu?: boolean;
  rowStart?: number;
  rowEnd?: number;
}

export interface BuiltManifest {
  manifest: Record<string, unknown>;
  shards: { file: string; bytes: Uint8Array }[];
}

// The sinusoidal position table of DistilBERT (transformers 5.0.0,
// _create_sinusoidal_embeddings): angle in float64, sin on even and cos on odd
// columns, rounded once to float32.
export function sinusoidalTable(rows: number, dim: number): Float32Array {
  const out = new Float32Array(rows * dim);
  for (let pos = 0; pos < rows; pos += 1) {
    for (let j = 0; j < dim; j += 1) {
      const angle = pos / 10000 ** ((2 * Math.floor(j / 2)) / dim);
      out[pos * dim + j] = j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return out;
}

// DeBERTa relative attention with share_att_key: the position projections of one layer are the
// key or query projection applied to LayerNorm(rel_embeddings). float64 throughout, rounded once
// to float32 (K28.6: the f16 manifest then casts that float32 value like any other tensor).
export function relProjection(file: StFile, d: RelProjection, cache: Map<string, Float64Array>): Float32Array {
  const relInfo = file.tensors.get(d.rel);
  const wInfo = file.tensors.get(d.projW);
  if (!relInfo || !wInfo) throw new Error(`relative projection: ${d.rel} or ${d.projW} is missing`);
  const [rows, width] = [d.rows, relInfo.shape[1]];
  const [out, inp] = wInfo.shape;
  if (relInfo.shape[0] < rows || inp !== width) {
    throw new Error(`relative projection: rel_embeddings ${relInfo.shape.join('x')}, ${d.projW} ${wInfo.shape.join('x')}, ${rows} rows needed`);
  }
  const key = `${d.rel}|${d.relNormW}|${d.eps}`;
  let ln = cache.get(key);
  if (!ln) {
    const rel = readF32(file, d.rel);
    const gamma = readF32(file, d.relNormW);
    const beta = readF32(file, d.relNormB);
    ln = new Float64Array(rows * width);
    for (let r = 0; r < rows; r += 1) {
      let mean = 0;
      for (let c = 0; c < width; c += 1) mean += rel[r * width + c];
      mean /= width;
      let variance = 0;
      for (let c = 0; c < width; c += 1) variance += (rel[r * width + c] - mean) ** 2;
      variance /= width;
      const inv = 1 / Math.sqrt(variance + d.eps);
      for (let c = 0; c < width; c += 1) {
        ln[r * width + c] = (rel[r * width + c] - mean) * inv * gamma[c] + beta[c];
      }
    }
    cache.set(key, ln);
  }
  const w = readF32(file, d.projW);
  const b = readF32(file, d.projB);
  const result = new Float32Array(rows * out);
  for (let r = 0; r < rows; r += 1) {
    const x = ln.subarray(r * width, (r + 1) * width);
    for (let o = 0; o < out; o += 1) {
      let acc = b[o];
      const wr = o * inp;
      for (let c = 0; c < inp; c += 1) acc += x[c] * w[wr + c];
      result[r * out + o] = acc;
    }
  }
  return result;
}

// Conv1d weight [out, in, k] (row-major) -> [out, k * in]: element (c, t * in + ci) = W[c, ci, t].
export function conv1dRows(w: Float32Array, out: number, inp: number, k: number): Float32Array {
  const r = new Float32Array(w.length);
  for (let c = 0; c < out; c += 1) {
    for (let ci = 0; ci < inp; ci += 1) {
      for (let t = 0; t < k; t += 1) r[c * k * inp + t * inp + ci] = w[(c * inp + ci) * k + t];
    }
  }
  return r;
}

// Read the planned tensors from the checkpoint (or from the additional file a
// tensor names, Dense modules): fuse Q, K and V by rows, check every shape
// against the description.
export function gatherTensors(
  file: StFile, plan: NamePlan, extraFiles: Record<string, StFile> = {},
): NamedTensor[] {
  const relCache = new Map<string, Float64Array>();
  return plan.tensors.map((t) => {
    if (t.derive) {
      const data = relProjection(file, t.derive, relCache);
      if (elementCount(t.shape) !== data.length) {
        throw new Error(`${t.name}: derived ${data.length} elements, expected ${t.shape.join('x')}`);
      }
      return { name: t.name, shape: t.shape, data };
    }
    if (t.generate === 'sinusoidal') {
      return { name: t.name, shape: t.shape, data: sinusoidalTable(t.shape[0], t.shape[1]) };
    }
    const src = t.file === undefined ? file : extraFiles[t.file];
    if (!src) throw new Error(`${t.name}: additional file ${t.file} was not given`);
    if (t.transform === 'conv1d') {
      const info = src.tensors.get(t.sources[0])!;
      // One group only: the source is exactly [H, H, k], the flat shape [H, k * H] cannot tell
      // the input width from the kernel width.
      const H = t.shape[0];
      const k = t.shape[1] / H;
      if (!Number.isInteger(k) || info.shape.length !== 3
        || info.shape[0] !== H || info.shape[1] !== H || info.shape[2] !== k) {
        throw new Error(`${t.name}: ${t.sources[0]} has shape ${info.shape.join('x')}, expected ${H}x${H}x${k}`);
      }
      const [out, inp] = info.shape;
      return { name: t.name, shape: t.shape, data: conv1dRows(readF32(src, t.sources[0]), out, inp, k) };
    }
    const parts = t.sources.map((s) => ({ s, shape: src.tensors.get(s)!.shape, data: readF32(src, s) }));
    const total = parts.reduce((n, p) => n + p.data.length, 0);
    const data = new Float32Array(total);
    let off = 0;
    for (const p of parts) {
      data.set(p.data, off);
      off += p.data.length;
    }
    if (elementCount(t.shape) !== total || parts.some((p) => p.shape.length !== t.shape.length)) {
      throw new Error(`${t.name}: checkpoint shapes [${parts.map((p) => p.shape.join('x')).join(' + ')}] do not fit ${t.shape.join('x')}`);
    }
    const cols = t.shape.slice(1);
    for (const p of parts) {
      if (p.shape.slice(1).join() !== cols.join()) {
        throw new Error(`${t.name}: source ${p.s} has shape ${p.shape.join('x')}, expected ${t.shape.join('x')}`);
      }
    }
    return { name: t.name, shape: t.shape, data };
  });
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as BufferSource);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// A finite weight that f16 cannot hold (|w| above 65504, rounded to Inf): the
// model needs the f32 manifest. Separate from a non-finite source value, which
// no manifest can carry.
export class F16RangeError extends Error {}

// Rejects what a run could not tell from a model: NaN or Inf in the source
// (both dtypes), and in f16 any finite value that overflows to Inf.
function castBytes(name: string, data: Float32Array, dtype: 'f32' | 'f16'): Uint8Array {
  for (let i = 0; i < data.length; i += 1) {
    if (!Number.isFinite(data[i])) throw new Error(`${name}: source value ${data[i]} at index ${i} is not finite`);
  }
  const arr = dtype === 'f16' ? new Float16Array(data) : data;
  if (dtype === 'f16') {
    for (let i = 0; i < arr.length; i += 1) {
      if (!Number.isFinite(arr[i])) {
        throw new F16RangeError(`${name}: ${data[i]} at index ${i} overflows f16 (Inf); use the f32 manifest`);
      }
    }
  }
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

export async function buildManifest(
  tensors: NamedTensor[], dtype: 'f32' | 'f16', meta: ManifestMeta,
): Promise<BuiltManifest> {
  const entries: TensorEntry[] = [];
  const shardBytes: Uint8Array[] = [];
  let parts: Uint8Array[] = [];
  let length = 0;

  const flush = (): void => {
    if (length === 0) return;
    const payload = new Uint8Array(length);
    let off = 0;
    for (const p of parts) {
      payload.set(p, off);
      off += p.length;
    }
    shardBytes.push(payload);
    parts = [];
    length = 0;
  };
  const pad = (): void => {
    const rest = length % ALIGN;
    if (rest === 0) return;
    parts.push(new Uint8Array(ALIGN - rest));
    length += ALIGN - rest;
  };
  const write = (name: string, shape: number[], data: Float32Array, rows?: [number, number]): void => {
    const raw = castBytes(name, data, dtype);
    pad();
    if (length + raw.length > SHARD_LIMIT) {
      flush();
      pad();
    }
    const entry: TensorEntry = {
      name, dtype, shape, shard: shardBytes.length, offset: length, byteLength: raw.length,
    };
    if (rows) {
      entry.keepOnCpu = true;
      entry.rowStart = rows[0];
      entry.rowEnd = rows[1];
    }
    entries.push(entry);
    parts.push(raw);
    length += raw.length;
  };

  for (const t of tensors) {
    if (t.name !== 'embeddings.word.weight') {
      write(t.name, t.shape, t.data);
      continue;
    }
    // The table alone can exceed the shard limit: row chunks, joined by rowStart.
    const [rows, width] = [t.shape[0], t.shape[1]];
    for (let start = 0; start < rows; start += EMBED_CHUNK_ROWS) {
      const end = Math.min(start + EMBED_CHUNK_ROWS, rows);
      write(t.name, [end - start, width], t.data.subarray(start * width, end * width), [start, end]);
    }
  }
  flush();

  const shards = await Promise.all(shardBytes.map(async (bytes, i) => ({
    file: `weights-${i}.bin`, bytes,
    meta: { file: `weights-${i}.bin`, bytes: bytes.length, sha256: await sha256Hex(bytes) },
  })));
  const manifest: Record<string, unknown> = {
    format: FORMAT, version: 2, source: meta.source,
    encoder: {}, // empty for kleinhirn-weights-2; src/weights.ts only needs the key
    spec: meta.spec, head: meta.head, task: meta.task, labels: meta.labels,
    tokenizer: meta.tokenizer, maxLength: meta.maxLength,
  };
  if (meta.sentenceTransformers !== undefined) manifest.sentenceTransformers = meta.sentenceTransformers;
  manifest.tensors = entries;
  manifest.shards = shards.map((s) => s.meta);
  return { manifest, shards: shards.map((s) => ({ file: s.file, bytes: s.bytes })) };
}
