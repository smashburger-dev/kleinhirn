// Manifest + shard loading: fetch, sha256 verification, GPU upload.
// Embedding rows marked keepOnCpu stay in a joined CPU array (f32 or f16
// bits); every other tensor becomes its own STORAGE buffer.

import { MINIMUM_LIMITS } from './device.ts';

export interface TensorDesc {
  name: string;
  dtype: string;
  shape: number[];
  shard: number;
  offset: number;
  byteLength: number;
  keepOnCpu?: boolean;
  rowStart?: number;
  rowEnd?: number;
}

export interface Manifest {
  format: string;
  source: { checkpointSha256?: string; repo?: string; revision?: string };
  encoder: Record<string, unknown>;
  head: { type: string; temperature: number; hiddenSize?: number };
  // kleinhirn-weights-2 (K28.2): `encoder` is empty, `head` is a HeadSpec
  // (no temperature) and the description travels in `spec`.
  spec?: unknown;
  task?: string;
  labels?: Record<string, string>;
  sentenceTransformers?: unknown;
  tokenizer?: string; // absent for models without text input (e.g. the judge)
  // kleinhirn-weights-2 (K28.5): longest token sequence of the model
  maxLength?: number;
  // K28.S3: the f16 manifest of a model whose f16 gate failed because of f16 storage (simulated)
  // names f32 here. EncoderModel.load with precision 'auto' then reads the f32 manifest beside it.
  recommendedPrecision?: 'f32';
  tensors: TensorDesc[];
  shards: { file: string; bytes: number; sha256: string }[];
}

export interface LoadedWeights {
  manifest: Manifest;
  tensors: Map<string, GPUBuffer>;
  embeddings: Float32Array | Uint16Array;
  downloadBytes: number;
  gpuBytes: number;
  timing: {
    fetchMs: number;
    bodyMs: number;
    sha256Ms: number;
    uploadMs: number;
    embedJoinMs: number;
  };
}

function hex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchBytes(url: string): Promise<{ buf: ArrayBuffer; bodyMs: number }> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  // res.arrayBuffer() covers body download + decode; the two cannot be
  // separated from JS, so the load decomposition reports them as one field.
  const t = performance.now();
  const buf = await res.arrayBuffer();
  return { buf, bodyMs: performance.now() - t };
}

export async function fetchManifest(manifestUrl: string): Promise<Manifest> {
  const res = await fetch(manifestUrl);
  if (!res.ok) throw new Error(`fetch ${manifestUrl}: ${res.status}`);
  return res.json() as Promise<Manifest>;
}

export interface ResolvedManifest {
  url: string;
  manifest: Manifest;
  // Set when the manifest recommends f32 and the f32 manifest could not be used: the caller reports it.
  note?: string;
}

// precision 'auto' (or none) with an f16 manifest that recommends f32: the f32 manifest in the sibling
// directory (.../f16/manifest.json -> .../f32/manifest.json), else the f16 manifest plus a note.
export async function resolveManifest(
  url: string, precision: 'auto' | 'f16' | 'f32' | undefined, first: Manifest,
  fetcher: (u: string) => Promise<Manifest> = fetchManifest,
): Promise<ResolvedManifest> {
  if ((precision ?? 'auto') !== 'auto' || first.recommendedPrecision !== 'f32') return { url, manifest: first };
  const alt = url.replace(/\/f16\/manifest\.json(\?.*)?$/, '/f32/manifest.json$1');
  if (alt === url) {
    return { url, manifest: first, note: 'manifest recommends f32, but its URL is not .../f16/manifest.json' };
  }
  try {
    return { url: alt, manifest: await fetcher(alt) };
  } catch (e) {
    return { url, manifest: first, note: `manifest recommends f32, f32 manifest not available: ${String(e instanceof Error ? e.message : e)}` };
  }
}

export interface FetchedShards {
  manifest: Manifest;
  shardBytes: ArrayBuffer[];
  downloadBytes: number;
  fetchMs: number;
  bodyMs: number;
  sha256Ms: number;
}

// Fetch every shard in parallel, verify each sha256 as it lands.
// fetchMs is the wall time of the fetch+hash phase, sha256Ms the summed
// hashing time inside it. Shared by the GPU loader and the WASM backend,
// which consumes the same bytes on the CPU.
export async function fetchShardBytes(
  manifestUrl: string, manifest?: Manifest,
): Promise<FetchedShards> {
  const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
  const mf = manifest ?? await fetchManifest(manifestUrl);
  let downloadBytes = (new TextEncoder().encode(JSON.stringify(mf))).byteLength;
  let sha256Ms = 0;
  let bodyMs = 0;
  const tFetch = performance.now();
  const shardBytes = await Promise.all(mf.shards.map(async (shard) => {
    const { buf, bodyMs: d } = await fetchBytes(base + shard.file);
    bodyMs += d;
    const t1 = performance.now();
    const digest = hex(await crypto.subtle.digest('SHA-256', buf));
    sha256Ms += performance.now() - t1;
    if (digest !== shard.sha256) {
      throw new Error(`sha256 mismatch on ${shard.file}`);
    }
    downloadBytes += buf.byteLength;
    return buf;
  }));
  const fetchMs = performance.now() - tFetch;
  return { manifest: mf, shardBytes, downloadBytes, fetchMs, bodyMs, sha256Ms };
}

export async function loadWeights(
  device: GPUDevice,
  manifestUrl: string,
  manifest?: Manifest,
): Promise<LoadedWeights> {
  const { manifest: mf, shardBytes, downloadBytes, fetchMs, bodyMs, sha256Ms } =
    await fetchShardBytes(manifestUrl, manifest);

  // Every GPU tensor against maxBufferSize before the first allocation.
  const maxBuffer = device.limits?.maxBufferSize ?? MINIMUM_LIMITS.maxBufferSize;
  for (const t of mf.tensors) {
    const padded = Math.ceil(t.byteLength / 4) * 4;
    if (!t.keepOnCpu && padded > maxBuffer) {
      throw new Error(`tensor ${t.name} needs ${padded} B, maxBufferSize is ${maxBuffer}`);
    }
  }

  const tensors = new Map<string, GPUBuffer>();
  const cpuParts = new Map<string, { parts: Uint8Array[]; dtype: string }>();
  let gpuBytes = 0;
  const tUp = performance.now();
  for (const t of mf.tensors) {
    const bytes = new Uint8Array(shardBytes[t.shard], t.offset, t.byteLength);
    if (t.keepOnCpu) {
      let entry = cpuParts.get(t.name);
      if (!entry) {
        entry = { parts: [], dtype: t.dtype };
        cpuParts.set(t.name, entry);
      }
      entry.parts.push(bytes.slice(0));
      continue;
    }
    const padded = Math.ceil(t.byteLength / 4) * 4; // writeBuffer needs %4
    const buf = device.createBuffer({
      size: padded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        | GPUBufferUsage.COPY_SRC,
    });
    if (t.byteLength % 4 === 0) {
      device.queue.writeBuffer(buf, 0, bytes);
    } else {
      const pad = new Uint8Array(padded);
      pad.set(bytes);
      device.queue.writeBuffer(buf, 0, pad);
    }
    gpuBytes += padded;
    tensors.set(t.name, buf);
  }
  const uploadMs = performance.now() - tUp;

  const emb = cpuParts.get('embeddings.word.weight');
  const tJoin = performance.now();
  let embeddings: Float32Array | Uint16Array = new Float32Array(0);
  if (emb) {
    const total = emb.parts.reduce((n, p) => n + p.byteLength, 0);
    const joined = new Uint8Array(total);
    let off = 0;
    for (const p of emb.parts) {
      joined.set(p, off);
      off += p.byteLength;
    }
    embeddings = emb.dtype === 'f16'
      ? new Uint16Array(joined.buffer)
      : new Float32Array(joined.buffer);
  }
  const embedJoinMs = performance.now() - tJoin;

  return {
    manifest: mf, tensors, embeddings, downloadBytes, gpuBytes,
    timing: { fetchMs, bodyMs, sha256Ms, uploadMs, embedJoinMs },
  };
}
