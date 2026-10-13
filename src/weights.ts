// Manifest + shard loading: fetch and sha256 verification. The GPU upload (loadWeights) lives in
// src/webgpu-backend.ts; the WASM path reads the same fetched bytes on the CPU.

import { fetchWords, legacyTable, type WordsFetch, type WordTable } from './words.ts';
import type { WordsDesc } from './words-format.ts';

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
  // version 3 (R4): the word table, one row per token id with a checksum, in words.bin
  words?: WordsDesc;
}

export interface LoadedWeights {
  manifest: Manifest;
  tensors: Map<string, GPUBuffer>;
  words: WordTable;
  downloadBytes: number;
  gpuBytes: number;
  timing: {
    fetchMs: number;
    bodyMs: number;
    sha256Ms: number;
    uploadMs: number;
    embedJoinMs: number;
    wordsFetchMs: number;
    wordsSha256Ms: number;
  };
}

export interface LoadedWords {
  table: WordTable;
  bytes: number; // words.bin bytes, 0 for a legacy manifest; RemoteRows: the bytes received so far
  timing: { embedJoinMs: number; wordsFetchMs: number; wordsSha256Ms: number };
}

// The word table of a manifest. Version 3: words.bin, fetched by loadBackend before the shards
// (started) or here when no fetch was started. Legacy: the keepOnCpu parts of the shards, joined
// in row order; their shape holds the width.
export async function loadWords(
  base: string, mf: Manifest, shardBytes: ArrayBuffer[], started?: Promise<WordsFetch>,
): Promise<LoadedWords> {
  if (mf.words) {
    const f = await (started ?? fetchWords(base + mf.words.file, mf.words));
    // RemoteRows counts its bytes as they arrive: read them at the end of the load
    return { table: f.table, bytes: Number(f.table.info().bytes ?? f.bytes),
      timing: { embedJoinMs: 0, wordsFetchMs: f.fetchMs, wordsSha256Ms: f.sha256Ms } };
  }
  const tJoin = performance.now();
  const parts = mf.tensors.filter((t) => t.keepOnCpu)
    .sort((a, b) => (a.rowStart ?? 0) - (b.rowStart ?? 0));
  if (!parts.length) throw new Error('manifest has neither words.bin nor a keepOnCpu word table');
  if (parts.some((t) => t.name !== 'embeddings.word.weight')) throw new Error('only the word table stays on the CPU');
  const joined = new Uint8Array(parts.reduce((n, t) => n + t.byteLength, 0));
  let at = 0;
  for (const t of parts) {
    joined.set(new Uint8Array(shardBytes[t.shard], t.offset, t.byteLength), at);
    at += t.byteLength;
  }
  const table = legacyTable(joined.buffer, parts[0].dtype === 'f16' ? 'f16' : 'f32', parts[0].shape[1]);
  return { table, bytes: 0, timing: { embedJoinMs: performance.now() - tJoin, wordsFetchMs: 0, wordsSha256Ms: 0 } };
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
