// The executor seam of EncoderModel (R2, docs/R2_WORKORDER.md Festlegung 2). EncoderModel builds
// plans, tokenizes, packs inputs and reads heads; a Backend turns a plan into a PlanRunner and
// runs a call. Behind it sit the WebGPU executor (src/webgpu-backend.ts, loaded on demand) or the
// WASM executor in a worker (src/wasm-backend.ts, loaded on demand).

import type { KhDevice } from './device.ts';
import type { RunInput } from './plan/executor.ts';
import type { Plan } from './plan/ir.ts';
import type { Transport, WasmOptions } from './wasm-backend.ts';
import { fetchShardBytes, loadWords, type Manifest } from './weights.ts';
import { fetchWords, wordsView, type WordsFetch, type WordsMode, type WordTable } from './words.ts';
import type { WordsDesc } from './words-format.ts';

export interface PlanRunner {
  readonly length: number;
  readonly batch: number;
  readonly markers: number;
  // bytes held for this plan (buffers, capture buffer once it exists)
  readonly bytes: number;
  upload(input: RunInput): void;
  // one forward over seqLen rows (the real rows of a B = 1 call, all rows of a batch plan);
  // capture: also copy the layer states (f32 B = 1 parity tool, WebGPU only)
  run(seqLen: number, capture?: boolean): void;
  readOutput(): Promise<Float32Array>;
  readCapture(): Promise<Float32Array>;
  assertCapturable(): void;
  destroy(): void;
}

export interface Backend {
  readonly kind: 'webgpu' | 'wasm';
  // bytes of the loaded weights in the backend's memory
  readonly weightBytes: number;
  // Throws when a bucket plan cannot run here (device limits, f16 on WASM, ...).
  check(plan: Plan): void;
  runner(plan: Plan): PlanRunner;
  // The batch plan of the largest size up to `batch` that fits, or undefined: the rows then run on
  // the bucket plan one after the other.
  fitBatch(build: (batch: number) => Plan, batch: number): Plan | undefined;
  // Synchronous work at load (plan build), errors of the backend surface here.
  prepare<S>(work: () => S): Promise<S>;
  // One call: start uploads and runs, read reads the output.
  call<S, R>(start: () => S, read: (started: S) => Promise<R>): Promise<R>;
  info(): Record<string, unknown>;
  dispose(): void;
}

// The WASM path loads f32 weights: an f16 manifest URL maps to its f32 sibling.
export const f32ManifestUrl = (url: string): string => url.replace(/\/f16\//, '/f32/');

export interface LoadedBackend {
  backend: Backend;
  words: WordTable; // the word rows the engines gather (words.bin, or the keepOnCpu parts of a legacy manifest)
  downloadBytes: number;
  timing: Record<string, number>;
}

// The word table opens before the first await of loadBackend: rows and progressive import
// words-rows.ts and probe their Range while the shards load.
function startWords(url: string, desc: WordsDesc, mode: WordsMode): Promise<WordsFetch> {
  if (mode === 'full') return fetchWords(url, desc);
  const t0 = performance.now();
  return import('./words-rows.ts')
    .then((m) => m.RemoteRows.open(url, desc, { progressive: mode === 'progressive', view: wordsView }))
    .then((table) => ({ table, bytes: 0, fetchMs: performance.now() - t0, sha256Ms: 0 }));
}

// Loaders call webgpuChunk() before their first await, so the chunk downloads while the manifest
// and the adapter are fetched.
export const webgpuChunk = () => import('./webgpu-backend.ts');

// The weights of a manifest on WebGPU (kh given) or on the WASM executor in a worker (kh
// undefined; wasmTransport replaces the worker, as in the Node parity run). The WASM part loads
// on demand.
export async function loadBackend(
  manifestUrl: string, manifest: Manifest, kh: KhDevice | undefined, wasmTransport?: () => Transport,
  wasmOptions?: WasmOptions, wordsMode: WordsMode = 'full',
): Promise<LoadedBackend> {
  // version 3: words.bin is fetched next to the shards, before the backend is awaited
  const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
  const started = manifest.words ? startWords(base + manifest.words.file, manifest.words, wordsMode) : undefined;
  started?.catch(() => undefined); // a failure surfaces at the await in loadWords
  if (kh) return (await webgpuChunk()).loadGpu(kh, manifestUrl, manifest, started);
  const [{ loadWasm, workerTransport }, fetched] = await Promise.all([
    import('./wasm-backend.ts'), fetchShardBytes(manifestUrl, manifest)]);
  // the words are read before the worker takes the shard bytes (a legacy join reads them)
  const words = await loadWords(base, manifest, fetched.shardBytes, started);
  return loadWasm(fetched, manifest, (wasmTransport ?? workerTransport)(), wasmOptions, words);
}
