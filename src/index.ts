// Public API: Kleinhirn.load / classify / info / dispose, plus the internal
// runPrepared used by "nur Modell" benchmarking and parity checks.
// Exactly one GPU call is in flight at a time; further calls queue.
// R6: this is the core (dist/kleinhirn.js). The GLiNER engine, the Julia engine and
// the tokenizers load on demand; the facades below keep their names and call shapes.

import { webgpuChunk } from './backend.ts';
import { fetchManifest } from './weights.ts';
import type { JuliaEngine } from './julia.ts';
import type { LoadOptions, Kleinhirn as KleinhirnEngine } from './gliner.ts';
import type { Transport } from './wasm-backend.ts';
import { EncoderModel } from './encoder.ts';

export { BucketOverflowError } from './errors.ts';
export { EncoderModel };
export { groupSoftmax } from './tasks.ts';
export type {
  EncoderInput, EncoderLoadOptions, EncoderOutput, RerankResult, RunOptions, TextClassification,
  TokenSpan, ZeroShotResult,
} from './encoder.ts';
export type { JsonTokenizer } from './tokenizer/hf/index.ts';
export type {
  BucketSpec, ClassifyLabel, ClassifyResult, LoadOptions, PreparedResult,
} from './gliner.ts';

const glinerChunk = () => import('./gliner.ts');
const juliaChunk = () => import('./julia.ts');

// The GLiNER engine loads on demand. Its WebGPU chunk starts downloading with it
// instead of after it, as EncoderModel.load does.
export type Kleinhirn = KleinhirnEngine;
export const Kleinhirn = {
  load: async (options: LoadOptions, deps?: { wasmTransport?: () => Transport }): Promise<KleinhirnEngine> => {
    if (options.backend !== 'wasm') webgpuChunk().catch(() => undefined);
    return (await glinerChunk()).Kleinhirn.load(options, deps);
  },
};

// Entry point honouring the backend option / fallback chain (docs/PLAN.md K5, R2): Kleinhirn
// (GLiNER, DeBERTa), JuliaEngine (modernbert-julia arch) or EncoderModel (kleinhirn-weights-2),
// on WebGPU or on the WASM plan executor in a worker. 'auto' tries WebGPU and falls back to WASM
// when navigator.gpu is absent or the WebGPU load fails; 'webgpu' and 'wasm' force a path.
export async function loadEngine(
  options: LoadOptions,
): Promise<Kleinhirn | JuliaEngine | EncoderModel> {
  const backend = options.backend ?? 'auto';
  const manifest = await fetchManifest(options.manifestUrl);
  const load = async (o: LoadOptions): Promise<Kleinhirn | JuliaEngine | EncoderModel> => {
    if (isJuliaManifest(manifest)) return (await juliaChunk()).JuliaEngine.load(o, manifest);
    if (isEncoderManifest(manifest)) return EncoderModel.load(encoderOptions(o));
    return Kleinhirn.load(o);
  };
  // The WASM path loads the f32 manifest: a precision of 'f16' does not carry over to it.
  const wasm: LoadOptions = { ...options, backend: 'wasm', precision: 'f32' };
  if (backend === 'wasm') return load(wasm);
  if (backend === 'webgpu') return load(options);
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      return await load({ ...options, backend: 'webgpu' });
    } catch {
      // no adapter, no device or a load that fails on this GPU: the CPU path below
    }
  }
  return load(wasm);
}

function encoderOptions(options: LoadOptions) {
  return {
    manifestUrl: options.manifestUrl, limits: options.limits, precision: options.precision,
    backend: options.backend === 'wasm' ? 'wasm' as const : 'webgpu' as const,
    buckets: options.buckets?.map((b) => (typeof b === 'number' ? b : b.length)),
  };
}

function isJuliaManifest(mf: { encoder?: Record<string, unknown> }): boolean {
  return mf.encoder?.arch === 'modernbert-julia';
}

function isEncoderManifest(mf: { format?: string }): boolean {
  return mf.format === 'kleinhirn-weights-2';
}
