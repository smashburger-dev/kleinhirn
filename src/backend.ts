// The executor seam of EncoderModel (R2, docs/R2_WORKORDER.md Festlegung 2). EncoderModel builds
// plans, tokenizes, packs inputs and reads heads; a Backend turns a plan into a PlanRunner and
// runs a call. Behind it sit the WebGPU executor (GpuBackend, here) or the WASM executor in a
// worker (src/wasm-backend.ts, loaded on demand).

import { scopedCall, scopedSync, type KhDevice } from './device.ts';
import { assertPlan, fitBatchPlan } from './plan/check.ts';
import { PlanExecutor, type RunInput } from './plan/executor.ts';
import type { Plan } from './plan/ir.ts';
import type { Transport, WasmOptions } from './wasm-backend.ts';
import { fetchShardBytes, loadWeights, type LoadedWeights, type Manifest } from './weights.ts';

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
  embeddings: Float32Array | Uint16Array; // the word table, kept in JS (keepOnCpu)
  downloadBytes: number;
  timing: Record<string, number>;
}

// The weights of a manifest on WebGPU (kh given) or on the WASM executor in a worker (kh
// undefined; wasmTransport replaces the worker, as in the Node parity run). The WASM part loads
// on demand.
export async function loadBackend(
  manifestUrl: string, manifest: Manifest, kh: KhDevice | undefined, wasmTransport?: () => Transport,
  wasmOptions?: WasmOptions,
): Promise<LoadedBackend> {
  if (kh) {
    const weights = await loadWeights(kh.device, manifestUrl, manifest);
    return { backend: new GpuBackend(kh, weights), ...weights };
  }
  const [{ loadWasm, workerTransport }, fetched] = await Promise.all([
    import('./wasm-backend.ts'), fetchShardBytes(manifestUrl, manifest)]);
  return loadWasm(fetched, manifest, (wasmTransport ?? workerTransport)(), wasmOptions);
}

export class GpuBackend implements Backend {
  readonly kind = 'webgpu';
  readonly weightBytes: number;

  constructor(private kh: KhDevice, private weights: LoadedWeights) {
    this.weightBytes = weights.gpuBytes;
  }

  check(plan: Plan): void {
    assertPlan(plan, this.kh.device.limits, (name) => this.weights.tensors.get(name)?.size);
  }

  runner(plan: Plan): PlanRunner {
    return new PlanExecutor(this.kh.device, plan, this.weights.tensors);
  }

  fitBatch(build: (batch: number) => Plan, batch: number): Plan | undefined {
    return fitBatchPlan(build, this.kh.device.limits, batch);
  }

  prepare<S>(work: () => S): Promise<S> {
    return scopedSync(this.kh, work);
  }

  call<S, R>(start: () => S, read: (started: S) => Promise<R>): Promise<R> {
    return scopedCall(this.kh, start, read);
  }

  info(): Record<string, unknown> {
    return { adapter: this.kh.adapterInfo, limitsMode: this.kh.limitsMode, timestamps: this.kh.hasTimestamps };
  }

  dispose(): void {
    this.kh.device.destroy();
  }
}
