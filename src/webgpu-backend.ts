// R6 WebGPU chunk: the plan executor, the GPU backend and the weight upload to GPUBuffers.
// Loaded on demand through webgpuChunk() in backend.ts, so devices on the WASM path never
// download the executor, the WGSL kernels or loadWeights.

import { scopedCall, scopedSync, MINIMUM_LIMITS, type KhDevice } from './device.ts';
import { assertPlan, fitBatchPlan } from './plan/check.ts';
import { PlanExecutor } from './plan/executor.ts';
import type { Plan } from './plan/ir.ts';
import type { Backend, LoadedBackend, PlanRunner } from './backend.ts';
import { fetchShardBytes, loadWords, type LoadedWeights, type Manifest } from './weights.ts';
import type { WordsFetch } from './words.ts';

export { PlanExecutor } from './plan/executor.ts';

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


export async function loadWeights(
  device: GPUDevice,
  manifestUrl: string,
  manifest?: Manifest,
  started?: Promise<WordsFetch>,
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
  let gpuBytes = 0;
  const tUp = performance.now();
  for (const t of mf.tensors) {
    if (t.keepOnCpu) continue; // the word table: loadWords
    const bytes = new Uint8Array(shardBytes[t.shard], t.offset, t.byteLength);
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

  const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
  const words = await loadWords(base, mf, shardBytes, started);
  return {
    manifest: mf, tensors, words: words.table, downloadBytes: downloadBytes + words.bytes, gpuBytes,
    timing: { fetchMs, bodyMs, sha256Ms, uploadMs, embedJoinMs: words.timing.embedJoinMs,
      wordsFetchMs: words.timing.wordsFetchMs, wordsSha256Ms: words.timing.wordsSha256Ms },
  };
}

export async function loadGpu(
  kh: KhDevice, manifestUrl: string, manifest: Manifest, started?: Promise<WordsFetch>,
): Promise<LoadedBackend> {
  const weights = await loadWeights(kh.device, manifestUrl, manifest, started);
  return { backend: new GpuBackend(kh, weights), ...weights };
}
