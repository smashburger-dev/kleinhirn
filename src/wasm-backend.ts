// WASM backend of EncoderModel (R2): the plan executor of src/wasm/plan.ts in a worker, one
// thread. Loaded on demand (dynamic import), so the core bundle carries none of it. The main
// thread fetches and verifies the f32 shards, keeps the word table (keepOnCpu, Festlegung 3)
// and hands every other tensor to the worker; a call posts the word rows, mask and type ids
// and gets the output back.

import type { Backend, LoadedBackend, PlanRunner } from './backend.ts';
import type { RunInput } from './plan/executor.ts';
import type { Plan } from './plan/ir.ts';
import type { ThreadSplit, WasmBuild } from './plan/wasm.ts';
import type { HostReply, HostRequest } from './wasm-plan-host.ts';
import type { FetchedShards, Manifest } from './weights.ts';

export interface Transport {
  request(msg: HostRequest, transfer?: Transferable[]): Promise<HostReply>;
  close(): void;
}

export function workerTransport(): Transport {
  const worker = new Worker(new URL('./wasm-plan-worker.ts', import.meta.url), { type: 'module' });
  const pending = new Map<number, { resolve: (r: HostReply) => void; reject: (e: Error) => void }>();
  let seq = 0;
  const failAll = (e: Error): void => {
    for (const p of pending.values()) p.reject(e);
    pending.clear();
  };
  worker.onmessage = (e: MessageEvent<HostReply & { id: number }>) => {
    const p = pending.get(e.data.id);
    pending.delete(e.data.id);
    p?.resolve(e.data);
  };
  worker.onerror = (e: ErrorEvent) => failAll(new Error(`wasm worker failed: ${e.message ?? 'script error'}`));
  return {
    request(msg, transfer = []) {
      const id = (seq += 1);
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, msg }, transfer);
      });
    },
    close() {
      worker.terminate();
      failAll(new Error('wasm worker closed'));
    },
  };
}

const answer = (r: HostReply): HostReply => {
  if (r.error !== undefined) throw new Error(r.error);
  return r;
};

class WorkerRunner implements PlanRunner {
  readonly length: number;
  readonly batch: number;
  readonly markers: number;
  readonly bytes = 0; // the plan's share of the worker memory is in the backend's memoryBytes
  private input?: RunInput;
  private out?: Promise<Float32Array>;

  constructor(private backend: WasmBackend, plan: Plan, private pid: number) {
    this.length = plan.length;
    this.batch = plan.batch;
    this.markers = plan.markers;
  }

  upload(input: RunInput): void {
    this.input = input;
  }

  run(seqLen: number, capture = false): void {
    if (capture) this.assertCapturable();
    const input = this.input;
    if (!input) throw new Error('run before upload');
    this.input = undefined;
    // the arrays are built per call (EncoderModel.wordRows, maskAndTypes): hand them over
    const transfer = [input.embeddings, input.mask, input.typeIds, input.packedMarkers]
      .filter((a) => a !== undefined).map((a) => (a as ArrayBufferView).buffer as ArrayBuffer);
    this.out = this.backend.request({ type: 'run', pid: this.pid, input, rows: seqLen }, transfer)
      .then((r) => r.out as Float32Array);
  }

  readOutput(): Promise<Float32Array> {
    const out = this.out;
    if (!out) return Promise.reject(new Error('readOutput before run'));
    this.out = undefined;
    return out;
  }

  readCapture(): Promise<Float32Array> {
    return Promise.reject(new Error('layer capture is a WebGPU parity tool'));
  }

  assertCapturable(): void {
    throw new Error('layer capture is a WebGPU parity tool; the WASM path has none');
  }

  destroy(): void {
    void this.backend.request({ type: 'destroy', pid: this.pid }).catch(() => {});
  }
}

// Options of the WASM path (R8): the build ('auto' takes relaxed where it validates).
export interface WasmOptions {
  build?: WasmBuild | 'auto';
  // threads (R8): 'auto' is the core count; more than one needs cross-origin isolation, else one
  threads?: number | 'auto';
  split?: Partial<ThreadSplit>;
  // batch plans with threads (R8 hc9, default false): measured slower than rows one by one on the
  // bucket plan with 8 threads (FINDINGS §57), which computes only the real rows
  batchPlans?: boolean;
}

class WasmBackend implements Backend {
  readonly kind = 'wasm';
  weightBytes = 0;
  build: WasmBuild | undefined;
  threads = 1;
  threadNote: string | undefined;
  batchPlans = false;
  private memoryBytes = 0;
  private pid = 0;

  constructor(private t: Transport) {}

  async request(msg: HostRequest, transfer?: Transferable[]): Promise<HostReply> {
    const r = answer(await this.t.request(msg, transfer));
    if (r.memoryBytes !== undefined) this.memoryBytes = r.memoryBytes;
    if (r.weightBytes !== undefined) this.weightBytes = r.weightBytes;
    return r;
  }

  check(plan: Plan): void {
    if (plan.f16) throw new Error('the WASM path runs f32 plans');
  }

  runner(plan: Plan): PlanRunner {
    const pid = (this.pid += 1);
    // A failing translation answers the next sync (prepare) or the run of this plan.
    void this.t.request({ type: 'plan', pid, plan }).catch(() => {});
    return new WorkerRunner(this, plan, pid);
  }

  // Rows of a batch run one after the other on the bucket plan, which computes only their real rows.
  // A batch plan (threads and batchPlans true, R8 hc9) gives every op more rows for the helpers but
  // computes the padded rows too; with 8 threads it was 12 to 35 % slower per item.
  fitBatch(build: (batch: number) => Plan, batch: number): Plan | undefined {
    if (this.threads <= 1 || !this.batchPlans) return undefined;
    // the batch sizes of cache.ts (BATCH_SIZES) above 1; imported, they would split cache.ts out of the core
    const size = [16, 8, 4].find((s) => s <= batch);
    return size ? build(size) : undefined;
  }

  async prepare<S>(work: () => S): Promise<S> {
    const s = work();
    await this.request({ type: 'sync' });
    return s;
  }

  async call<S, R>(start: () => S, read: (started: S) => Promise<R>): Promise<R> {
    return read(start());
  }

  info(): Record<string, unknown> {
    return { backend: 'wasm', threads: this.threads, ...(this.threadNote ? { threadNote: this.threadNote } : {}),
      wasmBuild: this.build, wasmMemoryBytes: this.memoryBytes };
  }

  dispose(): void {
    // stop the helpers first (Node keeps a process alive while a worker waits)
    void this.t.request({ type: 'close' }).catch(() => {}).finally(() => this.t.close());
  }
}

// fetched: the verified shards (EncoderModel fetches them, so weights.ts stays in the core bundle).
export async function loadWasm(
  fetched: FetchedShards, manifest: Manifest, transport: Transport, options: WasmOptions = {},
): Promise<LoadedBackend> {
  const tJoin = performance.now();
  const parts = manifest.tensors.filter((t) => t.keepOnCpu)
    .sort((a, b) => (a.rowStart ?? 0) - (b.rowStart ?? 0));
  if (parts.some((t) => t.name !== 'embeddings.word.weight')) throw new Error('only the word table stays on the CPU');
  const embeddings = new Float32Array(parts.reduce((n, t) => n + t.byteLength, 0) / 4);
  let at = 0;
  for (const t of parts) {
    embeddings.set(new Float32Array(fetched.shardBytes[t.shard].slice(t.offset, t.offset + t.byteLength)), at);
    at += t.byteLength / 4;
  }
  const embedJoinMs = performance.now() - tJoin;
  const tUp = performance.now();
  const backend = new WasmBackend(transport);
  const init = await backend.request(
    { type: 'init', shards: fetched.shardBytes, tensors: manifest.tensors.filter((t) => !t.keepOnCpu),
      build: options.build, threads: options.threads, split: options.split },
    fetched.shardBytes);
  backend.build = init.build;
  backend.threads = init.threads ?? 1;
  backend.threadNote = init.threadNote;
  backend.batchPlans = options.batchPlans === true;
  return {
    backend, embeddings, downloadBytes: fetched.downloadBytes,
    timing: { fetchMs: fetched.fetchMs, bodyMs: fetched.bodyMs, sha256Ms: fetched.sha256Ms,
      uploadMs: performance.now() - tUp, embedJoinMs },
  };
}
