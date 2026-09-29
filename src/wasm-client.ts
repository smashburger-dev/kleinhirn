// Main-thread client for the WASM worker: same public surface as
// Kleinhirn (classify, runPrepared, info, dispose). Used by loadEngine in
// index.ts when the backend option resolves to WASM.

import type { ClassifyResult, LoadOptions, PreparedResult } from './index.ts';
import type { SchemaInput } from './tokenizer/schema.ts';
import { LruCache, dedupMerge, schemaMergeKey } from './cache.ts';

export class WasmClient {
  private worker!: Worker;
  private seq = 0;
  private pending = new Map<number, {
    resolve: (v: unknown) => void;
    reject: (e: Error) => void;
  }>();
  private infoData: Record<string, unknown> = {};
  private cache = new LruCache<ClassifyResult>(0);

  private constructor() {}

  static async load(options: LoadOptions): Promise<WasmClient> {
    const client = new WasmClient();
    client.cache = new LruCache(options.cacheSize ?? 256);
    client.worker = new Worker(
      new URL('./wasm-worker.ts', import.meta.url), { type: 'module' });
    // A failing worker URL or a startup exception otherwise leaves the load
    // promise pending forever.
    client.worker.onerror = (e: ErrorEvent) => {
      for (const p of client.pending.values()) {
        p.reject(new Error(`wasm worker failed: ${e.message ?? 'script error'}`));
      }
      client.pending.clear();
    };
    client.worker.onmessage = (e: MessageEvent) => {
      const m = e.data;
      const p = client.pending.get(m.id);
      if (!p) return;
      client.pending.delete(m.id);
      if (m.type === 'error') p.reject(new Error(m.message));
      else p.resolve(m.result ?? m.info);
    };
    const info = await client.call('load', { options }) as Record<string, unknown>;
    client.infoData = info;
    return client;
  }

  private call(type: string, payload: Record<string, unknown>): Promise<unknown> {
    const id = this.seq += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, ...payload });
    });
  }

  async classify(
    text: string,
    tasks: { task: string; labels: string[] }[],
  ): Promise<ClassifyResult> {
    return await this.call('classify', { text, tasks }) as ClassifyResult;
  }

  // WASM batch API (K16): the worker loops over the single-call path;
  // identical (text, tasks) pairs resolve once and hits come from the
  // client-side LRU keyed on the request pair.
  async classifyBatch(
    items: { text: string; tasks: { task: string; labels: string[] }[] }[],
  ): Promise<ClassifyResult[]> {
    const keyOf = (item: { text: string; tasks: unknown }) =>
      JSON.stringify([item.text, item.tasks]);
    const results: (ClassifyResult | undefined)[] = new Array(items.length);
    const pending: typeof items = [];
    const pendingIdx: number[] = [];
    for (const [i, item] of items.entries()) {
      const hit = this.cache.get(keyOf(item));
      if (hit) {
        results[i] = hit;
      } else {
        pending.push(item);
        pendingIdx.push(i);
      }
    }
    const { unique, slot } = dedupMerge(pending, keyOf);
    const uniqueResults: ClassifyResult[] = [];
    for (const item of unique) {
      const res = await this.classify(item.text, item.tasks);
      uniqueResults.push(res);
      this.cache.set(keyOf(item), res);
    }
    for (const [i] of pending.entries()) {
      results[pendingIdx[i]] = uniqueResults[slot[i]];
    }
    return results as ClassifyResult[];
  }

  async runPrepared(
    input: SchemaInput, capture = false, bucket?: number,
  ): Promise<PreparedResult> {
    return await this.call('runPrepared', { input, capture, bucket }) as PreparedResult;
  }

  // Prepared batch API (K16): loops single calls; merge-key dedup inside
  // one call matches the WebGPU contract (identical rows compute once).
  async runPreparedBatch(
    inputs: SchemaInput[], bucket?: number,
  ): Promise<PreparedResult[]> {
    const { unique, slot } = dedupMerge(inputs, schemaMergeKey);
    const uniqueResults: PreparedResult[] = [];
    for (const input of unique) {
      uniqueResults.push(await this.runPrepared(input, false, bucket));
    }
    return inputs.map((_, i) => uniqueResults[slot[i]]);
  }

  info(): Record<string, unknown> {
    return { ...this.infoData, worker: true };
  }

  dispose(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('worker disposed'));
    this.pending.clear();
  }
}
