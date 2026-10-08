// R1 adapter contract (docs/R1_WORKORDER.md, "Adapter-Schnittstelle"). One bench page
// (bench/kbench.ts) drives any engine through this interface with the same loop:
// warm-up, measured window, output log, parity afterwards. R9 attaches more engines here.

export type Way = 'webgpu-f16' | 'webgpu-f32' | 'wasm';

export interface LoadSpec {
  model: string;                          // K28.8 slug, e.g. sentence-transformers__all-MiniLM-L6-v2
  task: string;                           // embeddings | sequence-classification | reranking
  way: Way;
  buckets: number[];                      // sequence buckets the engine must be able to run
  options: Record<string, unknown>;       // engine specific (ORT: build, capture, graph, dynamic, threads)
}

export interface LoadInfo {
  loadMs: number;                         // download, compile, warm caches
}

export interface BenchInput {
  ids: Int32Array;                        // unpadded token ids of one text, length <= largest bucket
  typeIds?: Int32Array;
}

export interface EngineInfo {
  name: string;
  version: string;                        // library version or build id
  build: string;                          // ORT: jsep | webgpu | jspi; kleinhirn: bundle
  config: Record<string, unknown>;        // the setting that ran (capture, threads, graphs, precision)
  gpuBytes: number | null;                // engine reported GPU memory, null when unknown
  adapter: unknown;                       // GPU adapter description, null for wasm
}

export interface BenchEngine {
  readonly name: string;                  // 'kleinhirn' | 'ort-web' | ...
  load(spec: LoadSpec): Promise<LoadInfo>;
  // The measured window: input to the output as a Float32Array in the main thread.
  run(input: BenchInput): Promise<Float32Array>;
  info(): EngineInfo;
  dispose(): Promise<void>;
}

export const BUCKETS = [128, 512] as const;

// Smallest bucket that holds `length` tokens (same rule for both engines, Festlegung 1).
export function bucketFor(length: number, buckets: readonly number[] = BUCKETS): number {
  const sorted = [...buckets].sort((a, b) => a - b);
  const hit = sorted.find((b) => length <= b);
  if (hit === undefined) throw new Error(`length ${length} exceeds the largest bucket ${sorted[sorted.length - 1]}`);
  return hit;
}
