// R1 kleinhirn adapter: the bundled dist/kleinhirn.js (as bench/k28-latency.ts), EncoderModel.load
// with the manifest of the way, standard (minimum) limits, buckets from the spec. Inputs are
// unpadded; the engine picks the smallest bucket that fits and dispatches only the real rows.
// Way wasm (R2): the f32 manifest on the WASM plan executor, threads from the options (default 1).

// @ts-expect-error runtime bundle built by vite lib mode has no d.ts
import { EncoderModel as BundleEncoderModel } from '../../../dist/kleinhirn.js';
import type { EncoderModel as EncoderModelType } from '../../../src/encoder.ts';
import type { BenchEngine, BenchInput, EngineInfo, LoadInfo, LoadSpec } from '../engine.ts';

const EncoderModel = BundleEncoderModel as typeof EncoderModelType;

export class KleinhirnEngine implements BenchEngine {
  readonly name: string;
  private readonly build: string;
  private enc: EncoderModelType | null = null;
  private precision = 'f16';
  private wasm = false;
  private threads: number | 'auto' = 1;
  private split: Record<string, number> | undefined;

  // build: the bundle to measure. 'dist-ref/kleinhirn.js' is an older build of the same engine,
  // copied there for an ABAB comparison of two commits in one session (stage ab of run-kbench).
  constructor(build = 'dist/kleinhirn.js') {
    this.build = build;
    this.name = build === 'dist/kleinhirn.js' ? 'kleinhirn' : 'kleinhirn-ref';
  }

  async load(spec: LoadSpec): Promise<LoadInfo> {
    this.precision = spec.way === 'webgpu-f16' ? 'f16' : 'f32';
    this.wasm = spec.way === 'wasm';
    const t = spec.options.threads ?? 1;
    this.threads = t === 'auto' ? 'auto' : Number(t);
    // split=rb,cb,qb,minWork (R8 hc8 search)
    const sp = String(spec.options.split ?? '');
    this.split = sp ? Object.fromEntries(['rb', 'cb', 'qb', 'minWork'].map((k, i) => [k, Number(sp.split(',')[i])]).filter(([, v]) => Number.isFinite(v as number))) : undefined;
    const t0 = performance.now();
    const Model = this.build === 'dist/kleinhirn.js' ? EncoderModel
      : (await import(/* @vite-ignore */ `/${this.build}`)).EncoderModel as typeof EncoderModelType;
    this.enc = await Model.load({
      manifestUrl: `/models/k28/${spec.model}/${this.precision}/manifest.json`,
      precision: this.precision as 'f16' | 'f32', buckets: spec.buckets, limits: 'minimum',
      ...(this.wasm ? { backend: 'wasm' as const, threads: this.threads, ...(this.split ? { wasmSplit: this.split } : {}) } : {}),
    });
    const loaded = this.enc.info();
    if (loaded.precision !== this.precision) throw new Error(`engine runs ${String(loaded.precision)}`);
    return { loadMs: performance.now() - t0 };
  }

  async run(input: BenchInput): Promise<Float32Array> {
    if (!this.enc) throw new Error('not loaded');
    return (await this.enc.runIds({ inputIds: input.ids, typeIds: input.typeIds })).data;
  }

  info(): EngineInfo {
    const i = (this.enc?.info() ?? {}) as Record<string, unknown>;
    return {
      name: this.name, version: String(i.buildId ?? ''), build: this.build,
      config: { precision: this.precision, buckets: i.buckets ?? null, limits: 'minimum',
        ...(this.wasm ? { backend: 'wasm', threads: i.threads ?? this.threads, threadNote: i.threadNote ?? null,
          wasmBuild: i.wasmBuild ?? null, split: this.split ?? null } : {}) },
      gpuBytes: typeof i.gpuBytes === 'number' ? i.gpuBytes : null, adapter: i.adapter ?? null,
    };
  }

  async dispose(): Promise<void> {
    this.enc?.dispose();
    this.enc = null;
  }
}
