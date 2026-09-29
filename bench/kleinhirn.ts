// kleinhirn engine bench per docs/ARCHITECTURE.md, importing the built
// library bundle (dist/kleinhirn.js) so download and bundle sizes are real.
// "Ende zu Ende": clock before tokenizing (classify includes tokenizer +
// schema build), stop when probabilities lie as Float32Array in JS.
// "Nur Modell": golden input arrays, same single-flight rule, no tokenization.
// Query: ?model=small-upstream|small-upstream&precision=f32|f16
//        &backend=webgpu|wasm&goldens=texts1000_l128k16|long200_l256k16|...

// @ts-expect-error runtime bundle built by vite lib mode has no d.ts
import { Kleinhirn, loadEngine } from '../dist/kleinhirn.js';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import { compareLogits, summarizeLatency } from './metrics.ts';

const K_MAX = 16;

interface GoldenItem {
  title: string;
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups?: number[];
  logits: number[];
}

interface GoldenFile {
  task: string;
  labels: string[];
  count: number;
  items: GoldenItem[];
}

interface KhResult {
  stage: string;
  model?: string;
  precision?: string;
  backend?: string;
  adapterInfo?: unknown;
  info?: unknown;
  loadMs?: number;
  loadMsWarm?: number;
  endToEnd?: unknown;
  modelOnly?: unknown;
  parity?: unknown;
  profile?: Record<string, number> | null;
  n?: number;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khResult?: KhResult }
}

function toInput(item: GoldenItem): SchemaInput {
  const markerIndices = new Int32Array(K_MAX);
  markerIndices.set(item.marker_indices.slice(0, K_MAX));
  const markerMask = new Float32Array(K_MAX);
  markerMask.set(item.marker_mask.slice(0, K_MAX));
  const markerGroups = new Int32Array(K_MAX);
  markerGroups.set((item.marker_groups ?? []).slice(0, K_MAX));
  return {
    inputIds: Int32Array.from(item.input_ids),
    attentionMask: Int32Array.from(item.attention_mask),
    markerIndices, markerMask, markerGroups,
    seqLen: item.seq_len,
  };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f32';
  const backend = params.get('backend') ?? 'webgpu';
  const goldenName = params.get('goldens') ?? 'texts1000_l128k16';
  const result: KhResult = { stage: 'boot', model, precision, backend };
  window.khResult = result;
  try {
    let golden: GoldenFile;
    try {
      const gres = await fetch(`/tests/golden/${model}/${goldenName}.json`);
      golden = (await gres.json()) as GoldenFile;
    } catch (e) {
      throw new Error(`golden ${goldenName}: ${e instanceof Error ? e.message : e}`);
    }
    const tasks = [{ task: golden.task, labels: golden.labels }];
    const nLabels = golden.labels.length;
    const items = golden.items;

    const opts = {
      manifestUrl: backend === 'wasm'
        ? `/models/${model}/f32/manifest.json`
        : `/models/${model}/${precision}/manifest.json`,
      buckets: params.get('buckets')?.split(',').map(Number) ?? [128],
      precision: 'auto' as const,
      limits: params.get('limits') === 'default' ? 'default' as const : 'minimum' as const,
      backend: backend === 'wasm' ? 'wasm' as const : 'webgpu' as const,
      // Warmup items reappear in the timed loops; caching would serve
      // hits instead of computing, so benchmark pages disable the K16
      // result cache.
      cacheSize: 0,
    };
    result.stage = 'loading';
    const tLoad = performance.now();
    let kh = backend === 'wasm'
      ? await loadEngine(opts)
      : await Kleinhirn.load(opts);
    result.loadMs = performance.now() - tLoad;
    result.info = kh.info();
    result.adapterInfo = (result.info as { adapter?: unknown }).adapter;

    // Warm reload: same context, browser cache hot. This is the number the
    // K5 load-time gate applies to.
    kh.dispose();
    const tWarm = performance.now();
    kh = backend === 'wasm'
      ? await loadEngine(opts)
      : await Kleinhirn.load(opts);
    result.loadMsWarm = performance.now() - tWarm;
    result.stage = 'warmup';
    for (const item of items.slice(0, 20)) {
      await kh.classify(item.title, tasks);
      await kh.runPrepared(toInput(item));
    }

    result.stage = 'end-to-end';
    const e2eMs: number[] = [];
    for (const [i, item] of items.entries()) {
      const t0 = performance.now();
      const out = await kh.classify(item.title, tasks);
      const probs = Float32Array.from(
        out.tasks[0].labels.map((l: { probability: number }) => l.probability));
      const t1 = performance.now();
      void probs;
      e2eMs.push(t1 - t0);
      if (i % 100 === 0) result.stage = `end-to-end ${i}/${items.length}`;
    }
    result.endToEnd = summarizeLatency(e2eMs);

    result.stage = 'model-only';
    const moMs: number[] = [];
    const ref: number[][] = [];
    const cand: number[][] = [];
    for (const [i, item] of items.entries()) {
      const t0 = performance.now();
      const input = toInput(item);
      const res = await kh.runPrepared(input);
      const probs = Float32Array.from(res.probabilities.slice(0, nLabels));
      const t1 = performance.now();
      void probs;
      moMs.push(t1 - t0);
      const nValid = item.marker_mask.filter((m) => m > 0.5).length;
      ref.push(item.logits.slice(0, nValid));
      cand.push(Array.from(res.logits.slice(0, nValid)));
      if (i % 100 === 0) result.stage = `model-only ${i}/${items.length}`;
    }
    result.modelOnly = summarizeLatency(moMs);
    result.parity = compareLogits(ref, cand);
    result.n = items.length;

    // K5 timestamp-query profiling: ?profile=1 runs 30 timed forwards and
    // reports per-pass medians (GPU timestamps); null result means the
    // device lacks the feature.
    if (params.get('profile') === '1' && backend !== 'wasm') {
      result.stage = 'profile';
      const perPass = new Map<string, number[]>();
      for (let i = 0; i < 30; i += 1) {
        const kt = await kh.profile(toInput(items[i % items.length]));
        if (kt === null) { result.profile = null; break; }
        for (const [k, v] of Object.entries(kt) as [string, number][]) {
          if (!perPass.has(k)) perPass.set(k, []);
          (perPass.get(k) as number[]).push(v);
        }
      }
      if (result.profile === undefined && perPass.size > 0) {
        const p: Record<string, number> = {};
        for (const [k, vs] of perPass) {
          p[k] = summarizeLatency(vs).medianMs;
        }
        result.profile = p;
      }
    }
    result.stage = 'done';
    result.done = true;
    kh.dispose();
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
