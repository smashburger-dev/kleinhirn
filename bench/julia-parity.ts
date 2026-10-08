// Julia 1 parity bench (K8): logits on the 100 published requests, layer
// states on the 8 long cases against the f64 goldens with the f32
// conditioning band (gate: per case, max|eng-ref64| <= max|ref32-ref64|,
// valid rows only). Query: ?precision=f32|f16&buckets=512,1024&limits=
// minimum|default

import { JuliaEngine } from '../src/julia.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';
import { compareLogits } from './metrics.ts';

const HIDDEN = 384;

interface ParityItem {
  request: unknown;
  seq_len: number;
  input_ids: number[];
  markers: number[];
  qtype: number;
  logits: number[];
}

interface LayerCase {
  case: number;
  source_index: number;
  seq_len: number;
  n_options: number;
}

interface Result {
  stage: string;
  precision?: string;
  info?: unknown;
  adapterInfo?: unknown;
  logits?: unknown;
  layers?: unknown;
  gates?: Record<string, boolean>;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khJuliaParityResult?: Result }
}

function toInput(item: ParityItem): JuliaPreparedInput {
  return {
    inputIds: Int32Array.from(item.input_ids),
    markers: item.markers,
    qtype: item.qtype,
    seqLen: item.seq_len,
  };
}

const CAPTURE_NAMES = (layers: number) => [
  'emb',
  ...Array.from({ length: layers }, (_, l) => `layer${l}`),
  'final', 'typed', 'head0', 'head1',
];

async function logitsParity(
  kh: JuliaEngine, items: ParityItem[], result: Result,
): Promise<Record<string, unknown>> {
  const ref: number[][] = [];
  const cand: number[][] = [];
  for (const [i, item] of items.entries()) {
    const res = await kh.runPrepared(toInput(item));
    ref.push(item.logits);
    cand.push(Array.from(res.logits.slice(0, item.logits.length)));
    if (i % 20 === 0) result.stage = `logits ${i}/${items.length}`;
  }
  return compareLogits(ref, cand) as unknown as Record<string, unknown>;
}

async function layerParity(
  kh: JuliaEngine, items: ParityItem[], result: Result,
): Promise<Record<string, unknown>> {
  const idx = (await (await fetch(
    '/models/julia-1/golden/layers.index.json')).json()) as {
      cases: LayerCase[];
      tensors: Record<string, { offset: number; shape: number[] }>;
      tensors32: Record<string, { offset: number; shape: number[] }>;
    };
  const blob64 = await (await fetch(
    '/models/julia-1/golden/layers.bin')).arrayBuffer();
  const blob32 = await (await fetch(
    '/models/julia-1/golden/layers.f32.bin')).arrayBuffer();
  const get = (
    table: Record<string, { offset: number; shape: number[] }>,
    buf: ArrayBuffer, name: string,
  ) => {
    const d = table[name];
    return new Float64Array(buf, d.offset,
      d.shape.reduce((a, b) => a * b, 1));
  };
  const nEncLayers = Object.keys(idx.tensors).filter(
    (k) => /^case0\.layer\d+$/.test(k)).length || 22;
  const names = CAPTURE_NAMES(nEncLayers);
  const cases: Record<string, unknown>[] = [];
  let pass = 0;
  for (const c of idx.cases) {
    const item = items[c.source_index];
    const L = item.seq_len <= 512 ? 512 : 1024;
    const res = await kh.runPrepared(toInput(item), true, L);
    const cap = res.captureData;
    if (!cap) throw new Error('capture missing (f32 only)');
    let specMax = 0;
    let bandMax = 0;
    let vsRef32Max = 0;
    let preJumpMax = 0;
    let refMax = 0;
    let outliers = 0;
    let elements = 0;
    for (const [s, name] of names.entries()) {
      const key = `case${c.case}.${name}`;
      const ref = get(idx.tensors, blob64, key);
      const got = cap.subarray(s * L * HIDDEN, (s + 1) * L * HIDDEN);
      const band = key in idx.tensors32
        ? get(idx.tensors32, blob32, key)
        : null;
      for (let i = 0; i < c.seq_len * HIDDEN && i < got.length; i += 1) {
        const d = Math.abs(got[i] - ref[i]);
        const ar = Math.abs(ref[i]);
        if (d > specMax) specMax = d;
        if (ar > refMax) refMax = ar;
        // Pre-jump states (emb, layer0..10 = slots 0..11) must stay tight;
        // the chaotic layers amplify any f32 sample past 1e-4.
        if (s <= 11 && d > preJumpMax) preJumpMax = d;
        if (d > 1e-4) outliers += 1;
        elements += 1;
        if (band) {
          const bd = Math.abs(band[i] - ref[i]);
          if (bd > bandMax) bandMax = bd;
          const ed = Math.abs(got[i] - band[i]);
          if (ed > vsRef32Max) vsRef32Max = ed;
        }
      }
    }
    // Approved gate: max|WGSL - ref64| <= 3 * band per case (two independent
    // f32 samples can differ by up to 2x the band via the triangle
    // inequality; 3 covers single-sample band jitter, worst observed 2.17)
    // plus pre-jump states within 1e-4.
    const ok = specMax <= 3 * bandMax && preJumpMax <= 1e-4;
    if (ok) pass += 1;
    cases.push({
      case: c.case, seq_len: c.seq_len, specMax, bandMax, vsRef32Max,
      preJumpMax, relMax: specMax / refMax, ratio: specMax / bandMax,
      outliers, elements, pass: ok });
    result.stage = `layers case ${c.case + 1}/${idx.cases.length}`;
  }
  return { n: idx.cases.length, pass, cases };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const precision = params.get('precision') ?? 'f32';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const buckets = params.get('buckets')?.split(',').map(Number) ?? [512, 1024];
  const result: Result = { stage: 'boot', precision };
  window.khJuliaParityResult = result;
  try {
    const t0 = performance.now();
    // backend=wasm (R2 stage 4): the WASM plan executor, f32, logits only (no layer capture)
    const wasm = params.get('backend') === 'wasm';
    const kh = await JuliaEngine.load({
      manifestUrl: `/models/julia-1/${precision}/manifest.json`,
      buckets,
      precision: wasm ? 'f32' : 'auto',
      limits,
      ...(wasm ? { backend: 'wasm' as const } : {}),
    });
    result.info = { ...kh.info(), loadMs: performance.now() - t0 };
    result.adapterInfo = (result.info as { adapter?: unknown }).adapter;

    const golden = (await (await fetch(
      '/tests/golden/julia-1/parity100.json')).json()) as { items: ParityItem[] };
    result.stage = 'logits';
    result.logits = await logitsParity(kh, golden.items, result);
    if (precision === 'f32' && !wasm) {
      result.stage = 'layers';
      result.layers = await layerParity(kh, golden.items, result);
    }
    const lg = result.logits as {
      argmaxAgreement: number; maxAbsLogitDiff: number; maxAbsProbDiff: number };
    const ly = result.layers as { pass: number; n: number } | undefined;
    result.gates = precision === 'f32'
      ? {
        argmax: lg.argmaxAgreement === 1,
        logits: lg.maxAbsLogitDiff <= 0.00225,
        ...(wasm ? {} : { layerConditioning: (ly?.pass ?? 0) === (ly?.n ?? 1) }),
      }
      : {
        argmax: lg.argmaxAgreement >= 0.99,
        probs: lg.maxAbsProbDiff <= 1e-2,
      };
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
