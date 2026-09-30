// kleinhirn device benchmark page. Measures the fallback chain (f16, f32,
// wasm) of the engine bundle that sits next to this page (./kleinhirn.js) and
// produces a kleinhirn-device-result/1 JSON. The same page is driven by the
// official runner with ?autorun=1. Vanilla TS, no third-party requests.

import { compareLogits, summarizeLatency, type ParitySummary } from '../bench/metrics.ts';
import { STAGE_ORDER, gateRule, parityPasses, type StageName } from './gates.ts';

const SCHEMA = 'kleinhirn-device-result/1';
// Pin of the weights repository revision (commit of smashburger-dev/kleinhirn-weights).
const REVISION = 'f664c7a5ad15c81f1258e72c8f90925b64711d1d';
const DEFAULT_WEIGHTS =
  `https://huggingface.co/smashburger-dev/kleinhirn-weights/resolve/${REVISION}/small-upstream/`;
const LOCAL_WEIGHTS = '/models/small-upstream/';
const BUCKETS = [128, 256];
const BUCKET_KEYS = ['L128', 'L256'] as const;
type BucketKey = typeof BUCKET_KEYS[number];
const K_MAX = 16;
const DEFAULT_WARMUP = 20;
const MATRIX_ITEMS = 200;

// ---- run options (URL params) --------------------------------------------
// warmup=<n>: warm-up items per bucket (default 20). limit=<n>: timed items
// per bucket (default all). trace=1: record every item's time in the result.
// report=local: POST the final JSON to /__result on this origin (only the
// local test server has that endpoint). The public page never sets these.
const params = new URLSearchParams(location.search);
const autorun = params.get('autorun') === '1';
const intParam = (key: string): number | null => {
  const raw = params.get(key);
  if (raw === null) return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) throw new Error(`bad ${key}=${raw}`);
  return n;
};
const WARMUP = intParam('warmup') ?? DEFAULT_WARMUP;
const LIMIT = intParam('limit');
const TRACE = params.get('trace') === '1';
const REPORT_LOCAL = params.get('report') === 'local';

// ---- types ---------------------------------------------------------------

interface GoldenItem {
  title: string;
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups: number[];
  logits: number[];
}
interface GoldenSet { task: string; labels: string[]; bucket: unknown; items: GoldenItem[] }
interface GoldenDoc {
  schema: string;
  setsSha256: string;
  sourceSha256: Record<string, string>;
  sets: Record<BucketKey, GoldenSet>;
}
interface PreparedInput {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  markerIndices: Int32Array;
  markerMask: Float32Array;
  markerGroups: Int32Array;
  seqLen: number;
}
interface Engine {
  classify(text: string, tasks: { task: string; labels: string[] }[]):
    Promise<{ tasks: { labels: { probability: number }[] }[] }>;
  runPrepared(input: PreparedInput): Promise<{ logits: Float32Array; probabilities: Float32Array }>;
  info(): Record<string, unknown>;
  dispose(): void;
}
interface EngineModule {
  loadEngine(options: Record<string, unknown>): Promise<Engine>;
}
interface ManifestInfo {
  ok: boolean;
  error: string | null;
  manifestUrl: string;
  manifestSha256: string | null;
  shards: { file: string; bytes: number; sha256: string }[];
  weightBytes: number;
  tokenizerBytes: number | null;
}
type Json = Record<string, unknown>;

// ---- helpers -------------------------------------------------------------

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const round6 = (v: number): number => Number(v.toPrecision(6));
function progress(text: string): void {
  $('progress').textContent = text;
  console.log(`[kleinhirn-site] ${text}`);
}

async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string | null> {
  if (!crypto.subtle) return null;
  const buf = await crypto.subtle.digest('SHA-256', data as BufferSource);
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function normalizeBase(raw: string): string {
  const mapped = raw === 'local' ? LOCAL_WEIGHTS : raw;
  const abs = new URL(mapped, location.href).href;
  return abs.endsWith('/') ? abs : `${abs}/`;
}

function isOverflow(e: unknown): boolean {
  const m = errText(e);
  return (e instanceof Error && e.name === 'BucketOverflowError')
    || /exceeds?\b.*\b(bucket|loaded)|exceed loaded buckets/.test(m);
}

// Pretty JSON with flat number arrays, so per-case logits stay one line each.
function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2).replace(
    /\[\s*((?:-?\d[\d.eE+-]*|null)(?:,\s*(?:-?\d[\d.eE+-]*|null))*)\s*\]/g,
    (_m, body: string) => `[${body.replace(/\s*,\s*/g, ', ')}]`);
}

// ---- environment ---------------------------------------------------------

const LIMIT_KEYS = [
  'maxTextureDimension1D', 'maxTextureDimension2D', 'maxTextureDimension3D',
  'maxTextureArrayLayers', 'maxBindGroups', 'maxBindGroupsPlusVertexBuffers',
  'maxBindingsPerBindGroup', 'maxDynamicUniformBuffersPerPipelineLayout',
  'maxDynamicStorageBuffersPerPipelineLayout', 'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage',
  'maxUniformBufferBindingSize', 'maxStorageBufferBindingSize',
  'minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment',
  'maxVertexBuffers', 'maxBufferSize', 'maxVertexAttributes',
  'maxVertexBufferArrayStride', 'maxInterStageShaderVariables',
  'maxColorAttachments', 'maxColorAttachmentBytesPerSample',
  'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension', 'maxImmediateSize',
  'maxStorageBuffersInVertexStage', 'maxStorageBuffersInFragmentStage',
  'maxStorageTexturesInVertexStage', 'maxStorageTexturesInFragmentStage',
];
const INFO_KEYS = ['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize'];

async function collectWebGpu(): Promise<Json> {
  const present = typeof navigator !== 'undefined' && !!navigator.gpu;
  const out: Json = { present, adapter: null, error: null };
  if (!present) return out;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ...out, error: 'requestAdapter returned null' };
    const rawInfo = (adapter.info ?? {}) as unknown as Record<string, unknown>;
    const info: Json = {};
    for (const k of INFO_KEYS) info[k] = rawInfo[k] ?? null;
    info.isFallbackAdapter = rawInfo.isFallbackAdapter ?? null;
    const limits: Record<string, number> = {};
    const rawLimits = adapter.limits as unknown as Record<string, unknown>;
    for (const k of LIMIT_KEYS) {
      if (typeof rawLimits[k] === 'number') limits[k] = rawLimits[k] as number;
    }
    out.adapter = { info, features: [...adapter.features].sort(), limits };
  } catch (e) {
    out.error = errText(e);
  }
  return out;
}

// Tiny module using v128.const and i8x16.popcnt: validates only where WASM SIMD exists.
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

async function workerHardwareConcurrency(): Promise<number | null> {
  const src = 'postMessage(navigator.hardwareConcurrency)';
  const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  const worker = new Worker(url);
  try {
    return await new Promise<number | null>((resolve) => {
      const timer = setTimeout(() => resolve(null), 5000);
      worker.onmessage = (e): void => { clearTimeout(timer); resolve(Number(e.data)); };
      worker.onerror = (): void => { clearTimeout(timer); resolve(null); };
    });
  } catch { return null; } finally {
    worker.terminate();
    URL.revokeObjectURL(url);
  }
}

async function collectEnvironment(): Promise<Json> {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    userAgentData?: { brands: { brand: string; version: string }[]; mobile: boolean; platform: string };
  };
  let persisted: boolean | null = null;
  let quotaBytes: number | null = null;
  try { persisted = await navigator.storage.persisted(); } catch { /* unavailable */ }
  try { quotaBytes = (await navigator.storage.estimate()).quota ?? null; } catch { /* unavailable */ }
  const uad = nav.userAgentData;
  return {
    userAgent: nav.userAgent,
    userAgentData: uad ? { brands: uad.brands, mobile: uad.mobile, platform: uad.platform } : null,
    hardwareConcurrency: nav.hardwareConcurrency ?? null,
    deviceMemory: nav.deviceMemory ?? null,
    maxTouchPoints: nav.maxTouchPoints ?? null,
    isSecureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated,
    wasmSimd: (() => { try { return WebAssembly.validate(SIMD_PROBE); } catch { return false; } })(),
    workerHardwareConcurrency: await workerHardwareConcurrency(),
    float16Array: typeof (globalThis as { Float16Array?: unknown }).Float16Array,
    opfs: !!(navigator.storage && typeof navigator.storage.getDirectory === 'function'),
    storagePersisted: persisted,
    storageQuotaBytes: quotaBytes,
    webgpu: await collectWebGpu(),
    rangeProbe: null,
  };
}

// ---- storage-age marker --------------------------------------------------
// Answers "how long do OPFS, Cache Storage and localStorage persist on this
// device?" (K28). Each store holds {firstSeen, lastSeen, visits}. On every
// load the page reads all three before measuring, reports what it found, then
// writes the updated marker back. Every storage call sits in try/catch.

interface Marker { firstSeen: string; lastSeen: string; visits: number }
interface StoreReport {
  present: boolean;
  firstSeen: string | null;
  lastSeen: string | null;
  visits: number | null;
  ageDays: number | null;
  readError: string | null;
  writeError: string | null;
}
const MARKER_FILE = 'kh-marker.json';
const MARKER_CACHE = 'kh-marker';
const MARKER_KEY = '/kh-marker';

function parseMarker(text: string): Marker {
  const m = JSON.parse(text) as Partial<Marker>;
  if (typeof m.firstSeen !== 'string' || typeof m.lastSeen !== 'string' || typeof m.visits !== 'number'
    || Number.isNaN(Date.parse(m.firstSeen))) throw new Error(`malformed marker: ${text.slice(0, 120)}`);
  return { firstSeen: m.firstSeen, lastSeen: m.lastSeen, visits: m.visits };
}

// OPFS write: createWritable where it exists, else a sync access handle in a
// throw-away worker (Safari has no createWritable on the main thread).
async function opfsWrite(text: string): Promise<void> {
  const dir = await navigator.storage.getDirectory();
  const handle = await dir.getFileHandle(MARKER_FILE, { create: true });
  if (typeof (handle as { createWritable?: unknown }).createWritable === 'function') {
    const w = await (handle as unknown as { createWritable(): Promise<{ write(d: string): Promise<void>; close(): Promise<void> }> }).createWritable();
    await w.write(text);
    await w.close();
    return;
  }
  const src = `onmessage = async (e) => { try {
    const dir = await navigator.storage.getDirectory();
    const fh = await dir.getFileHandle(${JSON.stringify(MARKER_FILE)}, { create: true });
    const ah = await fh.createSyncAccessHandle();
    const bytes = new TextEncoder().encode(e.data);
    ah.truncate(0); ah.write(bytes, { at: 0 }); ah.flush(); ah.close();
    postMessage('ok');
  } catch (err) { postMessage('error: ' + (err && err.message || err)); } };`;
  const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  const worker = new Worker(url);
  try {
    const reply = await new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('opfs worker timeout')), 10000);
      worker.onmessage = (e): void => { clearTimeout(timer); resolve(String(e.data)); };
      worker.onerror = (e): void => { clearTimeout(timer); reject(new Error(e.message || 'opfs worker error')); };
      worker.postMessage(text);
    });
    if (reply !== 'ok') throw new Error(reply);
  } finally {
    worker.terminate();
    URL.revokeObjectURL(url);
  }
}

const STORES: Record<string, {
  read(): Promise<string | null>;
  write(text: string): Promise<void>;
}> = {
  opfs: {
    async read() {
      const dir = await navigator.storage.getDirectory();
      try {
        const handle = await dir.getFileHandle(MARKER_FILE);
        return await (await handle.getFile()).text();
      } catch (e) {
        if (e instanceof DOMException && e.name === 'NotFoundError') return null;
        throw e;
      }
    },
    write: opfsWrite,
  },
  cache: {
    async read() {
      const cache = await caches.open(MARKER_CACHE);
      const res = await cache.match(MARKER_KEY);
      return res ? await res.text() : null;
    },
    async write(text) {
      const cache = await caches.open(MARKER_CACHE);
      await cache.put(MARKER_KEY, new Response(text, { headers: { 'Content-Type': 'application/json' } }));
    },
  },
  localStorage: {
    async read() { return localStorage.getItem(MARKER_FILE); },
    async write(text) { localStorage.setItem(MARKER_FILE, text); },
  },
};

async function collectStorageMarker(): Promise<Json> {
  const nowIso = new Date().toISOString();
  const report: Record<string, StoreReport> = {};
  for (const [name, store] of Object.entries(STORES)) {
    const r: StoreReport = {
      present: false, firstSeen: null, lastSeen: null, visits: null, ageDays: null,
      readError: null, writeError: null,
    };
    report[name] = r;
    let prev: Marker | null = null;
    try {
      const text = await store.read();
      if (text !== null) {
        prev = parseMarker(text);
        r.present = true;
        r.firstSeen = prev.firstSeen;
        r.lastSeen = prev.lastSeen;
        r.visits = prev.visits;
        r.ageDays = round6((Date.parse(nowIso) - Date.parse(prev.firstSeen)) / 86400000);
      }
    } catch (e) {
      r.readError = errText(e);
    }
    try {
      const next: Marker = prev
        ? { firstSeen: prev.firstSeen, lastSeen: nowIso, visits: prev.visits + 1 }
        : { firstSeen: nowIso, lastSeen: nowIso, visits: 1 };
      await store.write(JSON.stringify(next));
    } catch (e) {
      r.writeError = errText(e);
    }
  }
  return { checkedAt: nowIso, ...report };
}

function storageLine(marker: Json): string {
  const label: Record<string, string> = { opfs: 'OPFS', cache: 'Cache', localStorage: 'localStorage' };
  const parts: string[] = [];
  let any = false;
  for (const name of Object.keys(label)) {
    const r = marker[name] as StoreReport;
    if (r.present) {
      any = true;
      const d = r.ageDays ?? 0;
      parts.push(`${label[name]} ${d < 1 ? 'under 1 day' : `${Math.floor(d)} day${Math.floor(d) === 1 ? '' : 's'}`} old (${r.visits} visit${r.visits === 1 ? '' : 's'})`);
    } else {
      parts.push(`${label[name]} ${r.readError ? `error (${r.readError})` : 'none'}`);
    }
  }
  return any ? `Storage from an earlier visit: ${parts.join(', ')}.` : `First visit (${parts.join(', ')}).`;
}

// ---- weights host --------------------------------------------------------

async function remoteSize(url: string): Promise<number | null> {
  try {
    const head = await fetch(url, { method: 'HEAD' });
    const len = Number(head.headers.get('content-length'));
    if (head.ok && len > 0) return len;
  } catch { /* fall through */ }
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-0' } });
    const total = /\/(\d+)$/.exec(res.headers.get('content-range') ?? '');
    await res.arrayBuffer();
    if (total) return Number(total[1]);
  } catch { /* unknown */ }
  return null;
}

async function loadManifestInfo(base: string, dir: 'f16' | 'f32'): Promise<ManifestInfo> {
  const manifestUrl = `${base}${dir}/manifest.json`;
  const info: ManifestInfo = {
    ok: false, error: null, manifestUrl, manifestSha256: null,
    shards: [], weightBytes: 0, tokenizerBytes: null,
  };
  try {
    const res = await fetch(manifestUrl);
    if (!res.ok) throw new Error(`fetch ${manifestUrl}: ${res.status}`);
    const bytes = await res.arrayBuffer();
    info.manifestSha256 = await sha256Hex(bytes);
    const mf = JSON.parse(new TextDecoder().decode(bytes)) as {
      shards: { file: string; bytes: number; sha256: string }[]; tokenizer?: string;
    };
    info.shards = mf.shards.map((s) => ({ file: s.file, bytes: s.bytes, sha256: s.sha256 }));
    info.weightBytes = mf.shards.reduce((n, s) => n + s.bytes, 0);
    if (mf.tokenizer) info.tokenizerBytes = await remoteSize(`${base}${dir}/${mf.tokenizer}`);
    info.ok = true;
  } catch (e) {
    info.error = errText(e);
  }
  return info;
}

async function rangeProbe(base: string, f32: ManifestInfo): Promise<Json> {
  const shard = f32.shards[0]?.file ?? 'weights-0.bin';
  const url = `${base}f32/${shard}`;
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
    const buf = await res.arrayBuffer();
    return {
      url, status: res.status, bytesReceived: buf.byteLength,
      contentRange: res.headers.get('content-range'), error: null,
    };
  } catch (e) {
    return { url, status: null, bytesReceived: null, contentRange: null, error: errText(e) };
  }
}

// ---- measurement ---------------------------------------------------------

function toInput(item: GoldenItem): PreparedInput {
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

class HeapSampler {
  private timer = 0;
  baseline: number | null = null;
  peak: number | null = null;
  static readonly available =
    typeof (performance as Performance & { memory?: unknown }).memory === 'object';
  private read(): number | null {
    const m = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
    return m ? m.usedJSHeapSize : null;
  }
  start(): void {
    this.baseline = this.read();
    this.peak = this.baseline;
    if (HeapSampler.available) {
      this.timer = window.setInterval(() => {
        const v = this.read();
        if (v !== null && (this.peak === null || v > this.peak)) this.peak = v;
      }, 50);
    }
  }
  stop(): void {
    window.clearInterval(this.timer);
    const v = this.read();
    if (v !== null && (this.peak === null || v > this.peak)) this.peak = v;
  }
}

const latencyJson = (samples: number[], skipped: number): Json => {
  const s = summarizeLatency(samples);
  return {
    n: s.n, medianMs: s.medianMs, p95Ms: s.p95Ms, minMs: s.minMs, maxMs: s.maxMs,
    meanMs: s.meanMs, skipped,
  };
};

async function measureBucket(
  engine: Engine, stage: StageName, key: BucketKey, set: GoldenSet,
): Promise<{ latency: Json; parity: Json; trace: Json | null }> {
  const tasks = [{ task: set.task, labels: set.labels }];
  const nLabels = set.labels.length;
  const items = LIMIT === null ? set.items : set.items.slice(0, LIMIT);
  const tag = `${stage} ${key}`;
  // Per-item timing (trace=1): ms per call and start offset from the first
  // timed call of this bucket, both for the end-to-end and model-only loops.
  const trace = TRACE
    ? { e2eMs: [] as number[], e2eStartMs: [] as number[], modelOnlyMs: [] as number[], modelOnlyStartMs: [] as number[] }
    : null;
  let origin = 0;

  progress(`${tag}: warm-up`);
  for (const item of items.slice(0, WARMUP)) {
    try { await engine.classify(item.title, tasks); } catch (e) { if (!isOverflow(e)) throw e; }
    try { await engine.runPrepared(toInput(item)); } catch (e) { if (!isOverflow(e)) throw e; }
  }

  // End to end: clock starts before tokenization, stops with the
  // probabilities as a Float32Array in JS.
  const e2e: number[] = [];
  let e2eSkipped = 0;
  for (const [i, item] of items.entries()) {
    if (TRACE || i % 20 === 0) progress(`${tag}: end to end ${i}/${items.length}${trace && e2e.length ? ` (last ${e2e[e2e.length - 1].toFixed(0)} ms)` : ''}`);
    try {
      const t0 = performance.now();
      if (i === 0) origin = t0;
      const out = await engine.classify(item.title, tasks);
      const probs = Float32Array.from(out.tasks[0].labels.map((l) => l.probability));
      const t1 = performance.now();
      void probs;
      e2e.push(t1 - t0);
      trace?.e2eMs.push(round6(t1 - t0));
      trace?.e2eStartMs.push(round6(t0 - origin));
    } catch (e) {
      if (!isOverflow(e)) throw e;
      e2eSkipped += 1;
    }
  }

  // Model only: prepared arrays in, probabilities out. The same pass yields
  // the logits for the parity check.
  const modelOnly: number[] = [];
  let moSkipped = 0;
  const perCase: (number[] | null)[] = [];
  const ref: number[][] = [];
  const cand: number[][] = [];
  let moOrigin = 0;
  for (const [i, item] of items.entries()) {
    if (TRACE || i % 20 === 0) progress(`${tag}: model only ${i}/${items.length}${trace && modelOnly.length ? ` (last ${modelOnly[modelOnly.length - 1].toFixed(0)} ms)` : ''}`);
    try {
      const t0 = performance.now();
      if (i === 0) moOrigin = t0;
      const input = toInput(item);
      const res = await engine.runPrepared(input);
      const probs = Float32Array.from(res.probabilities.slice(0, nLabels));
      const t1 = performance.now();
      void probs;
      modelOnly.push(t1 - t0);
      trace?.modelOnlyMs.push(round6(t1 - t0));
      trace?.modelOnlyStartMs.push(round6(t0 - moOrigin));
      const nValid = item.marker_mask.filter((m) => m > 0.5).length;
      const got = Array.from(res.logits.slice(0, nValid), round6);
      perCase.push(got);
      ref.push(item.logits.slice(0, nValid));
      cand.push(got);
    } catch (e) {
      if (!isOverflow(e)) throw e;
      moSkipped += 1;
      perCase.push(null);
    }
  }
  const summary: ParitySummary = compareLogits(ref, cand);
  const pass = moSkipped === 0 && parityPasses(stage, summary);
  return {
    latency: { e2e: latencyJson(e2e, e2eSkipped), modelOnly: latencyJson(modelOnly, moSkipped) },
    parity: { summary, pass, missing: moSkipped, perCaseLogits: perCase },
    trace,
  };
}

interface RunContext {
  mod: EngineModule;
  base: string;
  manifests: Record<'f16' | 'f32', ManifestInfo>;
  goldens: GoldenDoc;
  hasF16: boolean;
}

function loadOptions(ctx: RunContext, stage: StageName | 'auto'): Record<string, unknown> {
  const dir = stage === 'f16' || (stage === 'auto' && ctx.hasF16) ? 'f16' : 'f32';
  return {
    manifestUrl: `${ctx.base}${dir}/manifest.json`,
    buckets: BUCKETS,
    precision: stage === 'f16' ? 'f16' : stage === 'f32' ? 'f32' : 'auto',
    limits: 'minimum',
    backend: stage === 'auto' ? 'auto' : stage === 'wasm' ? 'wasm' : 'webgpu',
    // Warm-up items reappear in the timed loops; the result cache would
    // serve hits instead of computing.
    cacheSize: 0,
  };
}

function pickedStage(info: Record<string, unknown>): StageName {
  if (info.backend === 'wasm-simd') return 'wasm';
  return info.precision === 'f16' ? 'f16' : 'f32';
}

function loadRecord(
  engine: Engine, wallMs: number, hasF16: boolean,
): Json {
  const info = engine.info();
  return {
    wallMs,
    loadTiming: info.loadTiming ?? null,
    backend: (info.backend as string | undefined) ?? 'webgpu',
    precision: info.precision ?? null,
    hasF16,
    limitsMode: info.limitsMode ?? null,
    gpuBytes: typeof info.gpuBytes === 'number' ? info.gpuBytes : null,
    downloadBytes: typeof info.downloadBytes === 'number' ? info.downloadBytes : null,
    engineBuildId: (info.buildId as string | undefined) ?? null,
  };
}

async function runStage(
  ctx: RunContext, stage: StageName,
  preloaded?: { engine: Engine; wallMs: number },
): Promise<Json> {
  let engine: Engine | null = preloaded?.engine ?? null;
  let load: Json | null = null;
  let phase: 'load' | 'measure' = 'load';
  const heap = new HeapSampler();
  try {
    heap.start();
    if (!engine) {
      progress(`${stage}: loading`);
      const t0 = performance.now();
      engine = await ctx.mod.loadEngine(loadOptions(ctx, stage));
      load = loadRecord(engine, performance.now() - t0, ctx.hasF16);
    } else {
      load = loadRecord(engine, preloaded!.wallMs, ctx.hasF16);
    }
    phase = 'measure';
    const parityBuckets: Json = {};
    const latency: Json = {};
    const trace: Json = {};
    for (const key of BUCKET_KEYS) {
      const r = await measureBucket(engine, stage, key, ctx.goldens.sets[key]);
      parityBuckets[key] = r.parity;
      latency[key] = r.latency;
      if (r.trace) trace[key] = r.trace;
    }
    heap.stop();
    const parityOk = BUCKET_KEYS.every((k) => (parityBuckets[k] as { pass: boolean }).pass);
    const measurable = HeapSampler.available;
    return {
      name: stage, ok: true, error: null, errorPhase: null, parityPass: parityOk,
      gateRule: gateRule(stage), load, parity: parityBuckets, latency,
      ...(TRACE ? { trace } : {}),
      memory: {
        gpuBytes: (load as { gpuBytes: number | null }).gpuBytes,
        jsHeapMeasurable: measurable,
        jsHeapBaselineBytes: measurable ? heap.baseline : null,
        jsHeapPeakBytes: measurable ? heap.peak : null,
        note: measurable
          ? (stage === 'wasm'
            ? 'JS heap of the page thread only; the wasm module runs in a worker and is not included.'
            : 'JS heap of the page thread; GPU memory is gpuBytes, the sum of the buffers the engine allocates.')
          : 'Peak memory is not measurable in this browser (no performance.memory). iOS has no web API for peak memory.',
      },
    };
  } catch (e) {
    heap.stop();
    return {
      name: stage, ok: false, error: errText(e), errorPhase: phase, parityPass: false,
      gateRule: gateRule(stage), load, parity: null, latency: null, memory: null,
    };
  } finally {
    try { engine?.dispose(); } catch { /* device already lost */ }
  }
}

// ---- page ----------------------------------------------------------------

interface StageUi { name: StageName; box: HTMLInputElement }

const STAGE_TEXT: Record<StageName, string> = {
  f16: 'WebGPU with half-precision weights (needs the shader-f16 feature).',
  f32: 'WebGPU with full-precision weights.',
  wasm: 'WASM-SIMD on the CPU, in a worker.',
};

const mib = (n: number): string => `${(n / 1048576).toFixed(0)} MB`;

function renderSummary(result: Json): void {
  const stages = result.stages as Json[];
  const fmt = (v: unknown): string => (typeof v === 'number' ? v.toFixed(2) : 'n/a');
  const rows = stages.map((s) => {
    if (!s.ok) {
      return `<tr><td>${s.name}</td><td class="fail">failed</td><td colspan="4"></td></tr>`
        + `<tr><td colspan="6" class="muted">${escapeHtml(String(s.error))}</td></tr>`;
    }
    const lat = s.latency as Record<BucketKey, { modelOnly: Json; e2e: Json }>;
    const par = s.parity as Record<BucketKey, { summary: ParitySummary; pass: boolean }>;
    const verdict = s.parityPass ? '<span class="pass">pass</span>' : '<span class="fail">fail</span>';
    const one = (k: BucketKey): string =>
      `${fmt(lat[k].modelOnly.medianMs)} / ${fmt(lat[k].modelOnly.p95Ms)} ms`
      + `<br><span class="muted">e2e ${fmt(lat[k].e2e.medianMs)} / ${fmt(lat[k].e2e.p95Ms)}</span>`;
    return `<tr><td>${s.name}</td><td>${verdict}<br><span class="muted">`
      + `${(par.L128.summary.argmaxAgreement * 100).toFixed(1)} % / ${(par.L256.summary.argmaxAgreement * 100).toFixed(1)} % argmax</span></td>`
      + `<td>${one('L128')}</td><td>${one('L256')}</td></tr>`;
  }).join('');
  $('summary').innerHTML = '<div class="tablewrap"><table><thead><tr><th>Stage</th><th>Parity</th>'
    + '<th>L128 median / p95</th><th>L256 median / p95</th></tr></thead>'
    + `<tbody>${rows}</tbody></table></div>`
    + `<p class="muted">Auto picked: ${(result.autoStage as Json).picked ?? 'nothing'}. Times are model only, one call at a time.</p>`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] as string);
}

async function main(): Promise<void> {
  document.body.dataset.state = 'running';
  const base = normalizeBase(params.get('weights') ?? DEFAULT_WEIGHTS);
  const model = params.get('model') ?? 'small-upstream';
  if (model !== 'small-upstream') throw new Error(`unsupported model ${model}`);

  progress('reading storage marker');
  const storageMarker = await collectStorageMarker();
  $('storage-line').textContent = storageLine(storageMarker);
  progress('collecting environment');
  const environment = await collectEnvironment();
  environment.storageMarker = storageMarker;
  const webgpu = environment.webgpu as { present: boolean; adapter: { features: string[] } | null };
  const hasAdapter = !!webgpu.adapter;
  const hasF16 = !!webgpu.adapter?.features.includes('shader-f16');
  const mobile = matchMedia('(pointer: coarse)').matches && window.innerWidth < 820;
  const env = environment as Json;
  const gpuName = (() => {
    const a = (webgpu.adapter as unknown as { info?: Record<string, unknown> } | null)?.info;
    if (!a) return webgpu.present ? 'WebGPU present, no adapter' : 'no WebGPU';
    return [a.vendor, a.architecture, a.device, a.description].filter(Boolean).join(' ') || 'adapter without name';
  })();
  $('env-summary').textContent =
    `${gpuName}. shader-f16: ${hasF16 ? 'yes' : 'no'}. Cores: ${env.hardwareConcurrency ?? '?'}. `
    + `Secure context: ${env.isSecureContext ? 'yes' : 'no'}.`;

  const [m16, m32] = await Promise.all([loadManifestInfo(base, 'f16'), loadManifestInfo(base, 'f32')]);
  const manifests = { f16: m16, f32: m32 };
  environment.rangeProbe = await rangeProbe(base, m32);

  const goldenRes = await fetch('./goldens-small-upstream.json');
  if (!goldenRes.ok) throw new Error(`goldens: ${goldenRes.status}`);
  const goldens = (await goldenRes.json()) as GoldenDoc;
  const setsSha = await sha256Hex(new TextEncoder().encode(JSON.stringify(goldens.sets)));
  if (setsSha !== null && setsSha !== goldens.setsSha256) {
    throw new Error(`goldens hash mismatch: file says ${goldens.setsSha256}, content is ${setsSha}`);
  }

  // Fields prefilled by the runner.
  if (params.get('device')) ($('f-model') as HTMLInputElement).value = params.get('device') as string;
  if (params.get('os')) ($('f-os') as HTMLInputElement).value = params.get('os') as string;

  // Stage selection UI.
  const requested = params.get('stages')?.split(',').map((s) => s.trim()).filter(Boolean) as StageName[] | undefined;
  const defaults: Record<StageName, boolean> = { f16: hasF16, f32: hasAdapter, wasm: !mobile };
  const ui: StageUi[] = [];
  const list = $('stage-list');
  const totalLine = document.createElement('p');
  totalLine.className = 'muted';
  const sizeOf = (s: StageName): { text: string; bytes: number; key: string } => {
    const mi = s === 'f16' ? m16 : m32;
    if (!mi.ok) return { text: `weights unreachable: ${mi.error}`, bytes: 0, key: s };
    const bytes = mi.weightBytes + (mi.tokenizerBytes ?? 0);
    return { text: `${mib(bytes)}${mi.tokenizerBytes === null ? ' + tokenizer' : ''}`, bytes, key: s === 'f16' ? 'f16' : 'f32' };
  };
  for (const name of STAGE_ORDER) {
    const row = document.createElement('label');
    row.className = 'stage';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = requested ? requested.includes(name) : defaults[name];
    const sz = sizeOf(name);
    const avail = name === 'f16' && !hasF16 ? ' This adapter has no shader-f16.'
      : name === 'f32' && !hasAdapter ? ' No WebGPU adapter here.' : '';
    const share = name === 'wasm' ? ' Uses the f32 files.' : name === 'f32' ? ' Files are shared with wasm.' : '';
    const text = document.createElement('span');
    text.innerHTML = `<b>${name}</b> <span class="muted">download ${escapeHtml(sz.text)}</span>`
      + `<span class="meta">${STAGE_TEXT[name]}${share}${avail}</span>`;
    row.append(box, text);
    list.append(row);
    ui.push({ name, box });
    box.addEventListener('change', updateTotal);
  }
  list.append(totalLine);
  function selected(): StageName[] { return ui.filter((u) => u.box.checked).map((u) => u.name); }
  function updateTotal(): void {
    const keys = new Map<string, number>();
    for (const s of selected()) { const z = sizeOf(s); keys.set(z.key, z.bytes); }
    const bytes = [...keys.values()].reduce((a, b) => a + b, 0);
    totalLine.textContent = `Download for the selected stages: ${mib(bytes)} (files are fetched once per precision).`;
  }
  updateTotal();
  {
    // Intro sentence from the real manifest sizes: f16 alone, and all stages
    // (f16 + f32; wasm reuses the f32 files).
    const b16 = sizeOf('f16').bytes;
    const b32 = sizeOf('f32').bytes;
    if (b16 && b32) {
      $('lead-size').textContent = `Downloads: f16 ${mib(b16)}, f32 ${mib(b32)} (shared with wasm), `
        + `all stages ${mib(b16 + b32)}.`;
    }
  }

  const start = $<HTMLButtonElement>('start');
  const modelInput = $<HTMLInputElement>('f-model');
  const refresh = (): void => { start.disabled = !modelInput.value.trim() || !selected().length; };
  modelInput.addEventListener('input', refresh);
  for (const u of ui) u.box.addEventListener('change', refresh);
  refresh();
  progress(autorun ? 'starting' : 'Ready. Fill in the device model and start.');
  document.body.dataset.state = 'ready';

  const ctxBase = { base, manifests, goldens, hasF16 };
  const go = async (): Promise<void> => {
    start.disabled = true;
    document.body.dataset.state = 'running';
    const chosen = selected();
    const mod = await importEngine();
    const ctx: RunContext = { ...ctxBase, mod: mod.module };
    const results: Json[] = [];
    let auto: Json = { picked: null, manifest: null, error: null, reusedAsMeasured: false };
    // What would an app get from backend 'auto'? That load doubles as the
    // measured load when the picked stage is selected.
    let reuse: { stage: StageName; engine: Engine; wallMs: number } | null = null;
    try {
      progress('auto: loading');
      const opts = loadOptions(ctx, 'auto');
      const t0 = performance.now();
      const engine = await mod.module.loadEngine(opts);
      const wallMs = performance.now() - t0;
      const picked = pickedStage(engine.info());
      auto = { picked, manifest: opts.manifestUrl, error: null, reusedAsMeasured: false };
      $('auto-pick').textContent = `Auto picks: ${picked}.`;
      if (chosen.includes(picked)) {
        reuse = { stage: picked, engine, wallMs };
        auto.reusedAsMeasured = true;
      } else {
        engine.dispose();
      }
    } catch (e) {
      auto = { picked: null, manifest: null, error: errText(e), reusedAsMeasured: false };
      $('auto-pick').textContent = `Auto failed: ${errText(e)}`;
    }
    const order: StageName[] = reuse ? [reuse.stage, ...chosen.filter((s) => s !== reuse!.stage)] : chosen;
    for (const stage of order) {
      const pre = reuse && reuse.stage === stage ? reuse : undefined;
      results.push(await runStage(ctx, stage, pre));
    }
    results.sort((a, b) => STAGE_ORDER.indexOf(a.name as StageName) - STAGE_ORDER.indexOf(b.name as StageName));

    const weightsStages: Json = {};
    for (const d of ['f16', 'f32'] as const) {
      const mi = manifests[d];
      weightsStages[d] = {
        manifestSha256: mi.manifestSha256, shards: mi.shards,
        tokenizerBytes: mi.tokenizerBytes, error: mi.error,
      };
    }
    const result: Json = {
      schema: SCHEMA,
      createdAt: new Date().toISOString(),
      page: {
        url: location.origin + location.pathname,
        buildId: __SITE_BUILD_ID__,
        engineCommit: __SITE_COMMIT__,
        engineBuildId: mod.engineBuildId,
        bundleSha256: mod.bundleSha256,
        workerSha256: mod.workerSha256,
      },
      weights: { base, stages: weightsStages },
      goldens: {
        sha256: goldens.setsSha256, model, buckets: [...BUCKET_KEYS],
        sourceSha256: goldens.sourceSha256,
      },
      device: {
        model: modelInput.value.trim(),
        os: $<HTMLInputElement>('f-os').value.trim(),
        protocol: {
          pluggedIn: $<HTMLInputElement>('p-plugged').checked,
          screenOn: $<HTMLInputElement>('p-screen').checked,
          lowPowerModeOff: $<HTMLInputElement>('p-lowpower').checked,
          otherTabsClosed: $<HTMLInputElement>('p-tabs').checked,
        },
      },
      environment,
      autoStage: auto,
      stages: results,
    };
    const text = prettyJson(result);
    const hash8 = ((await sha256Hex(new TextEncoder().encode(text))) ?? 'nohash00').slice(0, 8);
    $('json').textContent = text;
    renderSummary(result);
    $('out').hidden = false;
    $('download').onclick = (): void => {
      const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `kleinhirn-result-${new Date().toISOString().slice(0, 10)}-${hash8}.json`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    };
    $('copy').onclick = async (): Promise<void> => {
      try { await navigator.clipboard.writeText(text); progress('Copied.'); } catch {
        const pre = $('json');
        const r = document.createRange();
        r.selectNodeContents(pre);
        getSelection()?.removeAllRanges();
        getSelection()?.addRange(r);
        progress('Clipboard blocked; the JSON is selected, copy it by hand.');
      }
    };
    result.protocol = {
      warmup: WARMUP, limit: LIMIT, trace: TRACE,
      diagnostic: LIMIT !== null && LIMIT < MATRIX_ITEMS,
    };
    window.__khResult = result;
    await reportLocal(result);
    progress('Done.');
    document.body.dataset.state = 'done';
    start.disabled = false;
  };
  start.addEventListener('click', () => {
    go().catch((e) => fail(e));
  });
  if (autorun) {
    if (!selected().length) throw new Error('no stage selected');
    await go();
  }
}

async function importEngine(): Promise<{
  module: EngineModule; bundleSha256: string | null; workerSha256: string | null; engineBuildId: string | null;
}> {
  // The bundle sits next to the page and is not part of the page build, so
  // the exact bytes that run can be hashed here.
  const url = new URL('kleinhirn.js', document.baseURI).href;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const bytes = await res.arrayBuffer();
  const bundleSha256 = await sha256Hex(bytes);
  const bundleText = new TextDecoder().decode(bytes);
  const workerRel = /assets\/wasm-worker-[\w-]+\.js/.exec(bundleText)?.[0] ?? null;
  let workerSha256: string | null = null;
  if (workerRel) {
    const w = await fetch(new URL(workerRel, url).href);
    if (w.ok) workerSha256 = await sha256Hex(await w.arrayBuffer());
  }
  const engineBuildId = /buildId\s*:\s*["']([a-z0-9]+)["']/.exec(bundleText)?.[1] ?? null;
  const module = (await import(/* @vite-ignore */ url)) as EngineModule;
  return { module, bundleSha256, workerSha256, engineBuildId };
}

// report=local: hand the final JSON to the local test server, so a run needs no
// WebDriver. The public page never sets the param; a failed POST is logged and
// leaves the result in window.__khResult.
async function reportLocal(result: unknown): Promise<void> {
  if (!REPORT_LOCAL) return;
  try {
    const rid = encodeURIComponent(params.get('rid') ?? `report-${Date.now()}`);
    const res = await fetch(`/__result?rid=${rid}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(result),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    console.log(`[kleinhirn-site] report=local failed: ${errText(e)}`);
  }
}

function fail(e: unknown): void {
  const msg = errText(e);
  progress(`Error: ${msg}`);
  const result = { schema: SCHEMA, error: msg };
  window.__khResult = result;
  document.body.dataset.state = 'error';
  void reportLocal(result);
}

main().catch(fail);
