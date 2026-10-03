// K28.1 bit runner: SHA-256 of the raw logit bytes of every golden case,
// per (golden file, batch size), so two commits can be compared bit for bit.
// GLiNER files run through runPrepared (B1) and runPreparedBatch in groups
// of four (B4); julia-1 uses parity100. Result cache off, minimum limits.
// A WebGPU validation error ends the run: without the listener a wrong plan
// silently returns zeros.
// Query: ?model=small-upstream|base-upstream|multi-upstream|julia-1
//        &precision=f32|f16

import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';
import { compareLogits } from './metrics.ts';

interface GoldenItem {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups?: number[];
  logits?: number[];
}

interface JuliaItem {
  seq_len: number;
  input_ids: number[];
  markers: number[];
  qtype: number;
  logits: number[];
}

interface BatchHash {
  sha: string;
  caseShas: string[];
  argmaxAgreement: number | null;
  maxAbsLogitDiff: number | null;
}

interface FileHash {
  cases: number;
  bucket: number;
  markers: number;
  B1?: BatchHash;
  B4?: BatchHash;
}

interface Result {
  stage: string;
  model?: string;
  precision?: string;
  info?: unknown;
  adapterInfo?: unknown;
  files?: Record<string, FileHash>;
  gpuErrors?: string[];
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khBitsResult?: Result }
}

const gpuErrors: string[] = [];

function watchDevices(): void {
  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function patched(
    this: GPUAdapter, ...args: Parameters<GPUAdapter['requestDevice']>
  ): Promise<GPUDevice> {
    const device = await orig.apply(this, args);
    device.addEventListener('uncapturederror', (e) => {
      gpuErrors.push((e as GPUUncapturedErrorEvent).error.message);
    });
    void device.lost.then((info) => gpuErrors.push(`device lost: ${info.message}`));
    return device;
  };
}

function checkErrors(): void {
  if (gpuErrors.length) throw new Error(`WebGPU: ${gpuErrors[0]}`);
}

const hex = (buf: ArrayBuffer): string =>
  [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

async function sha256(bytes: BufferSource): Promise<string> {
  return hex(await crypto.subtle.digest('SHA-256', bytes));
}

function bytesOf(a: Float32Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(a.byteLength);
  out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  return out;
}

async function digestAll(logits: Float32Array[]): Promise<BatchHash['caseShas']> {
  const out: string[] = [];
  for (const l of logits) out.push(await sha256(bytesOf(l)));
  return out;
}

async function summarize(
  logits: Float32Array[], refs: (number[] | undefined)[], valid: number[],
): Promise<BatchHash> {
  const caseShas = await digestAll(logits);
  const total = logits.reduce((n, l) => n + l.byteLength, 0);
  const all = new Uint8Array(total);
  let off = 0;
  for (const l of logits) {
    all.set(bytesOf(l), off);
    off += l.byteLength;
  }
  const idx = refs.map((r, i) => (r ? i : -1)).filter((i) => i >= 0);
  let argmaxAgreement: number | null = null;
  let maxAbsLogitDiff: number | null = null;
  if (idx.length) {
    const p = compareLogits(
      idx.map((i) => refs[i] as number[]),
      idx.map((i) => Array.from(logits[i].slice(0, valid[i]))));
    argmaxAgreement = p.argmaxAgreement;
    maxAbsLogitDiff = p.maxAbsLogitDiff;
  }
  return { sha: await sha256(all), caseShas, argmaxAgreement, maxAbsLogitDiff };
}

const chunksOf = <T>(arr: T[], n: number): T[][] => {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
};

function toInput(item: GoldenItem, kMax: number): SchemaInput {
  const markerIndices = new Int32Array(kMax);
  markerIndices.set(item.marker_indices.slice(0, kMax));
  const markerMask = new Float32Array(kMax);
  markerMask.set(item.marker_mask.slice(0, kMax));
  const markerGroups = new Int32Array(kMax);
  markerGroups.set((item.marker_groups ?? []).slice(0, kMax));
  return {
    inputIds: Int32Array.from(item.input_ids),
    attentionMask: Int32Array.from(item.attention_mask),
    markerIndices, markerMask, markerGroups,
    seqLen: item.seq_len,
  };
}

const GLINER_FILES: Record<string, string[]> = {
  'small-upstream': [
    'texts1000_l128k16', 'two_tasks_l128k16', 'long200_l256k16',
    'long200_l512k16', 'long200_l1024k16'],
  'base-upstream': [
    'texts1000_l128k16', 'two_tasks_l128k16', 'long200_l512k16',
    'long200_l1024k16', 'banking77_l1280k80'],
  'multi-upstream': [
    'texts1000_l128k16', 'two_tasks_l128k16', 'long200_l512k16',
    'long200_l1024k16', 'banking77_l1280k80'],
};

function fileBucket(name: string): { length: number; markers: number } {
  const m = /_l(\d+)k(\d+)$/.exec(name);
  if (!m) throw new Error(`bucket not in file name: ${name}`);
  return { length: Number(m[1]), markers: Number(m[2]) };
}

async function glinerBits(
  model: string, precision: string, result: Result,
): Promise<void> {
  const names = GLINER_FILES[model];
  if (!names) throw new Error(`unknown model ${model}`);
  const buckets = names.map(fileBucket);
  // One bucket per length; a longer marker budget wins.
  const byLength = new Map<number, number>();
  for (const b of buckets) {
    byLength.set(b.length, Math.max(byLength.get(b.length) ?? 0, b.markers));
  }
  const kh = await Kleinhirn.load({
    manifestUrl: `/models/${model}/${precision}/manifest.json`,
    buckets: [...byLength].map(([length, markers]) => ({ length, markers })),
    precision: 'auto', limits: 'minimum', cacheSize: 0,
  });
  checkErrors();
  result.info = kh.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  result.files = {};
  for (const name of names) {
    const { length, markers } = fileBucket(name);
    const golden = (await (await fetch(
      `/tests/golden/${model}/${name}.json`)).json()) as { items: GoldenItem[] };
    const items = golden.items;
    const inputs = items.map((i) => toInput(i, markers));
    const refs = items.map((i) => i.logits);
    const valid = items.map((i) => i.marker_mask.filter((m) => m > 0.5).length);
    const entry: FileHash = { cases: items.length, bucket: length, markers };

    const b1: Float32Array[] = [];
    for (const [i, input] of inputs.entries()) {
      b1.push((await kh.runPrepared(input, false, length)).logits);
      if (i % 100 === 0) {
        result.stage = `${name} B1 ${i}/${items.length}`;
        checkErrors();
      }
    }
    checkErrors();
    entry.B1 = await summarize(b1, refs, valid);

    const b4: Float32Array[] = [];
    for (const chunk of chunksOf(inputs, 4)) {
      const res = await kh.runPreparedBatch(chunk, length);
      for (const r of res) b4.push(r.logits);
      if (b4.length % 100 < 4) {
        result.stage = `${name} B4 ${b4.length}/${items.length}`;
        checkErrors();
      }
    }
    checkErrors();
    entry.B4 = await summarize(b4, refs, valid);
    result.files[name] = entry;
  }
  kh.dispose();
}

function toJuliaInput(item: JuliaItem): JuliaPreparedInput {
  return {
    inputIds: Int32Array.from(item.input_ids), markers: item.markers,
    qtype: item.qtype, seqLen: item.seq_len,
  };
}

async function juliaBits(precision: string, result: Result): Promise<void> {
  const kh = await JuliaEngine.load({
    manifestUrl: `/models/julia-1/${precision}/manifest.json`,
    buckets: [512, 1024], precision: 'auto', limits: 'minimum', cacheSize: 0,
  });
  checkErrors();
  result.info = kh.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  const golden = (await (await fetch(
    '/tests/golden/julia-1/parity100.json')).json()) as { items: JuliaItem[] };
  const items = golden.items;
  const inputs = items.map(toJuliaInput);
  const refs = items.map((i) => i.logits);
  const valid = items.map((i) => i.markers.length);
  const entry: FileHash = { cases: items.length, bucket: 0, markers: 20 };

  const b1: Float32Array[] = [];
  for (const [i, input] of inputs.entries()) {
    b1.push((await kh.runPrepared(input)).logits);
    if (i % 20 === 0) {
      result.stage = `parity100 B1 ${i}/${items.length}`;
      checkErrors();
    }
  }
  checkErrors();
  entry.B1 = await summarize(b1, refs, valid);

  const b4: Float32Array[] = [];
  for (const chunk of chunksOf(inputs, 4)) {
    const res = await kh.runPreparedBatch(chunk);
    for (const r of res) b4.push(r.logits);
  }
  checkErrors();
  entry.B4 = await summarize(b4, refs, valid);
  result.files = { parity100: entry };
  kh.dispose();
}

async function main(): Promise<void> {
  watchDevices();
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f32';
  const result: Result = { stage: 'boot', model, precision };
  window.khBitsResult = result;
  try {
    if (model === 'julia-1') await juliaBits(precision, result);
    else await glinerBits(model, precision, result);
    checkErrors();
    result.gpuErrors = [...gpuErrors];
    result.stage = 'done';
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
  }
  result.done = true;
}

void main();
