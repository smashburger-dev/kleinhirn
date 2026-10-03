// K28.6b G4: a batch of 16 rows near the bucket limit must reproduce the single calls bit for
// bit. The residual add of Julia needs 98,304 workgroups at B16 and stride 1024, and the one of
// a ModernBERT with H 768 at B16 and stride 512, both above the 65,535 of one dispatch
// dimension. A dispatch above the limit is a validation error that the device reports only
// through uncapturederror, so the page listens for it and counts dispatches and submits.
// Query: ?kind=julia|encoder[&model=<slug>]&precision=f32|f16&sizes=8,16
// Rows: golden cases of the model, lengthened by repeating their own tokens to about 1,000
// (Julia, bucket 1024, stride 1024) or about 500 (encoder, bucket 512, stride 512). The
// repeated part is inserted after the first token, markers move with the tokens behind it.

import { EncoderModel } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';

interface JuliaItem { seq_len: number; input_ids: number[]; markers: number[]; qtype: number }
interface EncItem { input_ids: number[]; token_type_ids: number[] }

interface SizeResult {
  rows: number;
  differingRows: number;
  maxWorkgroupsX: number;
  submits: number;
  gpuErrors: string[];
  allZero: number;
}

interface Result {
  stage: string;
  kind?: string;
  model?: string;
  precision?: string;
  info?: unknown;
  adapterInfo?: unknown;
  lengths?: number[];
  sizes?: Record<string, SizeResult>;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khK286bResult?: Result }
}

let gpuErrors: string[] = [];
let maxX = 0;
let submits = 0;

function watchDevice(): void {
  const origReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function patched(
    this: GPUAdapter, ...args: Parameters<GPUAdapter['requestDevice']>
  ): Promise<GPUDevice> {
    const device = await origReq.apply(this, args);
    device.addEventListener('uncapturederror', (e) => {
      gpuErrors.push((e as GPUUncapturedErrorEvent).error.message);
    });
    void device.lost.then((info) => gpuErrors.push(`device lost: ${info.message}`));
    return device;
  };
  const origDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function patched(
    this: GPUComputePassEncoder, x: number, y?: number, z?: number,
  ): undefined {
    maxX = Math.max(maxX, x);
    origDispatch.call(this, x, y, z);
  };
  const origSubmit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function patched(
    this: GPUQueue, buffers: Iterable<GPUCommandBuffer>,
  ): undefined {
    submits += 1;
    origSubmit.call(this, buffers);
  };
}

const sameBits = (a: ArrayLike<number>, b: ArrayLike<number>): boolean => {
  if (a.length !== b.length) return false;
  const fa = Float32Array.from(a);
  const fb = Float32Array.from(b);
  const ua = new Uint32Array(fa.buffer);
  const ub = new Uint32Array(fb.buffer);
  return ua.every((v, i) => v === ub[i]);
};

const isZero = (a: ArrayLike<number>): boolean => Array.from(a).every((v) => v === 0);

// Inserts copies of ids[1 .. first) after the first token until the row has `target` tokens.
function lengthen(ids: number[], first: number, target: number): { ids: number[]; shift: number } {
  const unit = ids.slice(1, Math.max(first, 2));
  const extra: number[] = [];
  while (ids.length + extra.length < target) extra.push(unit[extra.length % unit.length]);
  return { ids: [ids[0], ...extra, ...ids.slice(1)], shift: extra.length };
}

async function juliaRows(sizes: number[], precision: string, result: Result): Promise<void> {
  const engine = await JuliaEngine.load({
    manifestUrl: `/models/julia-1/${precision}/manifest.json`,
    buckets: [512, 1024], precision: 'auto', limits: 'minimum', cacheSize: 0,
  });
  result.info = engine.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  const golden = (await (await fetch('/tests/golden/julia-1/parity100.json')).json()) as { items: JuliaItem[] };
  const rows: JuliaPreparedInput[] = golden.items.slice(0, 16).map((item, i) => {
    const target = 985 + i * 2;
    const { ids, shift } = lengthen(item.input_ids.slice(0, item.seq_len), item.markers[0], target);
    return {
      inputIds: Int32Array.from(ids), markers: item.markers.map((m) => m + shift),
      qtype: item.qtype, seqLen: ids.length,
    };
  });
  result.lengths = rows.map((r) => r.seqLen);
  result.sizes = {};
  const singles = [];
  for (const row of rows) singles.push(await engine.runPrepared(row, false, 1024));
  for (const B of sizes) {
    gpuErrors = []; maxX = 0; submits = 0;
    const batch = await engine.runPreparedBatch(rows.slice(0, B), 1024);
    let differing = 0; let zero = 0;
    for (const [i, out] of batch.entries()) {
      if (!sameBits(out.logits, singles[i].logits)) differing += 1;
      if (isZero(out.logits)) zero += 1;
    }
    result.sizes[`B${B}`] = {
      rows: B, differingRows: differing, maxWorkgroupsX: maxX, submits, gpuErrors: [...gpuErrors], allZero: zero };
    result.stage = `B${B} done`;
  }
  engine.dispose();
}

async function encoderRows(model: string, sizes: number[], precision: string, result: Result): Promise<void> {
  const dir = `/models/k28/${model}`;
  const manifest = (await (await fetch(`${dir}/${precision}/manifest.json`)).json()) as { task: string };
  const golden = (await (await fetch(`/tests/golden/k28/${model}/${manifest.task}.json`)).json()) as { items: EncItem[] };
  const enc = await EncoderModel.load({
    manifestUrl: `${dir}/${precision}/manifest.json`, precision: 'auto',
    buckets: [128, 512], limits: 'minimum',
  });
  result.info = enc.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  const rows = golden.items.slice(0, 16).map((item, i) => {
    const { ids } = lengthen(item.input_ids, item.input_ids.length - 1, 490 + i);
    return { inputIds: ids, typeIds: ids.map(() => 0) };
  });
  result.lengths = rows.map((r) => r.inputIds.length);
  result.sizes = {};
  const singles = [];
  for (const row of rows) singles.push(await enc.runIds(row, { bucket: 512 }));
  for (const B of sizes) {
    gpuErrors = []; maxX = 0; submits = 0;
    const batch = await enc.runIdsBatch(rows.slice(0, B), { bucket: 512 });
    let differing = 0; let zero = 0;
    for (const [i, out] of batch.entries()) {
      if (!sameBits(out.data, singles[i].data)) differing += 1;
      if (isZero(out.data)) zero += 1;
    }
    result.sizes[`B${B}`] = {
      rows: B, differingRows: differing, maxWorkgroupsX: maxX, submits, gpuErrors: [...gpuErrors], allZero: zero };
    result.stage = `B${B} done`;
  }
  enc.dispose();
}

async function main(): Promise<void> {
  watchDevice();
  const params = new URLSearchParams(location.search);
  const kind = params.get('kind') ?? 'julia';
  const model = params.get('model') ?? 'julia-1';
  const precision = params.get('precision') ?? 'f16';
  const sizes = (params.get('sizes') ?? '8,16').split(',').map(Number);
  const result: Result = { stage: 'boot', kind, model, precision };
  window.khK286bResult = result;
  try {
    if (kind === 'julia') await juliaRows(sizes, precision, result);
    else await encoderRows(model, sizes, precision, result);
    result.stage = 'done';
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
  }
  result.done = true;
}

void main();
