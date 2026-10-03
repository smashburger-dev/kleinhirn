// K28.8 shared browser code of the two measurement pages (k28-ort.ts and
// k28-latency.ts): the measurement inputs of convert/k28_8_inputs.py, the
// timer step, the accuracy on the golden cases up to L tokens (outside the
// measurement window) and the outputs of the measured calls for Festlegung 10.

import { argmax } from '../metrics.ts';

export interface InputMeta {
  model: string; task: string; length: number; rows: number; warmup: number;
  ids: string; idsSha256: string; typeIds: string; padId: number | null;
}

export interface Inputs { meta: InputMeta; ids: Int32Array<ArrayBuffer>; row(i: number): Int32Array<ArrayBuffer> }

const hex = (buf: ArrayBuffer): string =>
  [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

async function fetchOk(url: string): Promise<Response> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return res;
}

export async function loadInputs(slug: string, length: number): Promise<Inputs> {
  const dir = `/models/k28/${slug}/k28.8`;
  const meta = (await (await fetchOk(`${dir}/inputs-L${length}.json`)).json()) as InputMeta;
  if (meta.typeIds !== 'all zero') throw new Error('type ids other than zero are not wired');
  const buf = await (await fetchOk(`${dir}/${meta.ids}`)).arrayBuffer();
  if (hex(await crypto.subtle.digest('SHA-256', buf)) !== meta.idsSha256) {
    throw new Error(`sha256 mismatch on ${meta.ids}`);
  }
  const ids = new Int32Array(buf);
  if (ids.length !== meta.rows * length) throw new Error(`${meta.ids}: ${ids.length} ids`);
  return { meta, ids, row: (i) => ids.subarray(i * length, (i + 1) * length) };
}

// Smallest positive step of performance.now() over 200.000 reads.
export function timerStepMs(): number {
  let step = Infinity;
  let last = performance.now();
  for (let i = 0; i < 200000; i += 1) {
    const t = performance.now();
    if (t > last) { step = Math.min(step, t - last); last = t; }
  }
  return step;
}

export function isolation(): { crossOriginIsolated: boolean; timerStepMs: number } {
  return { crossOriginIsolated: self.crossOriginIsolated === true, timerStepMs: timerStepMs() };
}

export function allFinite(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i += 1) if (!Number.isFinite(a[i])) return false;
  return true;
}

// Rows of the measured calls, as base64 float32 (the runner writes the binary file).
export class OutputLog {
  private rows: Float32Array[] = [];
  nonFinite = 0;

  push(row: ArrayLike<number>): void {
    const copy = Float32Array.from(row);
    if (!allFinite(copy)) this.nonFinite += 1;
    this.rows.push(copy);
  }

  get count(): number { return this.rows.length; }

  base64(): string {
    const width = this.rows[0]?.length ?? 0;
    const all = new Float32Array(this.rows.length * width);
    this.rows.forEach((r, i) => all.set(r, i * width));
    const bytes = new Uint8Array(all.buffer);
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  get width(): number { return this.rows[0]?.length ?? 0; }
}

interface BinRef { file: string; name: string }
interface GoldenItem {
  input_ids: number[]; argmax?: number; logits?: number[]; logit?: number; query?: number;
}
interface Golden {
  items: GoldenItem[]; queries?: { query: number; best: number }[]; pooled?: BinRef;
}

const cosine = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let dot = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / Math.sqrt(na * nb);
};

// run(ids) gets the unpadded golden ids; each side pads (ORT) or not (kleinhirn) itself.
export async function goldenAccuracy(
  slug: string, task: string, length: number, run: (ids: number[]) => Promise<Float32Array>,
): Promise<Record<string, unknown>> {
  const golden = (await (await fetchOk(`/tests/golden/k28/${slug}/${task}.json`)).json()) as Golden;
  const use = golden.items.map((it, i) => [it, i] as const).filter(([it]) => it.input_ids.length <= length);
  const outs = new Map<number, Float32Array>();
  let finite = true;
  for (const [it, i] of use) {
    const o = await run(it.input_ids);
    if (!allFinite(o)) finite = false;
    outs.set(i, o);
  }
  const m: Record<string, unknown> = { cases: use.length, of: golden.items.length, finite };
  if (task === 'sequence-classification') {
    let agree = 0; let diff = 0;
    for (const [it, i] of use) {
      const o = outs.get(i) as Float32Array;
      if (argmax(Array.from(o)) === it.argmax) agree += 1;
      (it.logits as number[]).forEach((v, k) => { diff = Math.max(diff, Math.abs(o[k] - v)); });
    }
    m.argmaxAgreement = agree / use.length;
    m.maxAbsLogitDiff = diff;
  } else if (task === 'reranking') {
    const queries = (golden.queries ?? []).filter((q) => golden.items
      .every((it) => it.query !== q.query || it.input_ids.length <= length));
    let ok = 0; let diff = 0;
    for (const q of queries) {
      const rows = golden.items.map((it, i) => [it, i] as const).filter(([it]) => it.query === q.query);
      if (argmax(rows.map(([, i]) => (outs.get(i) as Float32Array)[0])) === q.best) ok += 1;
    }
    for (const [it, i] of use) diff = Math.max(diff, Math.abs((outs.get(i) as Float32Array)[0] - (it.logit as number)));
    m.queries = queries.length;
    m.bestPassageAgreement = ok / queries.length;
    m.maxAbsLogitDiff = diff;
  } else if (task === 'embeddings') {
    const dir = `/models/k28/${slug}/golden`;
    const index = (await (await fetchOk(`${dir}/index.json`)).json()) as {
      entries: { name: string; file: string; offset: number; bytes: number; shape: number[] }[];
    };
    const ref = golden.pooled as BinRef;
    const e = index.entries.find((x) => x.name === ref.name);
    if (!e) throw new Error(`golden entry ${ref.name} missing`);
    const buf = await (await fetchOk(`${dir}/${e.file}`)).arrayBuffer();
    const pooled = new Float32Array(buf.slice(e.offset, e.offset + e.bytes));
    const dim = e.shape[1];
    let minCos = 1; let diff = 0;
    for (const [, i] of use) {
      const o = outs.get(i) as Float32Array;
      const r = pooled.subarray(i * dim, (i + 1) * dim);
      minCos = Math.min(minCos, cosine(o, r));
      for (let k = 0; k < dim; k += 1) diff = Math.max(diff, Math.abs(o[k] - r[k]));
    }
    m.minCosine = minCos;
    m.maxAbsDiff = diff;
  } else {
    throw new Error(`task ${task}`);
  }
  return m;
}

// Download bytes of the resources whose URL matches, from resource timing.
export function resourceBytes(match: RegExp): { bytes: number; files: Record<string, number> } {
  const files: Record<string, number> = {};
  let bytes = 0;
  for (const e of performance.getEntriesByType('resource') as PerformanceResourceTiming[]) {
    if (!match.test(e.name)) continue;
    const n = e.transferSize || e.encodedBodySize || e.decodedBodySize || 0;
    files[new URL(e.name).pathname] = n;
    bytes += n;
  }
  return { bytes, files };
}
