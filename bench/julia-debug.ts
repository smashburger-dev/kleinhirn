// Julia parity debug: per-slot max |capture - ref64| on one layer case so
// the first diverging stage is visible. ?precision=f32&case=0

import { JuliaEngine } from '../src/julia.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';

const HIDDEN = 384;

declare global {
  interface Window { khJuliaDebug?: unknown }
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const precision = params.get('precision') ?? 'f32';
  const caseIdx = Number(params.get('case') ?? '0');
  const out: Record<string, unknown> = { stage: 'boot' };
  window.khJuliaDebug = out;
  try {
    const kh = await JuliaEngine.load({
      manifestUrl: `/models/julia-1/${precision}/manifest.json`,
      buckets: [512, 1024],
      precision: 'auto',
      limits: 'minimum',
    });
    const golden = (await (await fetch(
      '/tests/golden/julia-1/parity100.json')).json()) as {
        items: {
          seq_len: number; input_ids: number[]; markers: number[];
          qtype: number; logits: number[] }[] };
    const idx = (await (await fetch(
      '/models/julia-1/golden/layers.index.json')).json()) as {
        cases: { case: number; source_index: number; seq_len: number }[];
        tensors: Record<string, { offset: number; shape: number[] }> };
    const blob = await (await fetch(
      '/models/julia-1/golden/layers.bin')).arrayBuffer();
    const get = (name: string) => {
      const d = idx.tensors[name];
      return new Float64Array(
        blob, d.offset, d.shape.reduce((a, b) => a * b, 1));
    };
    const c = idx.cases[caseIdx];
    const item = golden.items[c.source_index];
    const input: JuliaPreparedInput = {
      inputIds: Int32Array.from(item.input_ids),
      markers: item.markers, qtype: item.qtype, seqLen: item.seq_len,
    };
    const forced = params.get('bucket');
    const L = forced ? Number(forced)
      : item.seq_len <= 512 ? 512 : 1024;
    const res = await kh.runPrepared(input, true, L);
    const cap = res.captureData as Float32Array;
    const names = Object.keys(idx.tensors)
      .filter((k) => k.startsWith(`case${c.case}.`))
      .map((k) => k.split('.')[1]);
    const order = [
      'emb',
      ...Array.from({ length: 22 }, (_, l) => `layer${l}`),
      'final', 'typed', 'head0', 'head1', 'markers', 'logits',
    ];
    const slots: Record<string, number> = {};
    const rows = c.seq_len;
    const cols = (name: string) =>
      name === 'logits' ? item.logits.length : HIDDEN;
    for (const [s, name] of order.entries()) {
      if (!names.includes(name)) continue;
      const ref = get(`case${c.case}.${name}`);
      if (name === 'markers' || name === 'logits') {
        // markers/logits are not in the capture buffer; skip.
        continue;
      }
      const got = cap.subarray(s * L * HIDDEN, s * L * HIDDEN + rows * cols(name));
      let mx = 0;
      let at = -1;
      for (let i = 0; i < ref.length && i < got.length; i += 1) {
        const d = Math.abs(got[i] - ref[i]);
        if (d > mx) { mx = d; at = i; }
      }
      slots[name] = Number(mx.toExponential(3));
      out[`${name}@`] = at;
    }
    out.logits = {
      got: Array.from(res.logits.slice(0, item.logits.length)),
      ref: item.logits,
    };
    out.slots = slots;
    out.stage = 'done';
    kh.dispose();
  } catch (e) {
    out.stage = 'error';
    out.error = String(e);
  }
}

void main();
