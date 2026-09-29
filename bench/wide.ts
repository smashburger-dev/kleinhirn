// Wide-schema parity: banking77 rows with all 72 labels in one schema,
// engine bucket {length:1280, markers:80} vs the PyTorch golden logits.
// Query: ?model=base-upstream&precision=f32
import { Kleinhirn } from '../src/index.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import { argmax, compareLogits } from './metrics.ts';

interface Item {
  example_id: string; seq_len: number;
  input_ids: number[]; attention_mask: number[];
  marker_indices: number[]; marker_mask: number[]; marker_groups: number[];
  logits: number[]; label_index: number; target_index: number;
}

interface Result {
  stage: string; model?: string; precision?: string;
  n?: number; argmaxAgreement?: number; maxAbsLogitDiff?: number;
  maxAbsProbDiff?: number; labelAgreement?: number;
  over?: number; info?: unknown; error?: string; done?: boolean;
}

declare global {
  interface Window { khWideResult?: Result }
}

function toInput(b: Item, k: number): SchemaInput {
  const markerIndices = new Int32Array(k);
  markerIndices.set(b.marker_indices.slice(0, k));
  const markerMask = new Float32Array(k);
  markerMask.set(b.marker_mask.slice(0, k));
  const markerGroups = new Int32Array(k);
  markerGroups.set(b.marker_groups.slice(0, k));
  return {
    inputIds: Int32Array.from(b.input_ids),
    attentionMask: Int32Array.from(b.attention_mask),
    markerIndices, markerMask, markerGroups,
    seqLen: b.seq_len,
  };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'base-upstream';
  const precision = params.get('precision') ?? 'f32';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const result: Result = { stage: 'boot', model, precision };
  window.khWideResult = result;
  try {
    const golden = await (await fetch(
      `/tests/golden/${model}/banking77_l1280k80.json`)).json() as {
      bucket: { length: number; max_options: number };
      items: Item[]; over_bucket: number; count: number;
    };
    const { length, max_options } = golden.bucket;
    const t0 = performance.now();
    const kh = await Kleinhirn.load({
      manifestUrl: `/models/${model}/${precision}/manifest.json`,
      buckets: [{ length, markers: max_options }],
      precision: 'auto',
      limits,
    });
    result.info = { ...kh.info(), loadMs: performance.now() - t0 };

    const ref: number[][] = [];
    const cand: number[][] = [];
    let agree = 0;
    let labelHits = 0;
    for (const [i, item] of golden.items.entries()) {
      const res = await kh.runPrepared(toInput(item, max_options));
      const nValid = item.marker_mask.filter((m) => m > 0.5).length;
      ref.push(item.logits.slice(0, nValid));
      const got = Array.from(res.logits.slice(0, nValid));
      cand.push(got);
      agree += argmax(item.logits) === argmax(got) ? 1 : 0;
      labelHits += argmax(item.logits) === item.target_index ? 1 : 0;
      result.stage = `wide ${i + 1}/${golden.items.length}`;
    }
    const cmp = compareLogits(ref, cand) as {
      argmaxAgreement: number; maxAbsLogitDiff: number; maxAbsProbDiff: number;
    };
    result.n = golden.count;
    result.over = golden.over_bucket;
    result.argmaxAgreement = agree / golden.count;
    result.maxAbsLogitDiff = cmp.maxAbsLogitDiff;
    result.maxAbsProbDiff = cmp.maxAbsProbDiff;
    result.labelAgreement = labelHits / golden.count;
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
