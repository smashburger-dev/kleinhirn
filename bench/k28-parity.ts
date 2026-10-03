// K28.4 parity runner (browser side): every golden case of one pilot model
// through EncoderModel.runIds against the PyTorch goldens, then the same
// cases in groups of four through runIdsBatch against a single call in the
// same bucket (bit compare). Minimum limits, buckets [128, 512]. A WebGPU
// validation error ends the run: without the listener a wrong plan silently
// returns zeros.
// Query: ?model=<slug>&precision=f32|f16[&layers=1|&text=1]
// text=1 (K28.3): the same golden cases through the text methods. Every
// runIds / runIdsBatch call the methods make is recorded and compared with
// the golden ids (input mismatch) and, bit for bit, with the same call on the
// golden ids (G3); task results are compared with the goldens (G2 for spans).
// layers=1: f32 B1 capture run on the 8 layer-state cases, largest deviation
// after the embedding and after every layer (debugging tool, no gate).

import { EncoderModel } from '../src/index.ts';
import { argmaxRows, l2normalize, zeroShot } from '../src/tasks.ts';
import { compareLogits } from './metrics.ts';

interface BinRef { file: string; name: string }
interface Entry {
  bytes: number; dtype: string; file: string; name: string; offset: number; shape: number[];
}
interface Item {
  input_ids: number[];
  token_type_ids: number[];
  text_index?: number;
  text?: string;
  passage_text_index?: number;
  spans?: { entity_group: string; score: number; start: number; end: number }[];
  logits?: number[] | BinRef;
  logit?: number;
  argmax?: number | number[];
  query?: number;
  passage?: number;
}
interface Golden {
  task: string;
  bins: Record<string, string>;
  hidden: BinRef[];
  items: Item[];
  entail_index?: number;
  texts?: { text_index: number; zero_shot_choice: number }[];
  queries?: { query: number; best: number }[];
  pooled?: BinRef;
  final?: BinRef;
  prompt?: string | null;
  lengths?: { model: number };
}

interface Result {
  stage: string;
  model?: string;
  precision?: string;
  task?: string;
  info?: unknown;
  adapterInfo?: unknown;
  cases?: number;
  metrics?: Record<string, unknown>;
  g4?: { rows: number; differingRows: number; groups: number };
  layers?: { maxAbsDiff: number[]; worstLayer: number; cases: number };
  gpuErrors?: string[];
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khK28ParityResult?: Result }
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

class GoldenBins {
  private files = new Map<string, ArrayBuffer>();
  private entries = new Map<string, Entry>();

  private constructor(private base: string) {}

  static async open(base: string, golden: Golden): Promise<GoldenBins> {
    const g = new GoldenBins(base);
    const index = (await (await fetch(`${base}/index.json`)).json()) as { entries: Entry[] };
    for (const e of index.entries) g.entries.set(e.name, e);
    // Verify the sha256 of every binary the golden file points to.
    const used = new Set<string>();
    for (const ref of [...golden.hidden, golden.pooled, golden.final]) if (ref) used.add(ref.file);
    for (const item of golden.items) {
      if (item.logits && !Array.isArray(item.logits)) used.add(item.logits.file);
    }
    for (const file of used) {
      const res = await fetch(`${base}/${file}`);
      if (!res.ok) throw new Error(`fetch ${base}/${file}: ${res.status}`);
      const buf = await res.arrayBuffer();
      const digest = hex(await crypto.subtle.digest('SHA-256', buf));
      if (digest !== golden.bins[file]) throw new Error(`sha256 mismatch on golden ${file}`);
      g.files.set(file, buf);
    }
    return g;
  }

  array(ref: BinRef): { data: Float32Array; shape: number[] } {
    const e = this.entries.get(ref.name);
    const buf = this.files.get(ref.file);
    if (!e || !buf) throw new Error(`golden entry ${ref.name} missing`);
    if (e.dtype !== 'float32') throw new Error(`golden entry ${ref.name} is ${e.dtype}`);
    return { data: new Float32Array(buf.slice(e.offset, e.offset + e.bytes)), shape: e.shape };
  }
}

const cosine = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let dot = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
};

const bucketOf = (seqLen: number): number => (seqLen <= 128 ? 128 : 512);

const sameBits = (a: Float32Array, b: Float32Array): boolean => {
  if (a.length !== b.length) return false;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < ua.length; i += 1) if (ua[i] !== ub[i]) return false;
  return true;
};

type Input = { inputIds: number[]; typeIds: number[] };
type Out = { data: Float32Array };
interface Call { kind: 'ids' | 'batch'; inputs: Input[]; outs: Out[] }

const NLI_LABELS = ['politics', 'sports', 'technology', 'business', 'health'];

// UTF-16 index of a text to the code point index Python reports.
const toCodePoint = (text: string, u16: number): number => [...text.slice(0, u16)].length;

interface TextStats {
  cases: number; calls: number; inputMismatch: number; rows: number; bitDiffRows: number;
  batchRows: number; batchVsSingleDiffRows: number;
}

async function runText(
  enc: EncoderModel, golden: Golden, bins: GoldenBins, task: string, result: Result,
): Promise<void> {
  const corpusDoc = await (await fetch('/tests/corpus/texts1000.json')).json();
  const corpus: string[] = Array.isArray(corpusDoc) ? corpusDoc : corpusDoc.texts;
  const inputOf = (i: Item): Input => ({ inputIds: i.input_ids, typeIds: i.token_type_ids });
  const origIds = enc.runIds.bind(enc);
  const origBatch = enc.runIdsBatch.bind(enc);
  let rec: Call[] = [];
  enc.runIds = async (input, options) => {
    const out = await origIds(input, options);
    rec.push({ kind: 'ids', inputs: [input as Input], outs: [out] });
    return out;
  };
  enc.runIdsBatch = async (inputs, options) => {
    const outs = await origBatch(inputs, options);
    rec.push({ kind: 'batch', inputs: inputs as Input[], outs });
    return outs;
  };
  const stats: TextStats = {
    cases: 0, calls: 0, inputMismatch: 0, rows: 0, bitDiffRows: 0, batchRows: 0,
    batchVsSingleDiffRows: 0,
  };
  const sameInput = (a: Input, b: Input): boolean => a.inputIds.length === b.inputIds.length
    && a.inputIds.every((v, k) => v === b.inputIds[k])
    && a.typeIds.length === b.typeIds.length && a.typeIds.every((v, k) => v === b.typeIds[k]);
  // expected: the golden inputs of each engine call the method should make
  const check = async (expected: Input[][]): Promise<void> => {
    const calls = rec;
    rec = [];
    stats.cases += 1;
    stats.calls += calls.length;
    if (calls.length !== expected.length) { stats.inputMismatch += 1; return; }
    for (const [c, call] of calls.entries()) {
      const want = expected[c];
      if (call.inputs.length !== want.length
        || !call.inputs.every((inp, k) => sameInput(inp, want[k]))) {
        stats.inputMismatch += 1;
        continue;
      }
      // Same call on the golden ids, bit for bit.
      const again = call.kind === 'ids' ? [await origIds(want[0])] : await origBatch(want);
      for (const [k, out] of call.outs.entries()) {
        stats.rows += 1;
        if (!sameBits(out.data, again[k].data)) stats.bitDiffRows += 1;
      }
      // Information: rows of a batch call against single calls.
      if (call.kind === 'batch') {
        for (const [k, out] of call.outs.entries()) {
          stats.batchRows += 1;
          if (!sameBits(out.data, (await origIds(want[k])).data)) stats.batchVsSingleDiffRows += 1;
        }
      }
    }
    checkErrors();
  };

  const items = golden.items;
  const m: Record<string, unknown> = {};
  if (task === 'sequence-classification') {
    let agree = 0;
    const choices: number[] = []; // decisions of the text path, for convert/k28_text_reference.py
    for (const [i, item] of items.entries()) {
      const r = await enc.classify(item.text ?? corpus[item.text_index as number]);
      choices.push(r.index);
      if (r.index === item.argmax) agree += 1;
      await check([[inputOf(item)]]);
      if (i % 50 === 0) { result.stage = `text ${i}/${items.length}`; checkErrors(); }
    }
    m.argmaxAgreement = agree / items.length;
    m.choices = choices;
  } else if (task === 'nli') {
    let ok = 0;
    const choices: number[] = [];
    const texts = golden.texts as NonNullable<Golden['texts']>;
    for (const [i, t] of texts.entries()) {
      const r = await enc.zeroShot(corpus[t.text_index], NLI_LABELS);
      choices.push(r.index);
      if (r.index === t.zero_shot_choice) ok += 1;
      await check([items.slice(t.text_index * 5, t.text_index * 5 + 5).map(inputOf)]);
      if (i % 10 === 0) { result.stage = `text ${i}/${texts.length}`; checkErrors(); }
    }
    m.zeroShotAgreement = ok / texts.length;
    m.zeroShotTexts = texts.length;
    m.choices = choices;
  } else if (task === 'reranking') {
    let ok = 0;
    const choices: number[] = [];
    const queries = golden.queries as NonNullable<Golden['queries']>;
    for (const q of queries) {
      const rows = items.filter((it) => it.query === q.query);
      const r = await enc.rerank(
        corpus[rows[0].text_index as number],
        rows.map((it) => corpus[it.passage_text_index as number]));
      choices.push(r.order[0]);
      if (r.order[0] === q.best) ok += 1;
      await check([rows.map(inputOf)]);
    }
    m.bestPassageAgreement = ok / queries.length;
    m.queries = queries.length;
    m.choices = choices;
  } else if (task === 'embeddings') {
    const finalRef = bins.array(golden.final as BinRef).data;
    const texts = items.map((it) => corpus[it.text_index as number]);
    const vecs = await enc.embed(texts, { prompt: golden.prompt ?? undefined });
    const dim = vecs[0].length;
    let minCos = 1; let sumCos = 0; let maxDiff = 0;
    for (const [i, v] of vecs.entries()) {
      const ref = finalRef.subarray(i * dim, (i + 1) * dim);
      const c = cosine(v, ref);
      minCos = Math.min(minCos, c); sumCos += c;
      for (let k = 0; k < dim; k += 1) maxDiff = Math.max(maxDiff, Math.abs(v[k] - ref[k]));
    }
    // embed sorts by length and chunks internally: expected inputs are in call order
    await check([items.map(inputOf)]);
    m.minCosineFinal = minCos;
    m.meanCosineFinal = sumCos / vecs.length;
    m.maxAbsDiffFinal = maxDiff;
  } else if (task === 'token-classification') {
    let textsEqual = 0; let spans = 0; let maxScore = 0; let firstBad: unknown = null;
    // the spans of every text, in code points like Python, for convert/k28_span_reference.py
    const spansGot: unknown[] = [];
    for (const [i, item] of items.entries()) {
      const text = corpus[item.text_index as number];
      const got = await enc.tokenClassify(text);
      spansGot.push(got.map((g) => ({
        entity_group: g.group, score: g.score, start: toCodePoint(text, g.start), end: toCodePoint(text, g.end) })));
      const want = item.spans as NonNullable<Item['spans']>;
      let equal = got.length === want.length;
      spans += want.length;
      for (let k = 0; equal && k < want.length; k += 1) {
        const g = got[k];
        const d = Math.abs(g.score - want[k].score);
        maxScore = Math.max(maxScore, d);
        if (g.group !== want[k].entity_group || toCodePoint(text, g.start) !== want[k].start
          || toCodePoint(text, g.end) !== want[k].end || d > 1e-4) equal = false;
      }
      if (equal) textsEqual += 1;
      else if (!firstBad) firstBad = { text_index: item.text_index, got, want };
      await check([[inputOf(item)]]);
      if (i % 25 === 0) { result.stage = `text ${i}/${items.length}`; checkErrors(); }
    }
    m.spanExactAgreement = textsEqual / items.length;
    m.spansGot = spansGot;
    m.spanTexts = items.length;
    m.spans = spans;
    m.maxAbsScoreDiff = maxScore;
    if (firstBad) m.firstDifferingText = firstBad;
  } else {
    throw new Error(`task ${task} has no text comparison`);
  }
  enc.runIds = origIds;
  enc.runIdsBatch = origBatch;
  m.textStats = stats;
  result.metrics = m;
}

async function main(): Promise<void> {
  watchDevices();
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? '';
  const precision = params.get('precision') ?? 'f32';
  const layers = params.get('layers') === '1';
  const result: Result = { stage: 'boot', model, precision };
  window.khK28ParityResult = result;
  try {
    const dir = `/models/k28/${model}`;
    const manifest = (await (await fetch(`${dir}/${precision}/manifest.json`)).json()) as {
      task: string; head: { type: string; classes?: number; normalize?: boolean; steps?: unknown[] };
    };
    const task = manifest.task;
    result.task = task;
    const golden = (await (await fetch(
      `/tests/golden/k28/${model}/${task}.json`)).json()) as Golden;
    const bins = await GoldenBins.open(`${dir}/golden`, golden);
    const enc = await EncoderModel.load({
      manifestUrl: `${dir}/${precision}/manifest.json`, precision: 'auto',
      buckets: [128, 512], limits: 'minimum',
    });
    checkErrors();
    result.info = enc.info();
    result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
    const items = golden.items;
    result.cases = items.length;
    const inputOf = (i: Item) => ({ inputIds: i.input_ids, typeIds: i.token_type_ids });

    if (params.get('text') === '1') {
      const textMax = (result.info as { textMaxLength: number }).textMaxLength;
      // the golden length is the model's own limit; the engine's largest bucket is 512
      // (d0rj/e5-small-en-ru: max_seq_length 514 in sentence_bert_config.json)
      const wantMax = Math.min(golden.lengths?.model ?? 0, 512);
      if (textMax !== wantMax) {
        throw new Error(`textMaxLength ${textMax} differs from the golden length ${golden.lengths?.model} (bucket cap 512)`);
      }
      await runText(enc, golden, bins, task, result);
      result.stage = 'done';
      result.gpuErrors = [...gpuErrors];
      enc.dispose();
      result.done = true;
      return;
    }

    if (layers) {
      if (precision !== 'f32') throw new Error('layers=1 is an f32 run');
      const worst: number[] = [];
      for (const [i, ref] of golden.hidden.entries()) {
        const want = bins.array(ref);
        const [n, len, hidden] = want.shape;
        const out = await enc.runIds(inputOf(items[i]), { capture: true });
        const cap = out.capture as Float32Array;
        const slot = out.captureSlotElements as number;
        for (let s = 0; s < n; s += 1) {
          let d = 0;
          for (let r = 0; r < len; r += 1) {
            for (let c = 0; c < hidden; c += 1) {
              d = Math.max(d, Math.abs(
                cap[s * slot + r * hidden + c] - want.data[(s * len + r) * hidden + c]));
            }
          }
          worst[s] = Math.max(worst[s] ?? 0, d);
        }
        checkErrors();
      }
      const w = worst.indexOf(Math.max(...worst));
      result.layers = { maxAbsDiff: worst, worstLayer: w, cases: golden.hidden.length };
      result.stage = 'done';
      result.gpuErrors = [...gpuErrors];
      enc.dispose();
      result.done = true;
      return;
    }

    // Single calls, one per golden case.
    const outs: { data: Float32Array; rows: number; cols: number; seqLen: number }[] = [];
    for (const [i, item] of items.entries()) {
      outs.push(await enc.runIds(inputOf(item)));
      if (i % 50 === 0) { result.stage = `single ${i}/${items.length}`; checkErrors(); }
    }
    checkErrors();

    const m: Record<string, unknown> = {};
    if (task === 'sequence-classification' || task === 'nli') {
      const ref = items.map((i) => i.logits as number[]);
      const cand = outs.map((o) => Array.from(o.data));
      const p = compareLogits(ref, cand);
      m.argmaxAgreement = p.argmaxAgreement;
      m.maxAbsLogitDiff = p.maxAbsLogitDiff;
      m.meanAbsLogitDiff = p.meanAbsLogitDiff;
      m.disagreements = p.disagreements;
      if (task === 'nli') {
        const ent = golden.entail_index as number;
        let ok = 0;
        const zeroShotDisagreements: number[] = []; // positions in golden.texts
        for (const [ti, t] of (golden.texts as NonNullable<Golden['texts']>).entries()) {
          const logits = [0, 1, 2, 3, 4].map((j) => outs[t.text_index * 5 + j].data[ent]);
          if (zeroShot(logits).choice === t.zero_shot_choice) ok += 1;
          else zeroShotDisagreements.push(ti);
        }
        m.zeroShotAgreement = ok / (golden.texts as unknown[]).length;
        m.zeroShotDisagreements = zeroShotDisagreements;
        m.zeroShotTexts = (golden.texts as unknown[]).length;
      }
    } else if (task === 'reranking') {
      const ref = items.map((i) => [i.logit as number]);
      const cand = outs.map((o) => Array.from(o.data));
      const p = compareLogits(ref, cand);
      m.maxAbsLogitDiff = p.maxAbsLogitDiff;
      m.meanAbsLogitDiff = p.meanAbsLogitDiff;
      const scores = new Map<number, number[]>();
      items.forEach((it, i) => {
        const q = it.query as number;
        const list = scores.get(q) ?? [];
        list[it.passage as number] = outs[i].data[0];
        scores.set(q, list);
      });
      let ok = 0;
      const bestPassageDisagreements: number[] = []; // positions in golden.queries
      for (const [qi, q] of (golden.queries as NonNullable<Golden['queries']>).entries()) {
        if (argmaxRows(scores.get(q.query) as number[], (scores.get(q.query) as number[]).length)[0]
          === q.best) ok += 1;
        else bestPassageDisagreements.push(qi);
      }
      m.bestPassageAgreement = ok / (golden.queries as unknown[]).length;
      m.bestPassageDisagreements = bestPassageDisagreements;
      m.queries = (golden.queries as unknown[]).length;
    } else if (task === 'token-classification') {
      let agree = 0; let total = 0; let maxDiff = 0; let sumDiff = 0; let count = 0;
      const tokenDisagreements: number[] = []; // flat token index over all items in order
      for (const [i, item] of items.entries()) {
        const want = bins.array(item.logits as BinRef).data;
        const o = outs[i];
        const got = argmaxRows(o.data, o.cols);
        const ref = item.argmax as number[];
        for (let t = 0; t < ref.length; t += 1) {
          if (got[t] === ref[t]) agree += 1; else tokenDisagreements.push(total);
          total += 1;
        }
        for (let k = 0; k < want.length; k += 1) {
          const d = Math.abs(want[k] - o.data[k]);
          maxDiff = Math.max(maxDiff, d); sumDiff += d; count += 1;
        }
      }
      m.argmaxPerTokenAgreement = agree / total;
      m.argmaxPerTokenDisagreements = tokenDisagreements;
      m.tokens = total;
      m.maxAbsLogitDiff = maxDiff;
      m.meanAbsLogitDiff = sumDiff / count;
    } else if (task === 'embeddings') {
      const pooledRef = bins.array(golden.pooled as BinRef).data;
      const finalRef = bins.array(golden.final as BinRef).data;
      const dim = outs[0].cols;
      const normalize = manifest.head.normalize === true;
      const hasSteps = (manifest.head.steps ?? []).length > 0;
      let minPooled = 1; let sumPooled = 0; let minFinal = 1; let sumFinal = 0;
      let maxDiff = 0;
      for (const [i, o] of outs.entries()) {
        const refF = finalRef.subarray(i * dim, (i + 1) * dim);
        const fin = normalize ? l2normalize(o.data) : o.data;
        const cf = cosine(fin, refF);
        minFinal = Math.min(minFinal, cf); sumFinal += cf;
        for (let k = 0; k < dim; k += 1) maxDiff = Math.max(maxDiff, Math.abs(fin[k] - refF[k]));
        if (!hasSteps) {
          const cp = cosine(o.data, pooledRef.subarray(i * dim, (i + 1) * dim));
          minPooled = Math.min(minPooled, cp); sumPooled += cp;
        }
      }
      m.minCosineFinal = minFinal;
      m.meanCosineFinal = sumFinal / outs.length;
      m.maxAbsDiffFinal = maxDiff;
      if (!hasSteps) {
        m.minCosinePooled = minPooled;
        m.meanCosinePooled = sumPooled / outs.length;
      }
    } else {
      throw new Error(`task ${task} has no comparison`);
    }
    result.metrics = m;
    result.stage = 'metrics done';

    // G4: groups of four through runIdsBatch against single calls in the
    // same bucket, bit for bit.
    let rows = 0; let differing = 0; let groups = 0;
    for (let start = 0; start + 4 <= items.length; start += 4) {
      const group = items.slice(start, start + 4);
      const bucket = bucketOf(Math.max(...group.map((g) => g.input_ids.length)));
      const batch = await enc.runIdsBatch(group.map(inputOf), { bucket });
      for (const [i, item] of group.entries()) {
        const single = await enc.runIds(inputOf(item), { bucket });
        rows += 1;
        if (!sameBits(single.data, batch[i].data)) differing += 1;
      }
      groups += 1;
      if (groups % 10 === 0) { result.stage = `batch ${groups}`; checkErrors(); }
    }
    checkErrors();
    result.g4 = { rows, differingRows: differing, groups };
    result.gpuErrors = [...gpuErrors];
    result.stage = 'done';
    enc.dispose();
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
  }
  result.done = true;
}

void main();
