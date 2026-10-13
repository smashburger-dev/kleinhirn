// R4 Stufe 4 load bench (browser side): time from EncoderModel.load to the first result,
// then single calls on further golden cases, which stand in for new texts (rows mode fetches
// their word rows on the call). Result on window.khR4Load, read by bench/run-r4-load.mjs.
// Query: ?model=<slug>&precision=f16|f32[&words=full|rows|progressive][&calls=20][&warm=0]
// warm: rounds over the same cases again after the new-text calls; every row is then in memory (G6).
// words is passed to the engine only when given: the pre-R4 engine (adac8a0) has no such option.

import { EncoderModel } from '../src/index.ts';

interface Item { input_ids: number[]; token_type_ids: number[] }
interface Golden { items: Item[] }
interface Bytes { models: number; engine: number; other: number; total: number }
interface R4Load {
  done: boolean;
  error?: string;
  model: string;
  precision: string;
  words?: unknown;
  loadMs?: number;
  firstCallMs?: number;
  ttfrMs?: number;
  callMs?: number[];
  warmMs?: number[];
  loadTiming?: Record<string, number>;
  info?: Record<string, unknown>;
  bytes?: Bytes;
  buildId?: string;
}

const params = new URLSearchParams(location.search);
const model = params.get('model') ?? '';
const precision = params.get('precision') ?? 'f16';
const wordsParam = params.get('words');
const calls = Number(params.get('calls') ?? 20);
const warm = Number(params.get('warm') ?? 0);

const result: R4Load = { done: false, model, precision };
(window as unknown as { khR4Load: R4Load }).khR4Load = result;

// Transfer bytes per resource class; entries from the HTTP cache count 0, the runner uses a fresh context.
function byteSplit(): Bytes {
  const out: Bytes = { models: 0, engine: 0, other: 0, total: 0 };
  for (const e of performance.getEntriesByType('resource') as PerformanceResourceTiming[]) {
    const n = e.transferSize || e.encodedBodySize || 0;
    if (e.name.includes('/models/')) out.models += n;
    else if (/\/(src|dist|node_modules)\//.test(e.name)) out.engine += n;
    else out.other += n;
    out.total += n;
  }
  return out;
}

async function main(): Promise<void> {
  const dir = `/models/k28/${model}`;
  const manifest = (await (await fetch(`${dir}/${precision}/manifest.json`)).json()) as { task: string };
  const golden = (await (await fetch(`/tests/golden/k28/${model}/${manifest.task}.json`)).json()) as Golden;
  const inputOf = (i: Item) => ({ inputIds: i.input_ids, typeIds: i.token_type_ids });

  const t0 = performance.now();
  const enc = await EncoderModel.load({
    manifestUrl: `${dir}/${precision}/manifest.json`, precision: precision as 'f16' | 'f32',
    buckets: [128, 512], limits: 'minimum',
    ...(wordsParam ? { words: wordsParam as 'full' | 'rows' | 'progressive' } : {}),
  }) as EncoderModel;
  const loadMs = performance.now() - t0;

  const t1 = performance.now();
  await enc.runIds(inputOf(golden.items[0]));
  const t2 = performance.now();

  const callMs: number[] = [];
  for (let k = 1; k <= calls; k += 1) {
    const item = golden.items[k % golden.items.length];
    const a = performance.now();
    await enc.runIds(inputOf(item));
    callMs.push(performance.now() - a);
  }

  const warmMs: number[] = [];
  for (let r = 0; r < warm; r += 1) {
    for (let k = 0; k <= calls; k += 1) {
      const a = performance.now();
      await enc.runIds(inputOf(golden.items[k % golden.items.length]));
      warmMs.push(performance.now() - a);
    }
  }

  const info = enc.info();
  result.warmMs = warmMs;
  result.loadMs = loadMs;
  result.firstCallMs = t2 - t1;
  result.ttfrMs = t2 - t0;
  result.callMs = callMs;
  result.loadTiming = info.loadTiming as Record<string, number>;
  result.info = info;
  result.words = info.words;
  result.buildId = info.buildId as string | undefined;
  result.bytes = byteSplit();
  result.done = true;
}

main().catch((e: unknown) => {
  result.error = String(e instanceof Error ? e.message : e).slice(0, 400);
  result.done = true;
});
