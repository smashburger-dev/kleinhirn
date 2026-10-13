// R4 Stufe T: time from tokenizer bytes to a usable tokenizer (Node, median of 5).
//
// Default (dirs as arguments): per directory, JsonTokenizer path: gunzip, decode, ctor, total.
// --survey: every models/k28/*/f16/tokenizer.bin (identical files by sha256 counted once) on the
//   JsonTokenizer path, plus the engine paths of models/small-upstream and models/multi-upstream
//   (GLiNER: HfTokenizer) and models/julia-1 (Julia: BpeTokenizer), built as the engines build
//   them: parseTokenizerFile, then the engine tokenizer (engineTotal).
// --encode (implied by --survey): encode time of the first 2000 lines of agnews.jsonl ("text")
//   and massive-de-DE.jsonl ("utt"), encodeJson for the JsonTokenizer, encodeEngine for the engines.
// --json <file> writes the survey rows and sums.
// Decoder under test: decodeKhtCompact when bin.ts exports it, else decodeKht.
// Usage: node --expose-gc --import ./tests/helpers/node-hooks.mjs tools/r4_tokload.ts [--survey] [--encode] [--json file] [dirs...]
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import * as bin from '../src/tokenizer/bin.ts';
import { JsonTokenizer } from '../src/tokenizer/hf/index.ts';
import { HfTokenizer } from '../src/tokenizer/tokenizer.ts';
import { BpeTokenizer } from '../src/tokenizer/bpe.ts';

const args = process.argv.slice(2);
const flag = (name: string) => args.includes(name);
const jsonOut = flag('--json') ? args[args.indexOf('--json') + 1] : null;
const positional = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--json');
const survey = flag('--survey');
const encode = flag('--encode') || survey;

const RUNS = 5;
const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const r2 = (x: number) => Number(x.toFixed(2));
const time = (f: () => unknown) => { const t = performance.now(); const v = f(); return [performance.now() - t, v] as const; };
const parseEngine: (p: Uint8Array) => Promise<Record<string, unknown>> = bin.parseTokenizerCompact ?? bin.parseTokenizerFile;
const decodeUnderTest = ((bin as Record<string, unknown>).decodeKhtCompact ?? bin.decodeKht) as (p: Uint8Array) => Record<string, unknown>;

async function measureJson(file: Buffer) {
  // a collection before each measurement: earlier tokenizers of the survey are garbage by now
  (globalThis as { gc?: () => void }).gc?.();
  const rows: number[][] = [];
  for (let i = 0; i < RUNS; i += 1) {
    const [g, raw] = time(() => gunzipSync(file));
    const [d, obj] = time(() => decodeUnderTest(new Uint8Array(raw as Buffer)));
    const [c] = time(() => JsonTokenizer.fromJson(obj as Record<string, unknown>));
    rows.push([g, d, c, g + d + c]);
  }
  const col = (k: number) => r2(med(rows.map((r) => r[k])));
  return { gunzip: col(0), decode: col(1), ctor: col(2), total: col(3) };
}

async function measureEngine(file: Buffer, kind: 'hf' | 'bpe') {
  (globalThis as { gc?: () => void }).gc?.();
  const rows: number[] = [];
  for (let i = 0; i < RUNS; i += 1) {
    const t = performance.now();
    const obj = await parseEngine(new Uint8Array(file));
    if (kind === 'hf') new HfTokenizer(obj as never); else new BpeTokenizer(obj as never);
    rows.push(performance.now() - t);
  }
  return r2(med(rows));
}

function measureEncode(texts: string[], enc: (t: string) => unknown): number {
  const rows: number[] = [];
  for (let i = 0; i < RUNS; i += 1) {
    const [t] = time(() => { for (const x of texts) enc(x); });
    rows.push(t);
  }
  return r2(med(rows));
}

const linesOf = (file: string, field: string): string[] => readFileSync(file, 'utf8')
  .split('\n').filter(Boolean).slice(0, 2000).map((l) => String(JSON.parse(l)[field]));

if (!survey) {
  const dirs = positional.length ? positional : ['models/k28/cross-encoder__mmarco-mMiniLMv2-L12-H384-v1/f16'];
  for (const d of dirs) {
    const m = await measureJson(readFileSync(`${d}/tokenizer.bin`));
    console.log(d.replace(/\/f(16|32)\/?$/, '').split('/').at(-1)!.padEnd(52), JSON.stringify(m));
  }
  process.exit(0);
}

const TASKS = 'models/k9-data/tasks';
const encodeTexts = [...linesOf(`${TASKS}/agnews.jsonl`, 'text'), ...linesOf(`${TASKS}/massive-de-DE.jsonl`, 'utt')];

// k28: identical tokenizer.bin files (by sha256) are measured once, under the first name.
const seen = new Set<string>();
const rows: Record<string, unknown>[] = [];
const k28 = readdirSync('models/k28').filter((n) => existsSync(`models/k28/${n}/f16/tokenizer.bin`)).sort();
let distinct = 0;
for (const name of k28) {
  const file = readFileSync(`models/k28/${name}/f16/tokenizer.bin`);
  const hash = createHash('sha256').update(file).digest('hex');
  if (seen.has(hash)) continue;
  seen.add(hash);
  distinct += 1;
  const row: Record<string, unknown> = { name, path: `models/k28/${name}/f16`, kind: 'json', ...(await measureJson(file)) };
  if (encode) {
    const tok = JsonTokenizer.fromJson(decodeUnderTest(new Uint8Array(gunzipSync(file))));
    row.encodeJson = measureEncode(encodeTexts, (t) => tok.encode(t));
  }
  rows.push(row);
}
console.error(`k28: ${k28.length} tokenizer.bin files, ${distinct} distinct`);

// Engine paths: GLiNER (HfTokenizer) and Julia (BpeTokenizer).
const engines: [string, 'hf' | 'bpe'][] = [
  ['models/small-upstream/f16', 'hf'], ['models/multi-upstream/f16', 'hf'], ['models/julia-1/f16', 'bpe'],
];
for (const [dir, kind] of engines) {
  const file = readFileSync(`${dir}/tokenizer.bin`);
  const row: Record<string, unknown> = { name: dir.split('/')[1], path: dir, kind: kind === 'hf' ? 'gliner' : 'julia' };
  Object.assign(row, await measureJson(file));
  row.engineTotal = await measureEngine(file, kind);
  if (encode) {
    const obj = await parseEngine(new Uint8Array(file));
    const eng = kind === 'hf' ? new HfTokenizer(obj as never) : new BpeTokenizer(obj as never);
    row.encodeEngine = measureEncode(encodeTexts, (t) => eng.encodeIds(t));
    const js = JsonTokenizer.fromJson(decodeUnderTest(new Uint8Array(gunzipSync(file))));
    row.encodeJson = measureEncode(encodeTexts, (t) => js.encode(t));
  }
  rows.push(row);
}

const sum = (k: string) => r2(rows.reduce((a, r) => a + Number(r[k] ?? 0), 0));
const sums = { total: sum('total'), engineTotal: sum('engineTotal'), encodeJson: sum('encodeJson'), encodeEngine: sum('encodeEngine') };
for (const r of rows) console.log(String(r.name).padEnd(52), JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k]) => k !== 'name' && k !== 'path'))));
console.log('sums', JSON.stringify(sums));
if (jsonOut) writeFileSync(jsonOut, `${JSON.stringify({ rows, sums }, null, 2)}\n`);
