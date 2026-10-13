// R6 stage 4: tokenizer.json to tokenizer.bin (format in src/tokenizer/bin.ts). Every file is
// decoded right after encoding and must give an object deepStrictEqual to JSON.parse of the source
// (scores compared with Object.is); otherwise the tool stops.
// Usage: node tools/r6_tokbin.ts <tokenizer.json> [<out tokenizer.bin>]
//        node tools/r6_tokbin.ts --survey   every tokenizer under models/ in memory, sizes to
//                                           bench/results/r6-tokbin-survey.json
//        node tools/r6_tokbin.ts --migrate  every converted model folder under models/: tokenizer.bin
//                                           next to its tokenizer.json, the manifest names tokenizer.bin
// tokenizer.json stays in the folder: the ORT side of the bench (bench/kbench/adapters/ort.ts) and
// tools/vocab_locality.py read it. Engines before R6 cannot read these folders any more.

import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import { decodeKht, f64Step, KHT_MAGIC, type KhtField } from '../src/tokenizer/bin.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const enc = new TextEncoder();

class Writer {
  private chunks: number[] = [];
  varint(v: number): void {
    do {
      let b = v % 128;
      v = Math.floor(v / 128);
      if (v > 0) b |= 0x80;
      this.chunks.push(b);
    } while (v > 0);
  }
  bytes(b: Uint8Array): void { for (const x of b) this.chunks.push(x); }
  done(): Uint8Array { return Uint8Array.from(this.chunks); }
}

const PREFIXES = ['▁', 'Ġ', '##'];
function pickPrefix(list: string[]): string {
  let best = ''; let count = list.length / 10;
  for (const p of PREFIXES) {
    const n = list.filter((s) => s.startsWith(p)).length;
    if (n > count) { best = p; count = n; }
  }
  return best;
}

function writeStrings(list: string[], prefix: string): Uint8Array {
  const w = new Writer();
  for (const s of list) {
    const has = prefix !== '' && s.startsWith(prefix);
    const body = enc.encode(has ? s.slice(prefix.length) : s);
    w.varint(body.length * 2 + (has ? 1 : 0));
    w.bytes(body);
  }
  return w.done();
}

function writeVarints(list: number[]): Uint8Array {
  const w = new Writer();
  for (const v of list) w.varint(v);
  return w.done();
}

function writeScores(scores: number[]): Uint8Array[] {
  const n = scores.length;
  const planes = new Uint8Array(4 * n);
  const kinds = new Uint8Array(n);
  const extra: number[] = [];
  const f32 = new DataView(new ArrayBuffer(4));
  for (const [i, v] of scores.entries()) {
    const f = Math.fround(v);
    f32.setFloat32(0, f, true);
    for (let b = 0; b < 4; b += 1) planes[b * n + i] = f32.getUint8(b);
    if (Object.is(f, v)) continue;
    if (f !== 0 && Number.isFinite(f) && f64Step(f, true) === v) kinds[i] = 1;
    else if (f !== 0 && Number.isFinite(f) && f64Step(f, false) === v) kinds[i] = 2;
    else { kinds[i] = 3; extra.push(v); }
  }
  const ex = new DataView(new ArrayBuffer(8 * extra.length));
  extra.forEach((v, i) => ex.setFloat64(8 * i, v, true));
  return [planes, kinds, new Uint8Array(ex.buffer)];
}

type Json = Record<string, unknown>;

// Paths of every Precompiled charsmap inside the normalizer (it may sit in a Sequence).
function charsmapPaths(node: unknown, path: (string | number)[], out: (string | number)[][]): void {
  if (Array.isArray(node)) node.forEach((x, i) => charsmapPaths(x, [...path, i], out));
  else if (node && typeof node === 'object') {
    const o = node as Json;
    if (o.type === 'Precompiled' && typeof o.precompiled_charsmap === 'string') out.push([...path, 'precompiled_charsmap']);
    for (const [k, v] of Object.entries(o)) if (v && typeof v === 'object') charsmapPaths(v, [...path, k], out);
  }
}

/** The uncompressed KHT payload of a parsed tokenizer.json. */
export function encodeKht(src: Json): Uint8Array {
  const head = structuredClone(src);
  const fields: KhtField[] = [];
  const sections: Uint8Array[] = [];
  const add = (f: Omit<KhtField, 'sections'>, secs: Uint8Array[]): void => {
    fields.push({ ...f, sections: secs.map((s) => s.length) });
    sections.push(...secs);
    let at = head as Record<string | number, unknown>;
    for (const k of f.path.slice(0, -1)) at = at[k] as Record<string | number, unknown>;
    at[f.path[f.path.length - 1]] = null;
  };
  const model = src.model as Json | undefined;
  let tokens: string[] | null = null;
  let vocabField = -1;
  if (model && Array.isArray(model.vocab) && model.type === 'Unigram') {
    const vocab = model.vocab as [string, number][];
    tokens = vocab.map((e) => e[0]);
    const prefix = pickPrefix(tokens);
    vocabField = fields.length;
    add({ path: ['model', 'vocab'], kind: 'unigram', prefix },
      [writeStrings(tokens, prefix), ...writeScores(vocab.map((e) => e[1]))]);
  } else if (model && model.vocab && typeof model.vocab === 'object') {
    const entries = Object.entries(model.vocab as Record<string, number>).sort((a, b) => a[1] - b[1]);
    tokens = entries.map((e) => e[0]);
    const ids = entries.map((e) => e[1]);
    const dense = ids.every((id, i) => id === i);
    const prefix = pickPrefix(tokens);
    vocabField = fields.length;
    add({ path: ['model', 'vocab'], kind: 'dict', prefix, dense },
      dense ? [writeStrings(tokens, prefix)] : [writeStrings(tokens, prefix), writeVarints(ids)]);
  }
  if (model && Array.isArray(model.merges) && tokens) {
    const merges = model.merges as (string | [string, string])[];
    const pos = new Map<string, number>();
    tokens.forEach((t, i) => { if (!pos.has(t)) pos.set(t, i); });
    const form = typeof merges[0] === 'string' ? 'str' : 'pair';
    const ix: number[] = [];
    const ok = merges.every((m) => {
      const parts = form === 'str' ? (m as string).split(' ') : m as string[];
      if (typeof m !== (form === 'str' ? 'string' : 'object') || parts.length !== 2) return false;
      const a = pos.get(parts[0]); const b = pos.get(parts[1]);
      if (a === undefined || b === undefined) return false;
      ix.push(a, b);
      return true;
    });
    if (ok) add({ path: ['model', 'merges'], kind: 'merges', of: vocabField, form }, [writeVarints(ix)]);
    else add({ path: ['model', 'merges'], kind: 'json' }, [enc.encode(JSON.stringify(merges))]);
  }
  const maps: (string | number)[][] = [];
  charsmapPaths(src.normalizer, ['normalizer'], maps);
  for (const path of maps) {
    let at = src as Record<string | number, unknown>;
    for (const k of path) at = at[k] as Record<string | number, unknown>;
    const s = at as unknown as string;
    const bytes = Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
    add({ path, kind: 'base64' }, [bytes]);
  }
  const desc = enc.encode(JSON.stringify({ head, fields }));
  const out = new Writer();
  out.bytes(enc.encode(KHT_MAGIC));
  const len = new Uint8Array(4);
  new DataView(len.buffer).setUint32(0, desc.length, true);
  out.bytes(len);
  out.bytes(desc);
  for (const s of sections) out.bytes(s);
  return out.done();
}

/** tokenizer.bin bytes of a tokenizer.json text, checked by decoding them again. */
export function tokenizerBin(text: string): Uint8Array {
  const src = JSON.parse(text) as Json;
  const gz = gzipSync(encodeKht(src), { level: 9 });
  assert.deepStrictEqual(decodeKht(new Uint8Array(gunzipSync(gz))), src);
  return new Uint8Array(gz);
}

function sources(): { name: string; path: string }[] {
  const out: { name: string; path: string }[] = [];
  const k28 = join(ROOT, 'models/k28');
  for (const slug of readdirSync(k28).sort()) {
    const own = join(k28, slug, 'tokenizer.json');
    const repo = join(k28, slug, 'repo/tokenizer.json');
    if (existsSync(own)) out.push({ name: slug, path: own });
    else if (existsSync(repo)) out.push({ name: slug, path: repo });
  }
  for (const m of ['small-upstream', 'base-upstream', 'multi-upstream', 'julia-1']) {
    const p = join(ROOT, 'models', m, 'f32/tokenizer.json');
    if (existsSync(p)) out.push({ name: m, path: p });
  }
  return out;
}

const args = process.argv.slice(2);
const cli = process.argv[1] === fileURLToPath(import.meta.url);
if (!cli) {
  // imported (tools/k28_convert.ts): no command line
} else if (args[0] === '--survey') {
  const rows = [];
  for (const { name, path } of sources()) {
    const text = readFileSync(path, 'utf8');
    const t0 = performance.now();
    const bin = tokenizerBin(text);
    const ms = performance.now() - t0;
    const type = (JSON.parse(text) as { model?: { type?: string } }).model?.type ?? '?';
    const row = { name, type, json: Buffer.byteLength(text), jsonGzip: gzipSync(text, { level: 9 }).length, bin: bin.length };
    rows.push(row);
    console.log(`${name.padEnd(70)} ${type.padEnd(9)} ${String(row.json).padStart(10)} ${String(row.jsonGzip).padStart(9)} ${String(row.bin).padStart(9)} ${ms.toFixed(0)} ms`);
  }
  writeFileSync(join(ROOT, 'bench/results/r6-tokbin-survey.json'), JSON.stringify({ date: new Date().toISOString().slice(0, 10), rows }, null, 1));
  console.log(`${rows.length} tokenizers, all decode equal`);
} else if (args[0] === '--migrate') {
  // every folder under models/ and models/k28/ with an f16/ or f32/ manifest; models/hf-upload holds
  // the public Hugging Face uploads, which engines before R6 read, so it stays as it is
  const dirs: string[] = [];
  for (const top of ['models/k28', 'models']) {
    if (!existsSync(join(ROOT, top))) continue;
    for (const name of readdirSync(join(ROOT, top)).sort()) if (name !== 'hf-upload') dirs.push(join(ROOT, top, name));
  }
  let n = 0;
  for (const base of dirs) {
    for (const p of ['f16', 'f32']) {
      const mf = join(base, p, 'manifest.json');
      if (!existsSync(mf)) continue;
      const text = readFileSync(mf, 'utf8');
      const manifest = JSON.parse(text) as Json;
      if (typeof manifest.tokenizer !== 'string') continue;
      const json = join(base, p, 'tokenizer.json');
      if (!existsSync(json)) continue;
      const bin = tokenizerBin(readFileSync(json, 'utf8'));
      writeFileSync(join(base, p, 'tokenizer.bin'), bin);
      // the transition field of the first migration (12.10.) goes; tokenizer names the bin
      const next = text.replace(/, "tokenizerBin": "tokenizer\.bin"/, '')
        .replace(/("tokenizer":\s*)"[^"]+"/, '$1"tokenizer.bin"');
      const { tokenizerBin: _, ...rest } = manifest;
      assert.deepStrictEqual(JSON.parse(next), { ...rest, tokenizer: 'tokenizer.bin' });
      if (next !== text) writeFileSync(mf, next);
      n += 1;
      console.log(`${join(base, p).slice(ROOT.length + 1)} ${bin.length} B`);
    }
  }
  console.log(`${n} model folders carry tokenizer.bin`);
} else if (args[0]) {
  const bin = tokenizerBin(readFileSync(args[0], 'utf8'));
  const out = args[1] ?? join(dirname(args[0]), 'tokenizer.bin');
  writeFileSync(out, bin);
  console.log(`${out} ${bin.length} B`);
} else {
  console.error('usage: node tools/r6_tokbin.ts <tokenizer.json> [out] | --survey');
  process.exit(2);
}
