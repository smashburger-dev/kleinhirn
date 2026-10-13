import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { decodeKht, decodeKhtCompact, expandCompact } from '../src/tokenizer/bin.ts';

// Every tokenizer.bin of the k28 survey (identical files once) and the engine tokenizers: the
// compact fast path expands to the object decodeKht returns, scores compared with Object.is.
const files: string[] = [];
if (existsSync('models/k28')) {
  const seen = new Set<string>();
  for (const name of readdirSync('models/k28').sort()) {
    const path = `models/k28/${name}/f16/tokenizer.bin`;
    if (!existsSync(path)) continue;
    const hash = createHash('sha256').update(readFileSync(path)).digest('hex');
    if (seen.has(hash)) continue;
    seen.add(hash);
    files.push(path);
  }
}
for (const dir of ['models/small-upstream/f16', 'models/multi-upstream/f16', 'models/julia-1/f16']) {
  if (existsSync(`${dir}/tokenizer.bin`)) files.push(`${dir}/tokenizer.bin`);
}
const skip = files.length ? false : 'models/ is not shipped';

for (const path of files) {
  test(`fast decode equals decodeKht: ${path}`, { skip }, () => {
    const raw = new Uint8Array(gunzipSync(readFileSync(path)));
    assert.deepStrictEqual(expandCompact(decodeKhtCompact(raw)), decodeKht(raw));
  });
}

// Synthetic payload: a token starting with U+FEFF, a 4-byte code point, a prefix, and
// unigram scores in all four kinds of a score (exact, one ULP up and down, full f64).
test('fast decode on a synthetic payload', { skip: false }, async () => {
  const { KHT_MAGIC } = await import('../src/tokenizer/bin.ts');
  const enc = new TextEncoder();
  const bytesOf = (s: string) => enc.encode(s);
  const varint = (n: number) => { const out: number[] = []; do { let b = n % 128; n = Math.floor(n / 128); if (n) b |= 0x80; out.push(b); } while (n); return out; };
  const strs = (list: string[], prefixFlag: boolean[]) => {
    const out: number[] = [];
    list.forEach((s, i) => { const b = bytesOf(s); out.push(...varint(b.length * 2 + (prefixFlag[i] ? 1 : 0))); out.push(...b); });
    return Uint8Array.from(out);
  };
  const tokens = ['﻿a', 'x\u{1F600}', 'é', 'bc'];
  const dictBytes = strs(tokens, [false, true, false, true]);
  const ids = Uint8Array.from([...varint(3), ...varint(7), ...varint(0), ...varint(200)]);
  const pieces = ['▁a', '▁b'];
  const pieceBytes = strs(pieces, [false, false]);
  const scores = [-1.5, 3.0000000000000004];
  const f32 = new Float32Array(scores.length);
  f32.set(scores);
  const planes = new Uint8Array(4 * scores.length);
  const f32b = new Uint8Array(f32.buffer);
  for (let i = 0; i < scores.length; i += 1) for (let b = 0; b < 4; b += 1) planes[b * scores.length + i] = f32b[4 * i + b];
  const kinds = Uint8Array.from([0, 3]);
  const extra = new Uint8Array(8);
  new DataView(extra.buffer).setFloat64(0, scores[1], true);
  const sections = [dictBytes, ids, pieceBytes, planes, kinds, extra];
  const desc = {
    head: { model: { vocab: null, unigram: null }, added: [{ id: 1, content: '<s>' }] },
    fields: [
      { path: ['model', 'vocab'], kind: 'dict', sections: [dictBytes.length, ids.length], prefix: '▁' },
      { path: ['model', 'unigram'], kind: 'unigram', sections: [pieceBytes.length, planes.length, kinds.length, extra.length], prefix: '' },
    ],
  };
  const descBytes = bytesOf(JSON.stringify(desc));
  const head = new Uint8Array(8 + descBytes.length + sections.reduce((a, s) => a + s.length, 0));
  head.set(bytesOf(KHT_MAGIC), 0);
  new DataView(head.buffer).setUint32(4, descBytes.length, true);
  head.set(descBytes, 8);
  let p = 8 + descBytes.length;
  for (const s of sections) { head.set(s, p); p += s.length; }
  assert.deepStrictEqual(expandCompact(decodeKhtCompact(head)), decodeKht(head));
});

// f64Step against the 64-bit reference (BigInt on the bit pattern): the neighbouring pattern, Object.is.
test('f64Step equals the 64-bit neighbour on f32 values and random doubles', async () => {
  const { f64Step } = await import('../src/tokenizer/bin.ts');
  const ref = (x: number, up: boolean): number => {
    const v = new DataView(new ArrayBuffer(8));
    v.setFloat64(0, x, true);
    const bits = v.getBigUint64(0, true);
    v.setBigUint64(0, (x > 0) === up ? bits + 1n : bits - 1n, true);
    return v.getFloat64(0, true);
  };
  const f32 = new Float32Array(1);
  const samples: number[] = [];
  for (let i = 0; i < 20000; i += 1) {
    f32[0] = (Math.random() - 0.5) * 2 ** (Math.floor(Math.random() * 40) - 20);
    samples.push(f32[0]);
    samples.push((Math.random() - 0.5) * 1e3);
  }
  samples.push(1, -1, 2 ** -149, -(2 ** -149), 1.7976931348623157e308, -1.7976931348623157e308, 2 ** -1022);
  for (const x of samples) {
    if (x === 0 || !Number.isFinite(x)) continue;
    for (const up of [true, false]) assert.ok(Object.is(f64Step(x, up), ref(x, up)), `${x} ${up}`);
  }
});
