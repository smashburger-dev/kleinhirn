// R4 stage 1: words.bin checksum and encoding, buildManifest without the table in the shards, and
// the migration of an old-format folder (tools/r4_words.ts).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildManifest, type ManifestMeta } from '../src/convert/manifest.ts';
import { checkRow, rowChecksum, type WordsDesc } from '../src/words-format.ts';
import { encodeWords } from '../src/words-encode.ts';
import { ensureRows, fetchWords, legacyTable, wordsView, type WordTable } from '../src/words.ts';
import { checkDir, migrateDir } from '../tools/r4_words.ts';

const TABLE = 'embeddings.word.weight';
const bytesOf = (a: ArrayBufferView): Uint8Array => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const sha = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');
const f16 = (values: number[]): Uint8Array => bytesOf(new Float16Array(values));
const f32 = (values: number[]): Uint8Array => bytesOf(new Float32Array(values));

test('rowChecksum: pinned vectors, one flipped byte or another row id or salt changes the result', () => {
  const a = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(rowChecksum(a, 0, 0x12345678), 1313458805);
  assert.equal(rowChecksum(a, 5, 0x12345678), 2842431045);
  assert.equal(rowChecksum(new Uint8Array(16), 0, 0), 3286763420);
  assert.equal(rowChecksum(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 9]), 0, 0x12345678), 1982651642);
  assert.equal(rowChecksum(a, 0, 0x12345679), 1216509786);
  assert.throws(() => rowChecksum(new Uint8Array(6), 0, 0), /whole u32/);
});

test('encodeWords: rows are the table rows plus a checksum, checkRow rejects a damaged row', async () => {
  const values = Array.from({ length: 80 }, (_, k) => (k % 8) + Math.floor(k / 8) * 0.5);
  const table = f16(values);
  const { bytes, salt } = await encodeWords(table, 10, 8, 'f16');
  const rowBytes = 8 * 2 + 4;
  assert.equal(bytes.length, 10 * rowBytes);
  const salted = new Uint8Array(await crypto.subtle.digest('SHA-256', table as BufferSource));
  assert.equal(salt, new DataView(salted.buffer).getUint32(0, true));
  for (let i = 0; i < 10; i += 1) {
    const row = bytes.subarray(i * rowBytes, (i + 1) * rowBytes);
    assert.deepEqual(Buffer.from(row.subarray(0, 16)), Buffer.from(table.subarray(i * 16, (i + 1) * 16)));
    assert.ok(checkRow(row, i, rowBytes, salt));
    assert.ok(!checkRow(row, i + 1, rowBytes, salt), 'a row at another id fails');
  }
  const bad = bytes.slice(0, rowBytes);
  bad[3] ^= 1;
  assert.ok(!checkRow(bad, 0, rowBytes, salt));
  await assert.rejects(encodeWords(table.subarray(0, 10), 10, 8, 'f16'), /expected/);
});

test('buildManifest: version 3, the table only in words.bin, no keepOnCpu entries', async () => {
  const meta = {
    source: { repo: 'x', revision: 'y', checkpointSha256: 'z' }, spec: {}, head: {},
    task: 't', labels: {}, tokenizer: 'tokenizer.bin', maxLength: 6,
  } as unknown as ManifestMeta;
  const table = Float32Array.from({ length: 80 }, (_, k) => (k % 7) * 0.25 - 1);
  for (const dtype of ['f16', 'f32'] as const) {
    const built = await buildManifest([
      { name: TABLE, shape: [10, 8], data: table },
      { name: 'a', shape: [3], data: new Float32Array([1, 2, 3]) },
      { name: 'b', shape: [2, 2], data: new Float32Array([4, 5, 6, 7]) },
    ], dtype, meta);
    const m = built.manifest;
    const words = m.words as WordsDesc;
    const entries = m.tensors as { name: string; keepOnCpu?: boolean }[];
    assert.equal(m.version, 3);
    assert.ok(entries.every((e) => e.name !== TABLE && e.keepOnCpu === undefined));
    assert.deepEqual(entries.map((e) => e.name), ['a', 'b']);
    assert.equal(words.file, 'words.bin');
    assert.equal(words.rows, 10);
    assert.equal(words.width, 8);
    assert.equal(words.rowBytes, 8 * (dtype === 'f16' ? 2 : 4) + 4);
    assert.equal(words.bytes, built.words.bytes.length);
    assert.equal(words.sha256, sha(built.words.bytes));
    const cast = dtype === 'f16' ? f16(Array.from(table)) : f32(Array.from(table));
    const rowBytes = words.rowBytes;
    for (let i = 0; i < 10; i += 1) {
      const row = built.words.bytes.subarray(i * rowBytes, (i + 1) * rowBytes);
      const dataBytes = rowBytes - 4;
      assert.deepEqual(Buffer.from(row.subarray(0, dataBytes)), Buffer.from(cast.subarray(i * dataBytes, (i + 1) * dataBytes)));
      assert.ok(checkRow(row, i, rowBytes, words.salt));
    }
    assert.ok(built.shards.every((s) => /^shard-\d+\.bin$/.test(s.file)));
    assert.equal((m.shards as unknown[]).length, built.shards.length);
  }
});

// Old-format folder: the table in two keepOnCpu parts; part A shares shard 0 with tensor a,
// part B shares shard 1 with tensor b.
function oldFolder(base: string, badSha = false): { dir: string; table: Uint8Array } {
  const dir = join(base, 'f16');
  mkdirSync(dir, { recursive: true });
  const table = f16(Array.from({ length: 24 }, (_, k) => k * 1.5 - 3));
  const a = f32([1, 2, 3]);
  const b = f16([4, 5]);
  const shard0 = new Uint8Array(256 + 12);
  shard0.set(table.subarray(0, 32), 0);
  shard0.set(a, 256);
  const shard1 = new Uint8Array(256 + 4);
  shard1.set(table.subarray(32, 48), 0);
  shard1.set(b, 256);
  writeFileSync(join(dir, 'weights-0.bin'), shard0);
  writeFileSync(join(dir, 'weights-1.bin'), shard1);
  const manifest = {
    format: 'kleinhirn-weights-2', version: 2, encoder: {}, marker: 'kept', tokenizer: 'tokenizer.bin',
    tensors: [
      { name: TABLE, dtype: 'f16', shape: [4, 4], shard: 0, offset: 0, byteLength: 32, keepOnCpu: true, rowStart: 0, rowEnd: 4 },
      { name: 'a', dtype: 'f32', shape: [3], shard: 0, offset: 256, byteLength: 12 },
      { name: TABLE, dtype: 'f16', shape: [2, 4], shard: 1, offset: 0, byteLength: 16, keepOnCpu: true, rowStart: 4, rowEnd: 6 },
      { name: 'b', dtype: 'f16', shape: [2], shard: 1, offset: 256, byteLength: 4 },
    ],
    shards: [
      { file: 'weights-0.bin', bytes: shard0.length, sha256: badSha ? '0'.repeat(64) : sha(shard0) },
      { file: 'weights-1.bin', bytes: shard1.length, sha256: sha(shard1) },
    ],
  };
  writeFileSync(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 1)}\n`);
  return { dir, table };
}

test('migrate: tensors byte-equal, table in words.bin, old shards gone, rerun reports already', async () => {
  const base = mkdtempSync(join(tmpdir(), 'r4-words-'));
  try {
    const { dir, table } = oldFolder(base);
    const r = await migrateDir(dir);
    assert.equal(r.status, 'migrated', r.error ?? '');
    assert.equal(r.tableBytes, 48);
    const m = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
    assert.equal(m.version, 3);
    assert.equal(m.marker, 'kept');
    assert.ok(m.tensors.every((t: { keepOnCpu?: boolean }) => t.keepOnCpu === undefined));
    assert.deepEqual(m.tensors.map((t: { name: string }) => t.name), ['a', 'b']);
    const shardOf = (name: string): Uint8Array => {
      const e = m.tensors.find((t: { name: string }) => t.name === name);
      const file = readFileSync(join(dir, m.shards[e.shard].file));
      return new Uint8Array(file.subarray(e.offset, e.offset + e.byteLength));
    };
    assert.deepEqual(Buffer.from(shardOf('a')), Buffer.from(f32([1, 2, 3])));
    assert.deepEqual(Buffer.from(shardOf('b')), Buffer.from(f16([4, 5])));
    const words = readFileSync(join(dir, m.words.file));
    const rb = m.words.rowBytes;
    for (let i = 0; i < 6; i += 1) {
      const row = new Uint8Array(words.subarray(i * rb, (i + 1) * rb));
      assert.deepEqual(Buffer.from(row.subarray(0, 8)), Buffer.from(table.subarray(i * 8, (i + 1) * 8)));
      assert.ok(checkRow(row, i, rb, m.words.salt));
    }
    assert.equal(checkDir(dir).status, 'ok');
    const names = readdirSync(dir).sort();
    assert.deepEqual(names, ['manifest.json', 'shard-0.bin', 'words.bin']);
    const before = readFileSync(join(dir, 'manifest.json'), 'utf8');
    const again = await migrateDir(dir);
    assert.equal(again.status, 'already');
    assert.equal(readFileSync(join(dir, 'manifest.json'), 'utf8'), before);
    assert.equal(checkDir(dir).status, 'ok');
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
});

test('migrate: a wrong old shard sha256 leaves the folder untouched', async () => {
  const base = mkdtempSync(join(tmpdir(), 'r4-words-bad-'));
  try {
    const { dir } = oldFolder(base, true);
    const before = readFileSync(join(dir, 'manifest.json'), 'utf8');
    const files = readdirSync(dir).sort();
    const r = await migrateDir(dir);
    assert.equal(r.status, 'error');
    assert.match(r.error!, /weights-0\.bin/);
    assert.equal(readFileSync(join(dir, 'manifest.json'), 'utf8'), before);
    assert.deepEqual(readdirSync(dir).sort(), files);
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
});

// FullTable over a synthetic words.bin (R4 Festlegung 3): rows past the checksum, the offset of
// copyRow, the out-of-vocabulary error, alloc in the table dtype.
test('FullTable over words.bin: copyRow returns the rows, ids outside the vocabulary throw, alloc dtype', async () => {
  const width = 8;
  const rows = 10;
  for (const dtype of ['f16', 'f32'] as const) {
    const values = Array.from({ length: rows * width }, (_, k) => (k % 5) * 0.5 - 1 + Math.floor(k / width));
    const table = dtype === 'f16' ? f16(values) : f32(values);
    const dataBytes = width * (dtype === 'f16' ? 2 : 4);
    const { bytes, salt } = await encodeWords(table, rows, width, dtype);
    const desc: WordsDesc = {
      file: 'words.bin', bytes: bytes.length, sha256: sha(bytes), dtype, rows, width, rowBytes: dataBytes + 4, salt,
    };
    const t = wordsView(bytes.buffer as ArrayBuffer, desc);
    assert.equal(t.dtype, dtype);
    assert.equal(t.rows, rows);
    assert.equal(t.width, width);
    assert.deepEqual(t.info(), { mode: 'full', source: 'words.bin', rows, width, dtype });
    const asTable: WordTable = t;
    assert.equal(asTable.ensure([0], 1), null);
    assert.ok(t.alloc(4) instanceof (dtype === 'f16' ? Uint16Array : Float32Array));
    assert.equal(t.alloc(4).length, 4);
    for (let id = 0; id < rows; id += 1) {
      const out = t.alloc(width);
      t.copyRow(id, out, 0);
      assert.deepEqual(Buffer.from(bytesOf(out)), Buffer.from(table.subarray(id * dataBytes, (id + 1) * dataBytes)));
    }
    const wide = t.alloc(3 * width);
    t.copyRow(7, wide, 2 * width);
    assert.deepEqual(Buffer.from(bytesOf(wide.subarray(2 * width))), Buffer.from(table.subarray(7 * dataBytes, 8 * dataBytes)));
    assert.deepEqual(Array.from(wide.subarray(0, 2 * width)), new Array(2 * width).fill(0));
    assert.throws(() => t.copyRow(rows, t.alloc(width), 0), /token id 10 is outside the vocabulary of 10 rows/);
    assert.throws(() => t.copyRow(-1, t.alloc(width), 0), /token id -1 is outside/);
    assert.throws(() => t.copyRow(1.5, t.alloc(width), 0), /token id 1.5 is outside/);
    assert.deepEqual(ensureRows(t, [{ inputIds: [0, 1, 2] }]), null);
  }
});

test('legacyTable: rows back to back with stride = width, out-of-vocabulary throws', () => {
  const width = 4;
  const table = f16(Array.from({ length: 24 }, (_, k) => k * 0.5));
  const t = legacyTable(table.buffer as ArrayBuffer, 'f16', width);
  assert.equal(t.rows, 6);
  assert.deepEqual(t.info(), { mode: 'full', source: 'legacy', rows: 6, width, dtype: 'f16' });
  const out = t.alloc(width);
  t.copyRow(5, out, 0);
  assert.deepEqual(Buffer.from(bytesOf(out)), Buffer.from(table.subarray(5 * 8, 6 * 8)));
  assert.throws(() => t.copyRow(6, out, 0), /outside the vocabulary of 6 rows/);
});

test('fetchWords: the bytes, the size and the sha256 checked against the manifest entry', async () => {
  const width = 4;
  const rows = 3;
  const table = f32(Array.from({ length: rows * width }, (_, k) => k * 0.25));
  const { bytes, salt } = await encodeWords(table, rows, width, 'f32');
  const desc: WordsDesc = {
    file: 'words.bin', bytes: bytes.length, sha256: sha(bytes), dtype: 'f32', rows, width, rowBytes: width * 4 + 4, salt,
  };
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(bytes.buffer as ArrayBuffer)) as unknown as typeof fetch;
  try {
    const got = await fetchWords('http://x/words.bin', desc);
    assert.equal(got.bytes, bytes.length);
    assert.equal(got.table.rows, rows);
    const row = got.table.alloc(width);
    got.table.copyRow(2, row, 0);
    assert.deepEqual(Buffer.from(bytesOf(row)), Buffer.from(table.subarray(2 * width * 4, 3 * width * 4)));
    await assert.rejects(fetchWords('http://x/words.bin', { ...desc, sha256: '0'.repeat(64) }), /sha256 mismatch on words\.bin/);
    await assert.rejects(fetchWords('http://x/words.bin', { ...desc, bytes: desc.bytes + 4 }), /manifest says/);
  } finally {
    globalThis.fetch = realFetch;
  }
});
