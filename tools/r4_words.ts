// R4 stage 1: the word table moves from the shards into words.bin (format in src/words.ts).
// --migrate turns an old folder (table parts keepOnCpu in weights-N.bin) into version 3:
// shard-N.bin without the table, words.bin, manifest.json swapped in by rename. Before the swap
// every new file is read back from disk and checked: sha256 against the new manifest, every
// non-table tensor byte-equal to the old shards, every word row checksummed and equal to the old
// table. Any failure removes the new files and leaves the old folder as it was.
// Usage: node --import ./tests/helpers/node-hooks.mjs tools/r4_words.ts --migrate <dir>...
//        ... tools/r4_words.ts --migrate-all [--dry-run]   models/*/{f16,f32}, models/k28/*/{f16,f32};
//                                                          paths with hf-upload or r4-pre are skipped
//        ... tools/r4_words.ts --check <dir>...            shard and words sha256, row checksums
//        add --report <file> to any mode for the per-dir JSON
// After an export (convert/export_weights.py, convert/export_julia.py) run --migrate on its folder.

import { createHash } from 'node:crypto';
import {
  existsSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { packShards, placeTensors, type TensorEntry } from '../src/convert/manifest.ts';
import { checkRow, type WordsDesc } from '../src/words-format.ts';
import { encodeWords } from '../src/words-encode.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TABLE = 'embeddings.word.weight';

interface OldEntry extends TensorEntry {
  keepOnCpu?: boolean;
  rowStart?: number;
  rowEnd?: number;
}

interface FolderManifest extends Record<string, unknown> {
  version?: number;
  tensors: OldEntry[];
  shards: { file: string; bytes: number; sha256: string }[];
  words?: WordsDesc;
}

export interface DirReport {
  dir: string;
  status: 'migrated' | 'already' | 'skipped' | 'dry' | 'ok' | 'error';
  oldBytes: number;
  newBytes: number;
  tableBytes: number;
  ms: number;
  error?: string;
}

const sha256 = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');
const dtypeBytes = (dtype: string): number => (dtype === 'f16' ? 2 : 4);
const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);
const same = (a: Uint8Array, b: Uint8Array): boolean => a.length === b.length && Buffer.compare(a, b) === 0;
const slice = (shards: Uint8Array[], e: TensorEntry): Uint8Array =>
  shards[e.shard].subarray(e.offset, e.offset + e.byteLength);

// Table parts in rowStart order: contiguous rows 0..rows, one dtype, one width.
function tableOf(m: FolderManifest): { parts: OldEntry[]; rows: number; width: number; dtype: 'f16' | 'f32' } {
  const parts = m.tensors.filter((t) => t.keepOnCpu).sort((a, b) => a.rowStart! - b.rowStart!);
  if (parts.length === 0) throw new Error('no keepOnCpu tensors');
  const { dtype, shape: [, width] } = parts[0];
  let next = 0;
  for (const p of parts) {
    if (p.name !== TABLE || p.dtype !== dtype || p.shape[1] !== width || p.rowStart !== next) {
      throw new Error(`table part ${p.name} rows ${p.rowStart}-${p.rowEnd} is not contiguous or differs`);
    }
    if (p.byteLength !== (p.rowEnd! - p.rowStart!) * width * dtypeBytes(dtype)) {
      throw new Error(`table part ${p.rowStart}-${p.rowEnd}: ${p.byteLength} bytes`);
    }
    next = p.rowEnd!;
  }
  return { parts, rows: next, width, dtype };
}

function readOldShards(dir: string, m: FolderManifest): Uint8Array[] {
  return m.shards.map((s) => {
    const bytes = new Uint8Array(readFileSync(join(dir, s.file)));
    if (bytes.length !== s.bytes || sha256(bytes) !== s.sha256) throw new Error(`${s.file}: does not match the manifest`);
    return bytes;
  });
}

// Writes nothing. Returns the new manifest, the shards, the words file and the table bytes.
async function build(m: FolderManifest, old: Uint8Array[]) {
  const t = tableOf(m);
  const table = new Uint8Array(t.rows * t.width * dtypeBytes(t.dtype));
  for (const p of t.parts) table.set(slice(old, p), p.rowStart! * t.width * dtypeBytes(t.dtype));
  const others = m.tensors.filter((e) => !e.keepOnCpu);
  const packed = packShards(others.map((e) => ({ name: e.name, dtype: e.dtype, shape: e.shape, bytes: slice(old, e) })));
  const words = await encodeWords(table, t.rows, t.width, t.dtype);
  const shards = packed.shards.map((bytes, i) => ({ file: `shard-${i}.bin`, bytes }));
  const wordsDesc: WordsDesc = {
    file: 'words.bin', bytes: words.bytes.length, sha256: sha256(words.bytes), dtype: t.dtype,
    rows: t.rows, width: t.width, rowBytes: t.width * dtypeBytes(t.dtype) + 4, salt: words.salt,
  };
  const next = {
    ...m, version: 3, tensors: packed.entries,
    shards: shards.map((s) => ({ file: s.file, bytes: s.bytes.length, sha256: sha256(s.bytes) })),
    words: wordsDesc,
  };
  return { next, shards, words: words.bytes, table, wordsDesc };
}

// Reads the written files back and checks them against the old shards and table.
function verify(
  dir: string, old: Uint8Array[], oldM: FolderManifest, next: FolderManifest, table: Uint8Array,
): void {
  const parsed = JSON.parse(readFileSync(join(dir, 'manifest.r4.json'), 'utf8')) as FolderManifest;
  if (JSON.stringify(parsed) !== JSON.stringify(next)) throw new Error('manifest.r4.json differs from the built manifest');
  const newShards = parsed.shards.map((s) => {
    const bytes = new Uint8Array(readFileSync(join(dir, s.file)));
    if (bytes.length !== s.bytes || sha256(bytes) !== s.sha256) throw new Error(`${s.file}: sha256 differs on disk`);
    return bytes;
  });
  const w = parsed.words!;
  const words = new Uint8Array(readFileSync(join(dir, w.file)));
  if (words.length !== w.bytes || sha256(words) !== w.sha256) throw new Error(`${w.file}: sha256 differs on disk`);
  const dataBytes = w.width * dtypeBytes(w.dtype);
  if (words.length !== w.rows * w.rowBytes || w.rowBytes !== dataBytes + 4) throw new Error(`${w.file}: row layout`);
  for (let i = 0; i < w.rows; i += 1) {
    const row = words.subarray(i * w.rowBytes, (i + 1) * w.rowBytes);
    if (!checkRow(row, i, w.rowBytes, w.salt)) throw new Error(`${w.file}: row ${i} checksum fails`);
    if (!same(row.subarray(0, dataBytes), table.subarray(i * dataBytes, (i + 1) * dataBytes))) {
      throw new Error(`${w.file}: row ${i} differs from the old table`);
    }
  }
  const newByName = new Map(parsed.tensors.map((e) => [e.name, e]));
  const oldOthers = oldM.tensors.filter((e) => !e.keepOnCpu);
  if (oldOthers.length !== parsed.tensors.length) throw new Error('tensor count differs');
  for (const e of oldOthers) {
    const n = newByName.get(e.name);
    if (!n || n.dtype !== e.dtype || JSON.stringify(n.shape) !== JSON.stringify(e.shape)) {
      throw new Error(`${e.name}: dtype or shape differs`);
    }
    if (!same(slice(newShards, n), slice(old, e))) throw new Error(`${e.name}: bytes differ from the old shards`);
  }
}

// Old shards (weights-N.bin) not named by the manifest that stays in the folder.
function removeWeights(dir: string, keep: Set<string>): void {
  for (const f of readdirSync(dir)) {
    if (/^weights-\d+\.bin$/.test(f) && !keep.has(f)) rmSync(join(dir, f), { force: true });
  }
}

export async function migrateDir(dir: string, dryRun = false): Promise<DirReport> {
  const t0 = performance.now();
  const text = readFileSync(join(dir, 'manifest.json'), 'utf8');
  const m = JSON.parse(text) as FolderManifest;
  const oldBytes = sum(m.shards.map((s) => s.bytes));
  const rep: DirReport = { dir, status: 'error', oldBytes, newBytes: oldBytes, tableBytes: 0, ms: 0 };
  const finish = (): DirReport => ({ ...rep, ms: Math.round(performance.now() - t0) });
  try {
    if (m.words) {
      rep.status = 'already';
      rep.newBytes = oldBytes + m.words.bytes;
      if (!dryRun) removeWeights(dir, new Set(m.shards.map((s) => s.file)));
      return finish();
    }
    if (!m.tensors.some((t) => t.keepOnCpu)) {
      rep.status = 'skipped';
      return finish();
    }
    const t = tableOf(m);
    rep.tableBytes = t.rows * t.width * dtypeBytes(t.dtype);
    if (dryRun) {
      // sizes only, no shard bytes read
      const others = m.tensors.filter((e) => !e.keepOnCpu);
      const shardBytes = sum(placeTensors(others.map((e) => e.byteLength)).lengths);
      rep.newBytes = shardBytes + t.rows * (t.width * dtypeBytes(t.dtype) + 4);
      rep.status = 'dry';
      return finish();
    }
    const old = readOldShards(dir, m);
    const built = await build(m, old);
    const created = [...built.shards.map((s) => s.file), built.wordsDesc.file, 'manifest.r4.json'];
    if (created.some((f) => existsSync(join(dir, f)))) throw new Error('new file names already exist in the folder');
    try {
      for (const s of built.shards) writeFileSync(join(dir, s.file), s.bytes);
      writeFileSync(join(dir, built.wordsDesc.file), built.words);
      writeFileSync(join(dir, 'manifest.r4.json'), `${JSON.stringify(built.next, null, 1)}\n`);
      verify(dir, old, m, built.next, built.table);
    } catch (e) {
      for (const f of created) rmSync(join(dir, f), { force: true });
      throw e;
    }
    renameSync(join(dir, 'manifest.r4.json'), join(dir, 'manifest.json'));
    removeWeights(dir, new Set(built.next.shards.map((s) => s.file)));
    rep.newBytes = sum(built.shards.map((s) => s.bytes.length)) + built.words.length;
    rep.status = 'migrated';
    return finish();
  } catch (e) {
    rep.error = (e as Error).message;
    return finish();
  }
}

export function checkDir(dir: string): DirReport {
  const t0 = performance.now();
  const m = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8')) as FolderManifest;
  const oldBytes = sum(m.shards.map((s) => s.bytes));
  const rep: DirReport = { dir, status: 'ok', oldBytes, newBytes: oldBytes, tableBytes: 0, ms: 0 };
  try {
    if (!m.words) throw new Error('manifest has no words');
    if (m.tensors.some((t) => t.keepOnCpu)) throw new Error('keepOnCpu tensor left in the manifest');
    for (const s of m.shards) {
      const bytes = new Uint8Array(readFileSync(join(dir, s.file)));
      if (bytes.length !== s.bytes || sha256(bytes) !== s.sha256) throw new Error(`${s.file}: sha256 differs`);
    }
    for (const e of m.tensors) {
      const shard = m.shards[e.shard];
      if (!shard || e.offset + e.byteLength > shard.bytes) throw new Error(`${e.name}: outside its shard`);
    }
    const w = m.words;
    const words = new Uint8Array(readFileSync(join(dir, w.file)));
    if (words.length !== w.bytes || sha256(words) !== w.sha256) throw new Error(`${w.file}: sha256 differs`);
    if (words.length !== w.rows * w.rowBytes) throw new Error(`${w.file}: ${words.length} bytes, ${w.rows} rows`);
    for (let i = 0; i < w.rows; i += 1) {
      if (!checkRow(words.subarray(i * w.rowBytes, (i + 1) * w.rowBytes), i, w.rowBytes, w.salt)) {
        throw new Error(`${w.file}: row ${i} checksum fails`);
      }
    }
    rep.newBytes = oldBytes + w.bytes;
    rep.tableBytes = w.bytes;
  } catch (e) {
    rep.status = 'error';
    rep.error = (e as Error).message;
  }
  return { ...rep, ms: Math.round(performance.now() - t0) };
}

function allDirs(): string[] {
  const models = join(ROOT, 'models');
  const found: string[] = [];
  const candidates = [
    ...readdirSync(models).flatMap((m) => ['f16', 'f32'].map((p) => join(models, m, p))),
    ...readdirSync(join(models, 'k28')).flatMap((m) => ['f16', 'f32'].map((p) => join(models, 'k28', m, p))),
  ];
  for (const d of candidates) {
    if (!existsSync(join(d, 'manifest.json'))) continue;
    if (/hf-upload|r4-pre/.test(d) || /hf-upload|r4-pre/.test(realpathSync(d))) continue;
    found.push(d);
  }
  return found.sort();
}

const mb = (n: number): string => (n / 1e6).toFixed(1);

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const ri = args.indexOf('--report');
  const report = ri >= 0 ? args[ri + 1] : undefined;
  const positional = args.filter((a, i) => !a.startsWith('--') && i !== ri + 1);
  const dryRun = args.includes('--dry-run');
  const rows: DirReport[] = [];
  let mode: 'migrate' | 'check';
  let dirs: string[];
  if (args.includes('--migrate-all')) {
    mode = 'migrate';
    dirs = allDirs();
  } else if (args.includes('--migrate') || args.includes('--check')) {
    mode = args.includes('--check') ? 'check' : 'migrate';
    dirs = positional.map((d) => resolve(d));
  } else {
    console.error('usage: r4_words.ts --migrate <dir>... | --migrate-all [--dry-run] | --check <dir>... [--report <file>]');
    process.exit(2);
  }
  if (dirs.length === 0) throw new Error('no folders given');
  for (const d of dirs) {
    const r = mode === 'check' ? checkDir(d) : await migrateDir(d, dryRun);
    rows.push(r);
    const tag = r.error ? ` ${r.error}` : '';
    console.log(`${r.status.padEnd(9)} ${d.slice(ROOT.length + 1)} old ${mb(r.oldBytes)} MB new ${mb(r.newBytes)} MB table ${mb(r.tableBytes)} MB ${r.ms} ms${tag}`);
  }
  const total = rows.reduce((acc, r) => ({ old: acc.old + r.oldBytes, new: acc.new + r.newBytes }), { old: 0, new: 0 });
  console.log(`${rows.length} folders, old ${mb(total.old)} MB, new ${mb(total.new)} MB${dryRun ? ' (dry run)' : ''}`);
  if (report) writeFileSync(report, `${JSON.stringify({ mode, dryRun, dirs: rows }, null, 1)}\n`);
  if (rows.some((r) => r.status === 'error')) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
