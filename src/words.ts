// Word table file (words.bin, R4 Festlegung 1): one row per token id, the row's values in the
// manifest dtype and a 32-bit checksum after them. The row format lives in words-format.ts.

import type { WordsDesc } from './words-format.ts';

// The word table as the engines read it (R4 Festlegung 3). FullTable holds every row in memory;
// ensure() is the seam for the partial tables of Stufe 3.
export type RowArray = Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer>;

export interface WordTable {
  readonly dtype: 'f16' | 'f32';
  readonly rows: number;
  readonly width: number;
  // Rows of ids[0..count) not in memory yet: a promise that settles once they are; null when the
  // caller can gather right away (FullTable: always null). Callers await it only when non-null.
  ensure(ids: ArrayLike<number>, count: number): Promise<void> | null;
  // The row of `id` into out at element offset `at`; ids outside the vocabulary throw with the
  // message of tokenRowOffset (src/tasks.ts).
  copyRow(id: number, out: RowArray, at: number): void;
  // A zeroed array of the table dtype with `elements` entries.
  alloc(elements: number): RowArray;
  info(): Record<string, unknown>;
}

// The whole table in one typed view: row i starts at element i * stride.
export class FullTable implements WordTable {
  readonly rows: number;

  constructor(
    private readonly view: RowArray,
    readonly dtype: 'f16' | 'f32',
    readonly width: number,
    private readonly stride: number,
    private readonly source: 'words.bin' | 'legacy',
  ) {
    this.rows = Math.floor(view.length / stride);
  }

  ensure(): Promise<void> | null {
    return null;
  }

  copyRow(id: number, out: RowArray, at: number): void {
    if (!Number.isInteger(id) || id < 0 || id >= this.rows) {
      throw new Error(`token id ${id} is outside the vocabulary of ${this.rows} rows (ids are integers from 0)`);
    }
    const off = id * this.stride;
    out.set(this.view.subarray(off, off + this.width), at);
  }

  alloc(elements: number): RowArray {
    return this.dtype === 'f16' ? new Uint16Array(elements) : new Float32Array(elements);
  }

  info(): Record<string, unknown> {
    return { mode: 'full', source: this.source, rows: this.rows, width: this.width, dtype: this.dtype };
  }
}

// words.bin as a FullTable over the buffer, no copy. The stride is the row bytes in elements, so
// the checksum after each row sits outside the row's width.
export function wordsView(buf: ArrayBuffer, desc: WordsDesc): FullTable {
  if (buf.byteLength !== desc.rows * desc.rowBytes) {
    throw new Error(`words.bin is ${buf.byteLength} bytes, ${desc.rows} rows of ${desc.rowBytes} need ${desc.rows * desc.rowBytes}`);
  }
  const view = desc.dtype === 'f16' ? new Uint16Array(buf) : new Float32Array(buf);
  return new FullTable(view, desc.dtype, desc.width, desc.rowBytes / view.BYTES_PER_ELEMENT, 'words.bin');
}

// The keepOnCpu parts of a legacy manifest joined in row order: rows are back to back, stride = width.
export function legacyTable(joined: ArrayBuffer, dtype: 'f16' | 'f32', width: number): FullTable {
  const view = dtype === 'f16' ? new Uint16Array(joined) : new Float32Array(joined);
  return new FullTable(view, dtype, width, width, 'legacy');
}

// The load option `words` (R4 Festlegung 5): full downloads words.bin, rows fetches the rows of the
// input over Range, progressive does rows first and then downloads words.bin whole.
export type WordsMode = 'full' | 'rows' | 'progressive';

// The engine's info().words: the table's info with the requested mode. A legacy manifest has no
// words.bin, so rows and progressive load its FullTable and say so.
export function wordsInfo(table: WordTable, requested: WordsMode): Record<string, unknown> {
  const info = table.info();
  if (info.source === 'legacy' && requested !== 'full') {
    return { ...info, requested, note: 'legacy manifest has no words.bin' };
  }
  return { ...info, requested };
}

export interface WordsFetch {
  table: WordTable;
  bytes: number;
  fetchMs: number;
  sha256Ms: number;
}

// words.bin of a version 3 manifest: size, row layout and sha256 checked against desc.
export async function fetchWords(url: string, desc: WordsDesc): Promise<WordsFetch> {
  if (desc.rowBytes !== desc.width * (desc.dtype === 'f16' ? 2 : 4) + 4) {
    throw new Error(`words: rowBytes ${desc.rowBytes} is not ${desc.width} ${desc.dtype} values and a checksum`);
  }
  const tFetch = performance.now();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const fetchMs = performance.now() - tFetch;
  if (buf.byteLength !== desc.bytes) throw new Error(`words.bin is ${buf.byteLength} bytes, manifest says ${desc.bytes}`);
  const tHash = performance.now();
  const digest = [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buf))]
    .map((b) => b.toString(16).padStart(2, '0')).join('');
  const sha256Ms = performance.now() - tHash;
  if (digest !== desc.sha256) throw new Error('sha256 mismatch on words.bin');
  return { table: wordsView(buf, desc), bytes: buf.byteLength, fetchMs, sha256Ms };
}

// The rows the calls gather that the table does not hold yet, as one promise. Null when every
// call can gather now (always, for FullTable): callers await only a non-null result.
export function ensureRows(
  table: WordTable, inputs: ArrayLike<{ inputIds: ArrayLike<number>; seqLen?: number }>,
): Promise<void> | null {
  let pending: Promise<void>[] | undefined;
  for (let k = 0; k < inputs.length; k += 1) {
    const { inputIds, seqLen } = inputs[k];
    const p = table.ensure(inputIds, seqLen ?? inputIds.length);
    if (p) (pending ??= []).push(p);
  }
  return pending ? Promise.all(pending).then(() => undefined) : null;
}
