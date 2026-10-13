// Word table row format (R4 Festlegung 1): WordsDesc and the row checksum. Shared by the core, the
// words-rows chunk and the converters; the encoder for converters lives in words-encode.ts.
// No runtime import from src/words.ts: the words-rows chunk uses this module, and the core keeps
// words.ts (R4 stage 3b), so a shared word module would split the core.

export interface WordsDesc {
  file: string;
  bytes: number;
  sha256: string;
  dtype: 'f16' | 'f32';
  rows: number;
  width: number;
  rowBytes: number;
  salt: number;
}

const rotl = (x: number, r: number): number => (x << r) | (x >>> (32 - r));

// murmur3_32 body over the row's u32 words, seeded with salt and the row id.
export function rowChecksum(row: Uint8Array, rowId: number, salt: number): number {
  if (row.byteLength % 4 !== 0) throw new Error(`words: row of ${row.byteLength} bytes is not whole u32 words`);
  const v = new DataView(row.buffer, row.byteOffset, row.byteLength);
  let h = (salt ^ Math.imul(rowId + 1, 0x9e3779b1)) >>> 0;
  for (let i = 0; i < row.byteLength; i += 4) {
    let k = v.getUint32(i, true);
    k = Math.imul(k, 0xcc9e2d51); k = rotl(k, 15); k = Math.imul(k, 0x1b873593);
    h ^= k; h = rotl(h, 13); h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  h ^= row.byteLength;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
  return h >>> 0;
}

// words: the bytes of exactly one row, checksum included.
export function checkRow(words: Uint8Array, rowId: number, rowBytes: number, salt: number): boolean {
  if (words.byteLength !== rowBytes) return false;
  const stored = new DataView(words.buffer, words.byteOffset, words.byteLength).getUint32(rowBytes - 4, true);
  return rowChecksum(words.subarray(0, rowBytes - 4), rowId, salt) === stored;
}
