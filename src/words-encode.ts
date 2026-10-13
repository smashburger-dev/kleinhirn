// words.bin encoder (R4 Festlegung 1), for the converters and the migration tool only. The
// row format is in words-format.ts; no module of the engines imports this file.

import { rowChecksum, type WordsDesc } from './words-format.ts';

// table: dense row-major rows in dtype bytes. Returns the words file bytes and the salt
// (first 4 bytes of SHA-256 of the table, u32 LE).
export async function encodeWords(
  table: Uint8Array, rows: number, width: number, dtype: 'f16' | 'f32',
): Promise<{ bytes: Uint8Array; salt: number }> {
  const dataBytes = width * (dtype === 'f16' ? 2 : 4);
  if (table.byteLength !== rows * dataBytes) {
    throw new Error(`words: table is ${table.byteLength} bytes, expected ${rows} x ${dataBytes}`);
  }
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', table as BufferSource));
  const salt = new DataView(digest.buffer).getUint32(0, true);
  const rowBytes = dataBytes + 4;
  const bytes = new Uint8Array(rows * rowBytes);
  const out = new DataView(bytes.buffer);
  for (let i = 0; i < rows; i += 1) {
    const at = i * rowBytes;
    bytes.set(table.subarray(i * dataBytes, (i + 1) * dataBytes), at);
    out.setUint32(at + dataBytes, rowChecksum(bytes.subarray(at, at + dataBytes), i, salt), true);
  }
  return { bytes, salt };
}
