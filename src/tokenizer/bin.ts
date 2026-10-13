// R6 tokenizer.bin (docs/R6_WORKORDER.md Festlegung 6): a gzip file holding a tokenizer.json with
// its large fields in binary sections. The decoder returns an object equal to
// JSON.parse(tokenizer.json) (deepStrictEqual, scores bit for bit), so the tokenizer classes stay
// as they are. The encoder lives in tools/r6_tokbin.ts and checks every file it writes with
// decodeKht.
//
// Payload after gunzip: "KHT1", u32 LE length of the descriptor JSON, the descriptor
// { head, fields }, then the sections of all fields in order. head is the tokenizer.json with every
// extracted field set to null; each field names its path in head, its kind and the byte length of
// each of its sections.
//   dict     {token: id}: strings, then ids as varints unless dense (id = position)
//   unigram  [[piece, score]]: strings, f32 scores in four byte planes, one kind byte per score
//            (0 the f32 value, 1 and 2 one f64 step up or down from it, 3 a full f64 that follows
//            in the fourth section)
//   merges   BPE merges as two varint positions in the strings of the field named by `of`,
//            joined by a space (form 'str') or as a pair (form 'pair')
//   base64   a base64 string (Precompiled charsmap) as its bytes
//   json     any value as JSON text (fallback)
// Strings: varint (UTF-8 length << 1 | starts with `prefix`), then the UTF-8 bytes without it.

import { StringIndex, Tokens } from './strtab.ts';

export interface KhtField {
  path: (string | number)[];
  kind: 'dict' | 'unigram' | 'merges' | 'base64' | 'json';
  sections: number[];
  prefix?: string;
  dense?: boolean;
  of?: number; // merges: index of the dict or unigram field whose strings it points into
  form?: 'str' | 'pair';
}

export interface KhtDescriptor {
  head: Record<string, unknown>;
  fields: KhtField[];
}

export const KHT_MAGIC = 'KHT1';

export class Reader {
  pos = 0;
  constructor(readonly bytes: Uint8Array) {}
  varint(): number {
    let v = 0; let shift = 0; let b: number;
    do {
      b = this.bytes[this.pos];
      this.pos += 1;
      v += (b & 0x7f) * 2 ** shift;
      shift += 7;
    } while (b & 0x80);
    return v;
  }
  get done(): boolean { return this.pos >= this.bytes.length; }
}

// ignoreBOM: a token may start with U+FEFF, which a default decoder drops.
const utf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

function readStrings(bytes: Uint8Array, prefix: string): string[] {
  const r = new Reader(bytes);
  const out: string[] = [];
  while (!r.done) {
    const head = r.varint();
    const len = Math.floor(head / 2);
    const s = utf8.decode(bytes.subarray(r.pos, r.pos + len));
    r.pos += len;
    out.push(head & 1 ? prefix + s : s);
  }
  return out;
}

function readVarints(bytes: Uint8Array): number[] {
  const r = new Reader(bytes);
  const out: number[] = [];
  while (!r.done) out.push(r.varint());
  return out;
}

// One f64 step away from x (x finite and not 0): the neighbouring bit pattern.
const STEP = new Float64Array(1);
const STEP_WORDS = new Uint32Array(STEP.buffer);
const WORD_LO = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 0 : 1;
const WORD_HI = 1 - WORD_LO;

export function f64Step(x: number, up: boolean): number {
  STEP[0] = x;
  let lo = STEP_WORDS[WORD_LO];
  let hi = STEP_WORDS[WORD_HI];
  if ((x > 0) === up) {
    lo = (lo + 1) >>> 0;
    if (lo === 0) hi = (hi + 1) >>> 0;
  } else {
    if (lo === 0) hi = (hi - 1) >>> 0;
    lo = (lo - 1) >>> 0;
  }
  STEP_WORDS[WORD_LO] = lo;
  STEP_WORDS[WORD_HI] = hi;
  return STEP[0];
}

function readScores(planes: Uint8Array, kinds: Uint8Array, extra: Uint8Array): number[] {
  const n = kinds.length;
  const f32 = new DataView(new ArrayBuffer(4));
  const ex = new DataView(extra.buffer, extra.byteOffset, extra.byteLength);
  const out: number[] = new Array(n);
  let e = 0;
  for (let i = 0; i < n; i += 1) {
    if (kinds[i] === 3) { out[i] = ex.getFloat64(e, true); e += 8; continue; }
    for (let b = 0; b < 4; b += 1) f32.setUint8(b, planes[b * n + i]);
    const f = f32.getFloat32(0, true);
    out[i] = kinds[i] === 0 ? f : f64Step(f, kinds[i] === 1);
  }
  return out;
}

function base64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

// JSON.parse makes "__proto__" an own key; plain assignment would set the prototype instead.
function setKey(obj: Record<string, unknown>, key: string, value: unknown): void {
  if (key === '__proto__') {
    Object.defineProperty(obj, key, { value, writable: true, enumerable: true, configurable: true });
  } else obj[key] = value;
}

/** The tokenizer.json object of an uncompressed KHT payload. */
export function decodeKht(payload: Uint8Array): Record<string, unknown> {
  if (new TextDecoder().decode(payload.subarray(0, 4)) !== KHT_MAGIC) throw new Error('tokenizer.bin: bad magic');
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const descLen = view.getUint32(4, true);
  const desc = JSON.parse(utf8.decode(payload.subarray(8, 8 + descLen))) as KhtDescriptor;
  let pos = 8 + descLen;
  const strings: string[][] = [];
  for (const [fi, f] of desc.fields.entries()) {
    const secs = f.sections.map((len) => { const s = payload.subarray(pos, pos + len); pos += len; return s; });
    let value: unknown;
    if (f.kind === 'dict') {
      const toks = readStrings(secs[0], f.prefix ?? '');
      const ids = f.dense ? null : readVarints(secs[1]);
      const obj: Record<string, unknown> = {};
      toks.forEach((t, i) => setKey(obj, t, ids ? ids[i] : i));
      strings[fi] = toks;
      value = obj;
    } else if (f.kind === 'unigram') {
      const pieces = readStrings(secs[0], f.prefix ?? '');
      const scores = readScores(secs[1], secs[2], secs[3]);
      if (scores.length !== pieces.length) throw new Error('tokenizer.bin: unigram sections disagree');
      strings[fi] = pieces;
      value = pieces.map((p, i) => [p, scores[i]]);
    } else if (f.kind === 'merges') {
      const toks = strings[f.of as number];
      const ix = readVarints(secs[0]);
      const out: unknown[] = new Array(ix.length / 2);
      for (let i = 0; i < out.length; i += 1) {
        const a = toks[ix[2 * i]]; const b = toks[ix[2 * i + 1]];
        out[i] = f.form === 'pair' ? [a, b] : `${a} ${b}`;
      }
      value = out;
    } else if (f.kind === 'base64') {
      value = base64(secs[0]);
    } else {
      value = JSON.parse(utf8.decode(secs[0]));
    }
    let at: Record<string | number, unknown> = desc.head;
    for (const k of f.path.slice(0, -1)) at = at[k] as Record<string | number, unknown>;
    const last = f.path[f.path.length - 1];
    if (typeof last === 'string') setKey(at, last, value); else at[last] = value;
  }
  if (pos !== payload.length) throw new Error('tokenizer.bin: trailing bytes');
  return desc.head;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('tokenizer.bin needs DecompressionStream (Chromium 80, Safari 16.4, Firefox 113)');
  }
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** The tokenizer.json object of a tokenizer file: tokenizer.bin (gzip) or tokenizer.json text. */
export async function parseTokenizerFile(bytes: Uint8Array): Promise<Record<string, unknown>> {
  if (bytes[0] === 0x1f) return decodeKht(await gunzip(bytes));
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
}

// Compact values of the fast path (decodeKhtCompact). Constructors take them directly; plain JSON
// objects and arrays are converted by asDict, asUnigram and asMergePairs, so both shapes build the same tokenizer.

/** A vocab as tokens with their ids. ids null: id = position (dense). */
export class CompactDict {
  constructor(readonly tokens: Tokens, readonly ids: ArrayLike<number> | null) {}
}

/** A unigram vocab as pieces with their scores (scores[i] belongs to pieces[i]). */
export class CompactUnigram {
  constructor(readonly pieces: Tokens, readonly scores: ArrayLike<number>) {}
}

/** BPE merges as index pairs into `tokens` (the token list of the dict or unigram they refer to). */
export class CompactMerges {
  constructor(readonly tokens: Tokens, readonly pairs: ArrayLike<number>, readonly form: 'str' | 'pair') {}
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// Strings in one pass: header and UTF-8 bytes of every string are copied into one buffer, then one
// decode of the whole buffer (ignoreBOM: a token may start with U+FEFF). The UTF-16 offset of each
// string comes from its lead bytes (4-byte sequences count two units). No string is cut here.
function readStringsFast(bytes: Uint8Array, prefix: string): Tokens {
  const prefixBytes = new TextEncoder().encode(prefix);
  let n = 0;
  let total = 0;
  for (let p = 0; p < bytes.length; n += 1) {
    const head = varintAt(bytes, p);
    const len = Math.floor(head / 2);
    total += len + (head & 1 ? prefixBytes.length : 0);
    p = vpos + len;
  }
  const buf = new Uint8Array(total);
  const offs = new Uint32Array(n + 1);
  let o = 0;
  let p = 0;
  for (let i = 0; i < n; i += 1) {
    const head = varintAt(bytes, p);
    const len = Math.floor(head / 2);
    const start = vpos;
    let u = 0;
    if (head & 1) {
      for (let k = 0; k < prefixBytes.length; k += 1) buf[o++] = prefixBytes[k];
      u = prefix.length;
    }
    for (let k = start; k < start + len; k += 1) {
      const c = bytes[k];
      buf[o++] = c;
      if ((c & 0xc0) !== 0x80) u += c >= 0xf0 ? 2 : 1;
    }
    offs[i + 1] = offs[i] + u;
    p = start + len;
  }
  return new Tokens(utf8.decode(buf), offs);
}

// The varint at p (as Reader.varint reads it); its end is left in vpos.
let vpos = 0;
function varintAt(bytes: Uint8Array, p: number): number {
  let v = 0; let mul = 1; let q = p; let b: number;
  do {
    b = bytes[q];
    q += 1;
    v += (b & 0x7f) * mul;
    mul *= 128;
  } while (b & 0x80);
  vpos = q;
  return v;
}

// Varints into an Int32Array (vocab ids). Values beyond int32 throw instead of wrapping.
function readIntsFast(bytes: Uint8Array): Int32Array {
  let count = 0;
  for (let k = 0; k < bytes.length; k += 1) if ((bytes[k] & 0x80) === 0) count += 1;
  const out = new Int32Array(count);
  let p = 0;
  for (let i = 0; i < count; i += 1) {
    const v = varintAt(bytes, p);
    p = vpos;
    if (v > 0x7fffffff) throw new Error('tokenizer.bin: id beyond int32');
    out[i] = v;
  }
  return out;
}

// Varints into a Uint32Array (token positions of merges).
function readU32Fast(bytes: Uint8Array): Uint32Array {
  let count = 0;
  for (let k = 0; k < bytes.length; k += 1) if ((bytes[k] & 0x80) === 0) count += 1;
  const out = new Uint32Array(count);
  let p = 0;
  for (let i = 0; i < count; i += 1) {
    out[i] = varintAt(bytes, p);
    p = vpos;
  }
  return out;
}

// readScores in typed arrays: the four byte planes are transposed into f32 bytes, read as one
// Float32Array, then widened to Float64Array with the same ULP steps and exact f64 values.
function readScoresFast(planes: Uint8Array, kinds: Uint8Array, extra: Uint8Array): Float64Array {
  const n = kinds.length;
  const f32bytes = new Uint8Array(4 * n);
  for (let b = 0; b < 4; b += 1) {
    const plane = b * n;
    for (let i = 0; i < n; i += 1) f32bytes[4 * i + b] = planes[plane + i];
  }
  const f32 = new Float32Array(f32bytes.buffer);
  const ex = new DataView(extra.buffer, extra.byteOffset, extra.byteLength);
  const out = new Float64Array(n);
  let e = 0;
  for (let i = 0; i < n; i += 1) {
    const k = kinds[i];
    if (k === 3) { out[i] = ex.getFloat64(e, true); e += 8; continue; }
    const f = f32[i];
    out[i] = k === 0 ? f : f64Step(f, k === 1);
  }
  return out;
}

/**
 * The tokenizer.json object of an uncompressed KHT payload with compact values: dict, unigram and
 * merges as CompactDict, CompactUnigram and CompactMerges, charsmap as its raw bytes. Equal to
 * decodeKht after expandCompact (tests/tokenizer_bin_fast.test.ts).
 */
export function decodeKhtCompact(payload: Uint8Array): Record<string, unknown> {
  if (!LITTLE_ENDIAN) throw new Error('tokenizer.bin: the fast path needs a little-endian host');
  if (new TextDecoder().decode(payload.subarray(0, 4)) !== KHT_MAGIC) throw new Error('tokenizer.bin: bad magic');
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const descLen = view.getUint32(4, true);
  const desc = JSON.parse(utf8.decode(payload.subarray(8, 8 + descLen))) as KhtDescriptor;
  let pos = 8 + descLen;
  const strings: Tokens[] = [];
  for (const [fi, f] of desc.fields.entries()) {
    const secs = f.sections.map((len) => { const s = payload.subarray(pos, pos + len); pos += len; return s; });
    let value: unknown;
    if (f.kind === 'dict') {
      const toks = readStringsFast(secs[0], f.prefix ?? '');
      value = new CompactDict(toks, f.dense ? null : readIntsFast(secs[1]));
      strings[fi] = toks;
    } else if (f.kind === 'unigram') {
      const pieces = readStringsFast(secs[0], f.prefix ?? '');
      const scores = readScoresFast(secs[1], secs[2], secs[3]);
      if (scores.length !== pieces.length) throw new Error('tokenizer.bin: unigram sections disagree');
      strings[fi] = pieces;
      value = new CompactUnigram(pieces, scores);
    } else if (f.kind === 'merges') {
      value = new CompactMerges(strings[f.of as number], readU32Fast(secs[0]), f.form === 'pair' ? 'pair' : 'str');
    } else if (f.kind === 'base64') {
      value = secs[0];
    } else {
      value = JSON.parse(utf8.decode(secs[0]));
    }
    let at: Record<string | number, unknown> = desc.head;
    for (const k of f.path.slice(0, -1)) at = at[k] as Record<string | number, unknown>;
    const last = f.path[f.path.length - 1];
    if (typeof last === 'string') setKey(at, last, value); else at[last] = value;
  }
  if (pos !== payload.length) throw new Error('tokenizer.bin: trailing bytes');
  return desc.head;
}

/** The plain tokenizer.json object of a compact head (what decodeKht returns). */
export function expandCompact(v: unknown): unknown {
  if (v instanceof CompactDict) {
    const obj: Record<string, unknown> = {};
    for (let i = 0; i < v.tokens.length; i += 1) setKey(obj, v.tokens.tokenAt(i), v.ids ? v.ids[i] : i);
    return obj;
  }
  if (v instanceof CompactUnigram) {
    const out: unknown[] = [];
    for (let i = 0; i < v.pieces.length; i += 1) out.push([v.pieces.tokenAt(i), v.scores[i]]);
    return out;
  }
  if (v instanceof CompactMerges) {
    const out: unknown[] = [];
    for (let i = 0; i + 1 < v.pairs.length; i += 2) {
      const a = v.tokens.tokenAt(v.pairs[i]); const b = v.tokens.tokenAt(v.pairs[i + 1]);
      out.push(v.form === 'pair' ? [a, b] : `${a} ${b}`);
    }
    return out;
  }
  if (v instanceof Uint8Array) return base64(v);
  if (Array.isArray(v)) return v.map(expandCompact);
  if (v !== null && typeof v === 'object') {
    const obj: Record<string, unknown> = {};
    for (const [k, x] of Object.entries(v)) setKey(obj, k, expandCompact(x));
    return obj;
  }
  return v;
}

/** A dict field as CompactDict: compact values as they are, a plain {token: id} object converted. */
export function asDict(v: unknown): CompactDict {
  if (v instanceof CompactDict) return v;
  const obj = v as Record<string, number>;
  const tokens = Object.keys(obj);
  return new CompactDict(Tokens.fromStrings(tokens), tokens.map((t) => obj[t]));
}

/** A unigram field as CompactUnigram: compact values as they are, [piece, score] rows converted. */
export function asUnigram(v: unknown): CompactUnigram {
  if (v instanceof CompactUnigram) return v;
  const rows = v as Array<[string, number]>;
  return new CompactUnigram(Tokens.fromStrings(rows.map((r) => r[0])), rows.map((r) => r[1]));
}

/**
 * BPE merges as index pairs into dict.tokens. Compact merges must index the dict's own token list;
 * plain merges (strings "a b" or pairs [a, b]) are looked up by token, an unknown token throws.
 */
export function asMergePairs(dict: CompactDict, v: unknown): ArrayLike<number> {
  if (v instanceof CompactMerges) {
    if (v.tokens !== dict.tokens) throw new Error('BPE: merges do not index the vocab of this tokenizer');
    return v.pairs;
  }
  const index = new StringIndex(dict.tokens);
  const rows = v as Array<string | [string, string]>;
  const pairs = new Uint32Array(rows.length * 2);
  rows.forEach((m, k) => {
    let a: string; let b: string;
    if (typeof m !== 'string') [a, b] = m;
    else {
      const parts = m.split(' ');
      if (parts.length !== 2) throw new Error(`BPE: invalid merge ${m}`);
      [a, b] = parts;
    }
    const ia = index.indexOf(a); const ib = index.indexOf(b);
    if (ia < 0) throw new Error(`BPE: merge token ${a} is not in the vocab`);
    if (ib < 0) throw new Error(`BPE: merge token ${b} is not in the vocab`);
    pairs[2 * k] = ia; pairs[2 * k + 1] = ib;
  });
  return pairs;
}

/** The id of the token at position i of a dict. */
export function idAt(dict: CompactDict, i: number): number {
  return dict.ids ? dict.ids[i] : i;
}

/** The tokenizer.json object of a tokenizer file, compact values (see decodeKhtCompact). */
export async function parseTokenizerCompact(bytes: Uint8Array): Promise<Record<string, unknown>> {
  if (bytes[0] === 0x1f) return decodeKhtCompact(await gunzip(bytes));
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
}

/**
 * The vocab of a dict: token to id (the id of a token is idAt of its position), and id to token.
 * Ids of a non-dense dict: a table of id to position when they are small, a Map when they are sparse.
 */
export class Vocab {
  readonly index: StringIndex;
  // position of the one-unit token with this code unit, -1 for none (direct lookup of a char)
  private readonly oneUnit: Int32Array;
  private readonly byId: Int32Array | null; // id to position, null when the dict is dense or sparse
  private readonly byIdMap: Map<number, number> | null; // id to position for sparse ids

  constructor(readonly dict: CompactDict) {
    this.index = new StringIndex(dict.tokens);
    this.oneUnit = new Int32Array(0x10000).fill(-1);
    const { blob, offs } = dict.tokens;
    for (let i = 0; i < dict.tokens.length; i += 1) {
      if (offs[i + 1] - offs[i] === 1) this.oneUnit[blob.charCodeAt(offs[i])] = i;
    }
    if (dict.ids === null) {
      this.byId = null;
      this.byIdMap = null;
      return;
    }
    let maxId = 0;
    for (let i = 0; i < dict.tokens.length; i += 1) {
      const id = idAt(dict, i);
      if (Number.isInteger(id) && id > maxId) maxId = id;
    }
    if (maxId < 2 ** 22) {
      this.byId = new Int32Array(maxId + 1).fill(-1);
      this.byIdMap = null;
      for (let i = 0; i < dict.tokens.length; i += 1) {
        const id = idAt(dict, i);
        if (Number.isInteger(id) && id >= 0) this.byId[id] = i;
      }
    } else {
      this.byId = null;
      this.byIdMap = new Map();
      for (let i = 0; i < dict.tokens.length; i += 1) this.byIdMap.set(idAt(dict, i), i);
    }
  }

  // the id of token s, or undefined
  id(s: string): number | undefined {
    const i = this.index.indexOf(s);
    return i < 0 ? undefined : idAt(this.dict, i);
  }

  has(s: string): boolean {
    return this.index.indexOf(s) >= 0;
  }

  // the id of a char (a string of one or two code units), as id()
  idOfChar(ch: string): number | undefined {
    if (ch.length !== 1) return this.id(ch);
    const i = this.oneUnit[ch.charCodeAt(0)];
    return i < 0 ? undefined : idAt(this.dict, i);
  }

  // the id of token a followed by token b without its first `from` units
  idOfConcat(a: number, b: number, from: number): number | undefined {
    const i = this.index.indexOfConcat(a, b, from);
    return i < 0 ? undefined : idAt(this.dict, i);
  }

  // the token with this id, or undefined
  tokenOfId(id: number): string | undefined {
    if (this.byIdMap !== null) {
      const at = this.byIdMap.get(id);
      return at === undefined ? undefined : this.dict.tokens.tokenAt(at);
    }
    if (this.byId === null) return id >= 0 && id < this.dict.tokens.length ? this.dict.tokens.tokenAt(id) : undefined;
    const i = id >= 0 && id < this.byId.length ? this.byId[id] : -1;
    return i < 0 ? undefined : this.dict.tokens.tokenAt(i);
  }
}
