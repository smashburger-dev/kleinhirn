// Pre-tokenizers of HF tokenizers 0.22.2: BertPreTokenizer, WhitespaceSplit, ByteLevel,
// Metaspace and Sequence. Ported from tokenizers/src/pre_tokenizers (Apache-2.0).
// A pre-tokenizer maps one normalized piece to its words; empty words are dropped.

import {
  matchPredicate, matchRegex, prepend, replaceMatches, slice, splitMatches, utf8Length,
  type Normalized,
} from './normalized.ts';
import { isWhitespace } from './normalizers.ts';

export type PreTokenizerFn = (n: Normalized) => Normalized[];

// Half-open range of normalized character indices.
export interface Split {
  start: number;
  end: number;
}

const PUNCTUATION = /^\p{P}$/u;
const ASCII_PUNCTUATION = /^[!-/:-@[-`{-~]$/;

// ASCII punctuation (including the symbols $+<=>^`|~) and Unicode P*.
export function isBertPunctuation(c: string): boolean {
  if (c.charCodeAt(0) < 128) return ASCII_PUNCTUATION.test(c);
  return PUNCTUATION.test(c);
}

export function bertPreTokenize(n: Normalized): Split[] {
  const out: Split[] = [];
  let start = -1;
  for (let i = 0; i < n.chars.length; i += 1) {
    const c = n.chars[i];
    if (isWhitespace(c)) {
      if (start >= 0) out.push({ start, end: i });
      start = -1;
    } else if (isBertPunctuation(c)) {
      if (start >= 0) out.push({ start, end: i });
      out.push({ start: i, end: i + 1 });
      start = -1;
    } else if (start < 0) {
      start = i;
    }
  }
  if (start >= 0) out.push({ start, end: n.chars.length });
  return out;
}

export function bertWords(n: Normalized): Normalized[] {
  return bertPreTokenize(n).map((sp) => slice(n, sp.start, sp.end));
}

// GPT-2 pattern; in Oniguruma \s is White_Space, which JS \s is not (it has U+FEFF, lacks U+0085).
const BYTE_LEVEL_RE = new RegExp(
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+"
  + '|\\p{White_Space}+(?!\\P{White_Space})|\\p{White_Space}+', 'gu');

// GPT-2 byte to char table.
const BYTE_CHAR: string[] = (() => {
  const bs: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) bs.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) bs.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) bs.push(b);
  const table: string[] = new Array(256);
  for (const b of bs) table[b] = String.fromCharCode(b);
  let k = 0;
  for (let b = 0; b < 256; b += 1) {
    if (table[b] === undefined) {
      table[b] = String.fromCharCode(256 + k);
      k += 1;
    }
  }
  return table;
})();

export const SPACE_CHAR = BYTE_CHAR[0x20];

const encoder = new TextEncoder();

// Every byte of a char becomes one char, all with the alignment of the source char.
function toByteChars(n: Normalized): Normalized {
  const chars: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  for (let i = 0; i < n.chars.length; i += 1) {
    const bytes = utf8Length(n.chars[i]) === 1 ? [n.chars[i].charCodeAt(0)] : encoder.encode(n.chars[i]);
    for (const b of bytes) {
      chars.push(BYTE_CHAR[b]);
      starts.push(n.starts[i]);
      ends.push(n.ends[i]);
    }
  }
  return { chars, starts, ends, origin: n.origin };
}

export function byteLevel(addPrefixSpace: boolean, useRegex: boolean): PreTokenizerFn {
  return (piece) => {
    let n = piece;
    if (addPrefixSpace && n.chars[0] !== ' ') n = prepend(n, ' ');
    const words = useRegex
      ? splitMatches(n, matchRegex(n.chars, BYTE_LEVEL_RE), 'isolated') : [n];
    return words.map(toByteChars);
  };
}

export function metaspace(
  replacement: string, scheme: 'always' | 'first' | 'never', split: boolean,
): PreTokenizerFn {
  const isSpace = (c: string) => c === ' ';
  return (piece) => {
    let n = replaceMatches(piece, matchPredicate(piece.chars, isSpace), replacement);
    if (scheme === 'always' && n.chars[0] !== replacement) n = prepend(n, replacement);
    else if (scheme === 'first' && n.chars[0] !== replacement && (n.origin ?? 0) === 0) {
      n = prepend(n, replacement);
    }
    return split
      ? splitMatches(n, matchPredicate(n.chars, (c) => c === replacement), 'merged_with_next') : [n];
  };
}

export function whitespaceSplit(n: Normalized): Normalized[] {
  return splitMatches(n, matchPredicate(n.chars, isWhitespace), 'removed');
}

export function parsePreTokenizer(json: Record<string, unknown> | null): PreTokenizerFn | null {
  if (json === null || json === undefined) return null;
  const type = String(json.type);
  switch (type) {
    case 'Sequence': {
      const fns = (json.pretokenizers as Array<Record<string, unknown>>)
        .map((j) => parsePreTokenizer(j)) as PreTokenizerFn[];
      return (n) => fns.reduce(
        (pieces, f) => pieces.flatMap((p) => f(p)).filter((p) => p.chars.length > 0), [n]);
    }
    case 'BertPreTokenizer':
      return bertWords;
    case 'WhitespaceSplit':
      return whitespaceSplit;
    case 'ByteLevel':
      return byteLevel(json.add_prefix_space !== false, json.use_regex !== false);
    case 'Metaspace': {
      const replacement = json.replacement as string;
      if (json.add_prefix_space === false && json.prepend_scheme !== undefined
        && json.prepend_scheme !== 'never') {
        throw new Error('Metaspace: add_prefix_space does not match declared prepend_scheme');
      }
      const scheme = json.add_prefix_space === false ? 'never'
        : ((json.prepend_scheme as string | undefined) ?? 'always') as 'always' | 'first' | 'never';
      return metaspace(replacement, scheme, json.split !== false);
    }
    default:
      throw new Error(`tokenizer.json: unsupported pre_tokenizer type ${type}`);
  }
}
