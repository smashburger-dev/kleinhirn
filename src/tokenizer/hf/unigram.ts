// Unigram model of HF tokenizers 0.22.2 (tokenizers/src/models/unigram/model.rs,
// encode_optimized and tokenize, Apache-2.0): Viterbi over the piece scores, unknown
// chars as unk with the lowest score minus 10, fused unk runs, byte_fallback.

import { asUnigram, type CompactUnigram } from '../bin.ts';
import { FNV_BASIS, fnv, StringIndex, type Tokens } from '../strtab.ts';
import type { PieceSpan } from './wordpiece.ts';

const K_UNK_PENALTY = 10.0;

// Code points in the UTF-16 range [a, b) of s, counted without spreading.
function codePointCount(s: string, a: number, b: number): number {
  let n = 0;
  for (let i = a; i < b; i += 1) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c < 0xdc00 && i + 1 < b) {
      const d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d < 0xe000) i += 1;
    }
    n += 1;
  }
  return n;
}

export class Unigram {
  // piece to id (the position; a repeated piece keeps its last position)
  private index: StringIndex;
  private minScore = Infinity;
  private maxChars = 0;
  private unkId: number | null;
  private byteFallback: boolean;
  // scores by id
  private byId: Float64Array;

  constructor(vocab: CompactUnigram, unkId: number | null, byteFallback: boolean) {
    this.unkId = unkId;
    this.byteFallback = byteFallback;
    const n = vocab.pieces.length;
    if (unkId !== null && (n === 0 || unkId >= n)) {
      throw new Error('Unigram: unk_id is not in the vocab');
    }
    this.byId = Float64Array.from(vocab.scores);
    const { blob, offs } = vocab.pieces;
    for (let id = 0; id < n; id += 1) {
      const score = this.byId[id];
      if (score < this.minScore) this.minScore = score;
      this.maxChars = Math.max(this.maxChars, codePointCount(blob, offs[id], offs[id + 1]));
    }
    this.index = new StringIndex(vocab.pieces);
  }

  static fromJson(model: Record<string, unknown>): Unigram {
    return new Unigram(
      asUnigram(model.vocab),
      (model.unk_id as number | null | undefined) ?? null,
      model.byte_fallback === true);
  }

  private idOf(s: string): number | undefined {
    const i = this.index.indexOf(s);
    return i < 0 ? undefined : i;
  }

  // chars: the code points of one word.
  tokenize(chars: string[]): PieceSpan[] {
    if (chars.length === 0) return [];
    const n = chars.length;
    const unkScore = this.minScore - K_UNK_PENALTY;
    const bestScore = new Array<number>(n + 1).fill(0);
    const startsAt = new Array<number>(n + 1).fill(-1);
    const bestId = new Array<number>(n + 1).fill(0);
    for (let start = 0; start < n; start += 1) {
      const here = bestScore[start];
      let hasSingle = false;
      // the hash of chars[start .. end) grows by one char per length; no string is built
      let h = FNV_BASIS;
      for (let len = 1; len <= this.maxChars && start + len <= n; len += 1) {
        const end = start + len;
        h = fnv(h, chars[end - 1]);
        const id = this.index.indexOfChars(chars, start, end, h);
        if (id < 0) continue;
        const candidate = this.byId[id] + here;
        if (startsAt[end] < 0 || candidate > bestScore[end]) {
          bestScore[end] = candidate;
          startsAt[end] = start;
          bestId[end] = id;
        }
        if (len === 1) hasSingle = true;
      }
      if (!hasSingle) {
        if (this.unkId === null) throw new Error('Unigram: the vocab has no unk id');
        const candidate = unkScore + here;
        if (startsAt[start + 1] < 0 || candidate > bestScore[start + 1]) {
          bestScore[start + 1] = candidate;
          startsAt[start + 1] = start;
          bestId[start + 1] = this.unkId;
        }
      }
    }
    // walk back, fusing runs of unk nodes into one piece
    const pieces: Array<[number, number]> = [];
    let end = n;
    let runEnd = -1;
    let runStart = -1;
    while (end > 0) {
      const s = startsAt[end];
      if (this.unkId !== null && bestId[end] === this.unkId) {
        if (runEnd < 0) runEnd = end;
        runStart = s;
      } else {
        if (runEnd >= 0) {
          pieces.push([runStart, runEnd]);
          runEnd = -1;
        }
        pieces.push([s, end]);
      }
      end = s;
    }
    if (runEnd >= 0) pieces.push([runStart, runEnd]);
    pieces.reverse();
    const out: PieceSpan[] = [];
    for (const [s, e] of pieces) {
      const text = chars.slice(s, e).join('');
      const id = this.idOf(text);
      if (id !== undefined) {
        out.push({ id, start: s, end: e });
        continue;
      }
      if (this.byteFallback) {
        const bytes = new TextEncoder().encode(text);
        const tokens: number[] = [];
        let all = true;
        for (const b of bytes) {
          const t = this.idOf(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
          if (t === undefined) {
            all = false;
            break;
          }
          tokens.push(t);
        }
        if (all) {
          for (const t of tokens) out.push({ id: t, start: s, end: e });
          continue;
        }
      }
      if (this.unkId === null) throw new Error('Unigram: the vocab has no unk id');
      out.push({ id: this.unkId, start: s, end: e });
    }
    return out;
  }
}
