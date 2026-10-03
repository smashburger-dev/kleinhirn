// Unigram model of HF tokenizers 0.22.2 (tokenizers/src/models/unigram/model.rs,
// encode_optimized and tokenize, Apache-2.0): Viterbi over the piece scores, unknown
// chars as unk with the lowest score minus 10, fused unk runs, byte_fallback.

import type { PieceSpan } from './wordpiece.ts';

const K_UNK_PENALTY = 10.0;

export class Unigram {
  private scores = new Map<string, number>();
  private ids = new Map<string, number>();
  private minScore = Infinity;
  private maxChars = 0;
  private unkId: number | null;
  private byteFallback: boolean;

  constructor(vocab: Array<[string, number]>, unkId: number | null, byteFallback: boolean) {
    this.unkId = unkId;
    this.byteFallback = byteFallback;
    if (unkId !== null && (vocab.length === 0 || unkId >= vocab.length)) {
      throw new Error('Unigram: unk_id is not in the vocab');
    }
    vocab.forEach(([token, score], id) => {
      // a repeated piece keeps its last id, the score of the lookup by id is that entry's own
      this.ids.set(token, id);
      this.scores.set(token, score);
      if (score < this.minScore) this.minScore = score;
      this.maxChars = Math.max(this.maxChars, [...token].length);
    });
    // scores by id for repeated pieces
    this.byId = vocab.map((v) => v[1]);
  }

  private byId: number[];

  static fromJson(model: Record<string, unknown>): Unigram {
    return new Unigram(
      model.vocab as Array<[string, number]>,
      (model.unk_id as number | null | undefined) ?? null,
      model.byte_fallback === true);
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
      let piece = '';
      for (let len = 1; len <= this.maxChars && start + len <= n; len += 1) {
        piece += chars[start + len - 1];
        const id = this.ids.get(piece);
        if (id === undefined) continue;
        const end = start + len;
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
      const id = this.ids.get(text);
      if (id !== undefined) {
        out.push({ id, start: s, end: e });
        continue;
      }
      if (this.byteFallback) {
        const bytes = new TextEncoder().encode(text);
        const tokens: number[] = [];
        let all = true;
        for (const b of bytes) {
          const t = this.ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
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
