// BPE model of HF tokenizers 0.22.2 without dropout (tokenizers/src/models/bpe/model.rs
// and word.rs, Apache-2.0): merge_word, ignore_merges, byte_fallback, fuse_unk,
// continuing_subword_prefix, end_of_word_suffix.

import { asDict, asMergePairs, idAt, Vocab, type CompactDict } from '../bin.ts';
import { PairTable } from '../pairs.ts';
import { utf8Length } from './normalized.ts';
import type { PieceSpan } from './wordpiece.ts';

interface Symbol {
  id: number;
  // UTF-8 length of the source chars it covers; a char that has no symbol (no unk token) adds none
  len: number;
}

export interface BpeOptions {
  unkToken: string | null;
  continuingSubwordPrefix: string | null;
  endOfWordSuffix: string | null;
  fuseUnk: boolean;
  byteFallback: boolean;
  ignoreMerges: boolean;
}

export class Bpe {
  private vocab: Vocab;
  // pair key a * base + b (base = largest id + 1, so no two pairs share a key) to [rank, merged id]
  private base: number;
  // pair of ids to the rank of its merge; mergedIds[rank] is the id of the merged token
  private ranks: PairTable;
  private mergedIds: Int32Array;
  private opt: BpeOptions;

  // dict: the vocab (compact or plain, see asDict); pairs: merges as index pairs into dict.tokens
  // (asMergePairs), so the merge loop works on ids and never builds a string per pair.
  constructor(dict: CompactDict, pairs: ArrayLike<number>, opt: BpeOptions) {
    this.vocab = new Vocab(dict);
    this.opt = opt;
    let maxId = 0;
    for (let i = 0; i < dict.tokens.length; i += 1) {
      const id = idAt(dict, i);
      if (!Number.isInteger(id) || id < 0) {
        throw new Error(`BPE: vocab id ${id} of ${dict.tokens.tokenAt(i)} is not a non-negative integer`);
      }
      maxId = Math.max(maxId, id);
    }
    this.base = maxId + 1;
    if (this.base * this.base > Number.MAX_SAFE_INTEGER) {
      throw new Error(`BPE: largest vocab id ${maxId} is too large for a collision-free pair key`);
    }
    const prefixLen = opt.continuingSubwordPrefix === null ? 0 : opt.continuingSubwordPrefix.length;
    const count = pairs.length / 2;
    this.ranks = new PairTable(count);
    this.mergedIds = new Int32Array(count);
    for (let k = 0, rank = 0; k + 1 < pairs.length; k += 2, rank += 1) {
      const aId = idAt(dict, pairs[k]);
      const bId = idAt(dict, pairs[k + 1]);
      const merged = this.vocab.idOfConcat(pairs[k], pairs[k + 1], prefixLen);
      if (merged === undefined) {
        const a = dict.tokens.tokenAt(pairs[k]);
        throw new Error(`BPE: merged token ${a + dict.tokens.tokenAt(pairs[k + 1]).slice(prefixLen)} is not in the vocab`);
      }
      // a later duplicate pair replaces the earlier one, as collecting into a map does
      this.ranks.set(aId, bId, rank);
      this.mergedIds[rank] = merged;
    }
    if (opt.unkToken !== null && !this.vocab.has(opt.unkToken)) {
      throw new Error(`BPE: unk token ${opt.unkToken} is not in the vocab`);
    }
  }

  static fromJson(model: Record<string, unknown>): Bpe {
    if (model.dropout !== null && model.dropout !== undefined) {
      throw new Error('BPE: dropout is not supported');
    }
    const dict = asDict(model.vocab);
    return new Bpe(dict, asMergePairs(dict, model.merges), {
      unkToken: (model.unk_token as string | null | undefined) ?? null,
      continuingSubwordPrefix: (model.continuing_subword_prefix as string | null | undefined) ?? null,
      endOfWordSuffix: (model.end_of_word_suffix as string | null | undefined) ?? null,
      fuseUnk: model.fuse_unk === true,
      byteFallback: model.byte_fallback === true,
      ignoreMerges: model.ignore_merges === true,
    });
  }

  // chars: the code points of one word.
  tokenize(chars: string[]): PieceSpan[] {
    if (chars.length === 0) return [];
    const word = chars.join('');
    if (this.opt.ignoreMerges) {
      const id = this.vocab.id(word);
      if (id !== undefined) return [{ id, start: 0, end: chars.length, value: word }];
    }
    // offsets are cumulative symbol lengths in bytes, turned into code point ranges
    const byteChar: number[] = [];
    chars.forEach((c, i) => {
      for (let k = utf8Length(c); k > 0; k -= 1) byteChar.push(i);
    });
    const out: PieceSpan[] = [];
    let pos = 0;
    for (const s of this.mergeWord(chars)) {
      const from = byteChar[Math.min(pos, byteChar.length - 1)];
      const to = byteChar[Math.min(pos + s.len, byteChar.length) - 1] + 1;
      out.push({ id: s.id, start: from, end: Math.max(to, from + 1), value: this.vocab.tokenOfId(s.id) });
      pos += s.len;
    }
    return out;
  }

  private mergeWord(chars: string[]): Symbol[] {
    const o = this.opt;
    const symbols: Symbol[] = [];
    let unk: Symbol | null = null;
    for (let i = 0; i < chars.length; i += 1) {
      let s = chars[i];
      if (i > 0 && o.continuingSubwordPrefix !== null) s = o.continuingSubwordPrefix + s;
      if (i === chars.length - 1 && o.endOfWordSuffix !== null) s += o.endOfWordSuffix;
      const id = this.vocab.id(s);
      if (id !== undefined) {
        if (unk) {
          symbols.push(unk);
          unk = null;
        }
        symbols.push({ id, len: utf8Length(chars[i]) });
        continue;
      }
      if (o.byteFallback) {
        const ids: number[] = [];
        let all = true;
        for (const b of new TextEncoder().encode(s)) {
          const t = this.vocab.id(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
          if (t === undefined) {
            all = false;
            break;
          }
          ids.push(t);
        }
        if (all) {
          for (const t of ids) symbols.push({ id: t, len: 1 });
          continue;
        }
      }
      if (o.unkToken !== null) {
        const unkId = this.vocab.id(o.unkToken) as number;
        if (unk && o.fuseUnk) {
          unk = { id: unk.id, len: unk.len + utf8Length(chars[i]) };
        } else {
          if (unk) symbols.push(unk);
          unk = { id: unkId, len: utf8Length(chars[i]) };
        }
      }
    }
    if (unk) symbols.push(unk);
    return this.mergeAll(symbols);
  }

  // Lowest rank first, the leftmost pair on a tie, until no pair is left.
  private mergeAll(symbols: Symbol[]): Symbol[] {
    let list = symbols;
    for (;;) {
      let bestRank = Infinity;
      let bestPos = -1;
      let bestId = 0;
      for (let i = 0; i + 1 < list.length; i += 1) {
        const rank = this.ranks.get(list[i].id, list[i + 1].id);
        if (rank !== -1 && rank < bestRank) {
          bestRank = rank;
          bestPos = i;
          bestId = this.mergedIds[rank];
        }
      }
      if (bestPos < 0) return list;
      list = [
        ...list.slice(0, bestPos),
        { id: bestId, len: list[bestPos].len + list[bestPos + 1].len },
        ...list.slice(bestPos + 2),
      ];
    }
  }
}

