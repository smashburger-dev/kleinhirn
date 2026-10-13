// HF tokenizer.json pipeline for the mmBERT/Gemma BPE tokenizer (Julia 1):
// Replace normalizer (" " -> "▁"), Metaspace pretokenizer (prepend always,
// split on the replacement char), BPE model with byte_fallback and fuse_unk.
// encodeIds matches tokenizer(text, add_special_tokens=False); the Julia
// sequence layout inserts <bos>/<sep>/<mask> ids itself.
// The merge loop works on vocab ids. Merged ids are resolved on first use (a merge that never
// fires costs no string), so the constructor builds no string per merge.

import { asDict, asMergePairs, idAt, Vocab } from './bin.ts';
import { PairTable } from './pairs.ts';

interface AddedToken { id: number; content: string; special: boolean }
interface TokenizerJson {
  // compact (parseTokenizerCompact) or the {token: id} object of tokenizer.json
  model: { vocab: unknown; merges: unknown; unk_token?: string };
  added_tokens: AddedToken[];
}

const META = '▁';
// a symbol with no vocab id (a char and its byte tokens are not in the vocab); it never merges
const NO_ID = -1;
// merged id not resolved yet
const UNRESOLVED = -2;

function byteTokens(char: string): string[] {
  const out: string[] = [];
  for (const b of new TextEncoder().encode(char)) {
    out.push(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
  }
  return out;
}

export class BpeTokenizer {
  private vocab: Vocab;
  // pair of ids to the rank of its merge (lower merges first)
  private ranks: PairTable;
  private pairs: ArrayLike<number>; // token positions of each merge, by rank
  // merged vocab id by rank: UNRESOLVED, NO_ID (the merged token is not in the vocab) or the id
  private mergedIds: Int32Array;
  private added: Map<string, number>;
  private addedList: string[];
  unkId: number;
  padTokenId = 0;
  bosTokenId = 2;
  eosTokenId = 1;
  maskTokenId = 4;
  maskToken = '<mask>';

  constructor(json: TokenizerJson) {
    const dict = asDict(json.model.vocab);
    this.vocab = new Vocab(dict);
    this.pairs = asMergePairs(dict, json.model.merges);
    const count = this.pairs.length / 2;
    this.ranks = new PairTable(count);
    this.mergedIds = new Int32Array(count).fill(UNRESOLVED);
    for (let rank = 0; rank < count; rank += 1) {
      this.ranks.set(idAt(dict, this.pairs[2 * rank]), idAt(dict, this.pairs[2 * rank + 1]), rank);
    }
    this.unkId = this.vocab.id(json.model.unk_token ?? '<unk>') ?? 3;
    this.added = new Map();
    this.addedList = [];
    for (const t of json.added_tokens) {
      this.added.set(t.content, t.id);
      this.addedList.push(t.content);
      if (t.content === '<pad>') this.padTokenId = t.id;
      else if (t.content === '<bos>') this.bosTokenId = t.id;
      else if (t.content === '<eos>') this.eosTokenId = t.id;
      else if (t.content === '<mask>') this.maskTokenId = t.id;
    }
    this.addedList.sort((a, b) => b.length - a.length);
  }

  // One Metaspace piece, e.g. "▁▁hello". Initial symbols: chars in vocab,
  // unknown chars decompose into their UTF-8 <0xNN> byte tokens. Then the
  // classic merge loop: repeatedly merge the lowest-rank adjacent pair
  // (all non-overlapping occurrences, left to right) until no pair merges.
  // Returns vocab ids (NO_ID for a symbol without one).
  // The merged id of a rank, looked up on first use.
  private mergedOf(rank: number): number {
    if (this.mergedIds[rank] === UNRESOLVED) {
      const id = this.vocab.idOfConcat(this.pairs[2 * rank], this.pairs[2 * rank + 1], 0);
      this.mergedIds[rank] = id === undefined ? NO_ID : id;
    }
    return this.mergedIds[rank];
  }

  private bpe(piece: string): number[] {
    let symbols: number[] = [];
    for (const ch of piece) {
      const id = this.vocab.idOfChar(ch);
      if (id !== undefined) symbols.push(id);
      else for (const t of byteTokens(ch)) symbols.push(this.vocab.id(t) ?? NO_ID);
    }
    for (;;) {
      let bestRank = -1;
      let bestA = NO_ID;
      let bestB = NO_ID;
      for (let i = 0; i + 1 < symbols.length; i += 1) {
        if (symbols[i] === NO_ID || symbols[i + 1] === NO_ID) continue;
        const rank = this.ranks.get(symbols[i], symbols[i + 1]);
        if (rank !== -1 && (bestRank === -1 || rank < bestRank)) {
          bestRank = rank;
          bestA = symbols[i];
          bestB = symbols[i + 1];
        }
      }
      if (bestRank === -1) break;
      const merged = this.mergedOf(bestRank);
      const next: number[] = [];
      for (let i = 0; i < symbols.length; i += 1) {
        if (i + 1 < symbols.length && symbols[i] === bestA && symbols[i + 1] === bestB) {
          next.push(merged);
          i += 1;
        } else {
          next.push(symbols[i]);
        }
      }
      symbols = next;
    }
    return symbols;
  }

  private encodePlain(text: string): number[] {
    // Normalizer: Replace " " with "▁". Metaspace: every "▁" starts a new
    // piece ("▁" + following non-"▁" chars); prepend "▁" only if the first
    // piece does not already start with one (prepend_scheme "always").
    let norm = text.replaceAll(' ', META);
    if (!norm.startsWith(META)) norm = META + norm;
    const ids: number[] = [];
    for (const m of norm.matchAll(/▁[^▁]*/g)) {
      for (const id of this.bpe(m[0])) {
        ids.push(id === NO_ID ? this.unkId : id);
      }
    }
    // fuse_unk: collapse runs of <unk>.
    const fused: number[] = [];
    for (const id of ids) {
      if (id === this.unkId && fused[fused.length - 1] === this.unkId) continue;
      fused.push(id);
    }
    return fused;
  }

  // tokenizer(text, add_special_tokens=False): added tokens (incl. <mask>,
  // <bos>, <eos>) are matched before the model path, leftmost-longest.
  encodeIds(text: string): number[] {
    const ids: number[] = [];
    let i = 0;
    while (i < text.length) {
      let hit = -1;
      let hitLen = 0;
      for (const tok of this.addedList) {
        const j = text.indexOf(tok, i);
        if (j !== -1 && (hit === -1 || j < hit)) {
          hit = j;
          hitLen = tok.length;
        }
      }
      const end = hit === -1 ? text.length : hit;
      if (end > i) ids.push(...this.encodePlain(text.slice(i, end)));
      if (hit === -1) break;
      ids.push(this.added.get(text.slice(hit, hit + hitLen)) as number);
      i = hit + hitLen;
    }
    return ids;
  }

  idToToken(id: number): string | undefined {
    return this.vocab.tokenOfId(id);
  }
}
