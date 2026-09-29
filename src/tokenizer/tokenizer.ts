// HF tokenizer.json pipeline for the GLiNER2.5 DeBERTa-v3 tokenizer:
// normalizer sequence (whitespace collapse, NFC, right strip), added-token
// matching, Metaspace pretokenizer, Unigram Viterbi. No [CLS]/[SEP] are
// added by tokenize; the GLiNER2 schema input inserts its own markers.

import { buildUnigram, unigramEncode, type UnigramVocab } from './unigram.ts';

interface AddedToken { id: number; content: string; special: boolean }
interface TokenizerJson {
  model: { vocab: [string, number][]; unk_id?: number };
  added_tokens: AddedToken[];
}

const COLLAPSE_WS = /\s{2,}|[\n\r\t]/g;
const METASPACE = '▁';

export class HfTokenizer {
  private unigram: UnigramVocab;
  private added: Map<string, number>;
  private addedList: string[];
  padTokenId = 0;

  constructor(json: TokenizerJson) {
    this.unigram = buildUnigram(json.model.vocab, json.model.unk_id ?? 3);
    this.added = new Map();
    this.addedList = [];
    for (const t of json.added_tokens) {
      this.added.set(t.content, t.id);
      this.addedList.push(t.content);
      if (t.content === '[PAD]') this.padTokenId = t.id;
    }
    this.addedList.sort((a, b) => b.length - a.length);
  }

  private normalize(text: string): string {
    return text.replace(COLLAPSE_WS, ' ').normalize('NFC').replace(/\s+$/, '');
  }

  // Token ids for one pretokenized span (already normalized).
  private encodePretoken(pretoken: string): number[] {
    if (this.added.has(pretoken)) return [this.added.get(pretoken) as number];
    // Metaspace splits on the ASCII space only (verified against the
    // tokenizer_ids goldens: U+00A0 stays inside the pretoken and becomes
    // UNK, it does not act as a separator).
    const ids: number[] = [];
    let start = 0;
    for (let i = 0; i <= pretoken.length; i += 1) {
      const isWs = i < pretoken.length && pretoken[i] === ' ';
      if (isWs || i === pretoken.length) {
        if (i > start) {
          ids.push(...unigramEncode(this.unigram, METASPACE + pretoken.slice(start, i)));
        }
        start = i + 1;
      }
    }
    return ids;
  }

  // tokenizer.tokenize-equivalent: added tokens are matched first on the
  // normalized input (specials use normalized=false in tokenizer.json, but
  // the schema path calls tokenize on already-atomic strings, so matching on
  // the normalized text covers every schema token).
  encodeIds(text: string): number[] {
    const norm = this.normalize(text);
    const ids: number[] = [];
    let i = 0;
    while (i < norm.length) {
      let hit = -1;
      let hitLen = 0;
      for (const tok of this.addedList) {
        const j = norm.indexOf(tok, i);
        if (j !== -1 && (hit === -1 || j < hit)) {
          hit = j;
          hitLen = tok.length;
        }
      }
      const end = hit === -1 ? norm.length : hit;
      if (end > i) ids.push(...this.encodePretoken(norm.slice(i, end)));
      if (hit === -1) break;
      ids.push(this.added.get(norm.slice(hit, hit + hitLen)) as number);
      i = hit + hitLen;
    }
    return ids;
  }
}
