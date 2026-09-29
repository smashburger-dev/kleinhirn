// HF tokenizer.json pipeline for the mmBERT/Gemma BPE tokenizer (Julia 1):
// Replace normalizer (" " -> "▁"), Metaspace pretokenizer (prepend always,
// split on the replacement char), BPE model with byte_fallback and fuse_unk.
// encodeIds matches tokenizer(text, add_special_tokens=False); the Julia
// sequence layout inserts <bos>/<sep>/<mask> ids itself.

interface AddedToken { id: number; content: string; special: boolean }
interface TokenizerJson {
  model: { vocab: Record<string, number>; merges: [string, string][]; unk_token?: string };
  added_tokens: AddedToken[];
}

const META = '▁';

function byteTokens(char: string): string[] {
  const out: string[] = [];
  for (const b of new TextEncoder().encode(char)) {
    out.push(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
  }
  return out;
}

export class BpeTokenizer {
  private vocab: Map<string, number>;
  private ranks: Map<string, number>;
  private ids: Map<number, string>;
  private added: Map<string, number>;
  private addedList: string[];
  unkId: number;
  padTokenId = 0;
  bosTokenId = 2;
  eosTokenId = 1;
  maskTokenId = 4;
  maskToken = '<mask>';

  constructor(json: TokenizerJson) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.ids = new Map([...this.vocab].map(([k, v]) => [v, k]));
    this.ranks = new Map();
    json.model.merges.forEach(([a, b], rank) => {
      this.ranks.set(`${a} ${b}`, rank);
    });
    this.unkId = this.vocab.get(json.model.unk_token ?? '<unk>') ?? 3;
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
  private bpe(piece: string): string[] {
    const symbols: string[] = [];
    for (const ch of piece) {
      if (this.vocab.has(ch)) symbols.push(ch);
      else symbols.push(...byteTokens(ch));
    }
    for (;;) {
      let bestRank = Infinity;
      let bestPair = '';
      for (let i = 0; i + 1 < symbols.length; i += 1) {
        const rank = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`) ?? Infinity;
        if (rank < bestRank) {
          bestRank = rank;
          bestPair = `${symbols[i]} ${symbols[i + 1]}`;
        }
      }
      if (bestRank === Infinity) break;
      const merged = bestPair.replace(' ', '');
      const next: string[] = [];
      for (let i = 0; i < symbols.length; i += 1) {
        if (i + 1 < symbols.length
            && `${symbols[i]} ${symbols[i + 1]}` === bestPair) {
          next.push(merged);
          i += 1;
        } else {
          next.push(symbols[i]);
        }
      }
      symbols.length = 0;
      symbols.push(...next);
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
      for (const sym of this.bpe(m[0])) {
        ids.push(this.vocab.get(sym) ?? this.unkId);
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
    return this.ids.get(id);
  }
}
