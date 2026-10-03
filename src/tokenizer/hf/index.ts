// General tokenizer from tokenizer.json (K28.3): normalizers, pre-tokenizers, the
// WordPiece, BPE and Unigram models, TemplateProcessing, RobertaProcessing and the
// ByteLevel post-processor, added tokens with lstrip, rstrip, single_word and
// normalized. Other component types throw when loading, with the type name. The
// reference is HF tokenizers 0.22.2; encode returns what AutoTokenizer returns with
// offsets in UTF-16 units.

import { Bpe } from './bpe.ts';
import { cpIndex, fromText, join, slice, type Normalized } from './normalized.ts';
import { isWhitespace, parseNormalizer, type NormalizerFn } from './normalizers.ts';
import {
  parsePostProcessor, type Encoding, type PostProcessor, type Sequence,
} from './postprocessors.ts';
import { parsePreTokenizer, type PreTokenizerFn } from './pretokenizers.ts';
import { truncateLengths, type TruncationStrategy } from './truncation.ts';
import { Unigram } from './unigram.ts';
import { WordPiece, type PieceSpan } from './wordpiece.ts';

export type { Encoding } from './postprocessors.ts';
export type { TruncationStrategy } from './truncation.ts';

export interface EncodeOptions {
  // Total length including the special tokens. Without it nothing is cut.
  maxLength?: number;
  // Default 'longest_first' when maxLength is set; false disables truncation.
  truncation?: TruncationStrategy | boolean;
}

interface Model {
  tokenize(chars: string[]): PieceSpan[];
}

interface AddedToken {
  id: number;
  content: string;
  lstrip: boolean;
  rstrip: boolean;
  singleWord: boolean;
  normalized: boolean;
  special: boolean;
}

// Added tokens matched in one pass over a string, leftmost and longest first.
interface TokenSet {
  tokens: AddedToken[];
  regex: RegExp | null;
}

type Part = { text: Normalized } | { token: AddedToken; span: Normalized };

const WORD = /^[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]$/u;

function typeOf(node: unknown): string | null {
  return node === null || node === undefined ? null : String((node as { type?: unknown }).type);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function utf8Bytes(s: string): number {
  return new TextEncoder().encode(s).length;
}

function buildModel(model: Record<string, unknown>): Model {
  const type = typeOf(model);
  if (type === 'WordPiece') return WordPiece.fromJson(model);
  if (type === 'BPE') return Bpe.fromJson(model);
  if (type === 'Unigram') return Unigram.fromJson(model);
  throw new Error(`tokenizer.json: unsupported model type ${type}`);
}

interface Found {
  token: AddedToken | null;
  start: number;
  end: number;
}

// AddedVocabulary::find_matches. A match can grow over the spaces next to it (lstrip,
// rstrip) and is dropped when single_word is set and a word char touches it. Units are the
// UTF-16 units of sentence. A null token is the text between two matches.
function findMatches(sentence: string, set: TokenSet): Found[] {
  if (sentence.length === 0) return [{ token: null, start: 0, end: 0 }];
  const out: Found[] = [];
  let startOffset = 0;
  if (set.regex) {
    set.regex.lastIndex = 0;
    for (let m = set.regex.exec(sentence); m !== null; m = set.regex.exec(sentence)) {
      // exec does not advance on a zero-length match; no token has an empty pattern (fromJson), this keeps a loop out
      if (m[0].length === 0) {
        set.regex.lastIndex += 1;
        continue;
      }
      let start = m.index;
      let stop = m.index + m[0].length;
      // one capture group per token: the group that matched names the token
      const token = set.tokens[m.slice(1).findIndex((g) => g !== undefined)];
      if (token.singleWord) {
        const before = [...sentence.slice(0, start)].pop();
        const after = [...sentence.slice(stop)][0];
        const startSpace = start === 0 || !WORD.test(before as string);
        const stopSpace = stop === sentence.length || !WORD.test(after as string);
        if (!stopSpace || !startSpace) continue;
      }
      if (token.lstrip) {
        let p = start;
        while (p > 0) {
          const c = [...sentence.slice(Math.max(0, p - 2), p)].pop() as string;
          if (!isWhitespace(c)) break;
          p -= c.length;
        }
        start = Math.max(p, startOffset);
      }
      if (token.rstrip) {
        let p = stop;
        while (p < sentence.length) {
          const c = String.fromCodePoint(sentence.codePointAt(p) as number);
          if (!isWhitespace(c)) break;
          p += c.length;
        }
        stop = p;
      }
      if (startOffset < start) out.push({ token: null, start: startOffset, end: start });
      out.push({ token, start, end: stop });
      startOffset = stop;
    }
  }
  if (startOffset !== sentence.length) out.push({ token: null, start: startOffset, end: sentence.length });
  return out;
}

function tokenSet(tokens: AddedToken[], patterns: string[]): TokenSet {
  if (tokens.length === 0) return { tokens, regex: null };
  // longest pattern first (UTF-8 bytes, stable), so the alternation picks the longest match
  const order = tokens.map((_, i) => i)
    .sort((x, y) => utf8Bytes(patterns[y]) - utf8Bytes(patterns[x]) || x - y);
  return {
    tokens: order.map((i) => tokens[i]),
    regex: new RegExp(order.map((i) => `(${escapeRegExp(patterns[i])})`).join('|'), 'g'),
  };
}

export class JsonTokenizer {
  private normalizer: NormalizerFn | null;
  private preTokenizer: PreTokenizerFn | null;
  private model: Model;
  private post: PostProcessor;
  private plain: TokenSet;
  private normalizedSet: TokenSet;

  private constructor(
    normalizer: NormalizerFn | null, preTokenizer: PreTokenizerFn | null, model: Model,
    post: PostProcessor, plain: TokenSet, normalizedSet: TokenSet,
  ) {
    this.normalizer = normalizer;
    this.preTokenizer = preTokenizer;
    this.model = model;
    this.post = post;
    this.plain = plain;
    this.normalizedSet = normalizedSet;
  }

  static fromJson(json: Record<string, unknown>): JsonTokenizer {
    const normalizer = parseNormalizer(json.normalizer as Record<string, unknown> | null);
    const preTokenizer = parsePreTokenizer(json.pre_tokenizer as Record<string, unknown> | null);
    const model = buildModel(json.model as Record<string, unknown>);
    const post = parsePostProcessor(json.post_processor as Record<string, unknown>);
    const added = ((json.added_tokens ?? []) as Array<Record<string, unknown>>).map((t): AddedToken => ({
      id: t.id as number, content: t.content as string, lstrip: t.lstrip === true, rstrip: t.rstrip === true,
      singleWord: t.single_word === true, normalized: t.normalized === true, special: t.special === true,
    })).filter((t) => t.content !== '');
    // special tokens come first, as in AddedVocabulary::refresh_added_tokens
    const ordered = [...added.filter((t) => t.special), ...added.filter((t) => !t.special)];
    const plainTokens = ordered.filter((t) => !t.normalized);
    const normalizedTokens = ordered.filter((t) => t.normalized);
    const patterns = normalizedTokens.map((t) => {
      const n = fromText(t.content);
      return join(normalizer ? normalizer(n) : n);
    });
    patterns.forEach((pattern, i) => {
      if (pattern === '') {
        throw new Error(`tokenizer.json: added token ${normalizedTokens[i].id} is empty after normalization`);
      }
    });
    return new JsonTokenizer(
      normalizer, preTokenizer, model, post,
      tokenSet(plainTokens, plainTokens.map((t) => t.content)),
      tokenSet(normalizedTokens, patterns));
  }

  static fromString(text: string): JsonTokenizer {
    return JsonTokenizer.fromJson(JSON.parse(text));
  }

  // Text pieces and added tokens of one text in order (AddedVocabulary::extract_and_normalize):
  // first the tokens that are not normalized, matched on the original text, then the normalized
  // ones on the normalized rest.
  private split(text: string): Part[] {
    const parts: Part[] = [];
    for (const m of findMatches(text, this.plain)) {
      const raw = fromText(text.slice(m.start, m.end), m.start);
      if (m.token) {
        parts.push({ token: m.token, span: raw });
        continue;
      }
      const n = this.normalizer ? this.normalizer(raw) : raw;
      if (n.chars.length === 0) continue;
      const map = cpIndex(n.chars);
      for (const f of findMatches(join(n), this.normalizedSet)) {
        const a = map[f.start];
        const b = map[f.end];
        if (a === b) continue;
        parts.push(f.token ? { token: f.token, span: slice(n, a, b) } : { text: slice(n, a, b) });
      }
    }
    return parts;
  }

  // Ids, offsets and pre-token indices of one text before truncation.
  private encodeSequence(text: string): Sequence {
    const seq: Sequence = { ids: [], offsets: [], wordIds: [], tokens: [] };
    let word = 0;
    for (const part of this.split(text)) {
      if ('token' in part) {
        const sp = part.span;
        seq.ids.push(part.token.id);
        seq.offsets.push(sp.starts[0], sp.ends[sp.chars.length - 1]);
        seq.wordIds.push(word);
        seq.tokens.push(join(sp));
        word += 1;
        continue;
      }
      const words = this.preTokenizer ? this.preTokenizer(part.text) : [part.text];
      for (const w of words) {
        if (w.chars.length === 0) continue;
        for (const p of this.model.tokenize(w.chars)) {
          seq.ids.push(p.id);
          seq.offsets.push(w.starts[p.start], w.ends[p.end - 1]);
          seq.wordIds.push(word);
          seq.tokens.push(p.value ?? '');
        }
        word += 1;
      }
    }
    return seq;
  }

  encode(text: string, pair?: string | null, options: EncodeOptions = {}): Encoding {
    let a = this.encodeSequence(text);
    let b = pair === undefined || pair === null ? null : this.encodeSequence(pair);
    const strategy = options.truncation === false ? null
      : options.truncation === undefined || options.truncation === true ? 'longest_first'
        : options.truncation;
    if (options.maxLength !== undefined && strategy) {
      const max = options.maxLength - this.post.addedTokens(b !== null);
      const [k1, k2] = truncateLengths(a.ids.length, b ? b.ids.length : null, max, strategy);
      const cut = (s: Sequence, k: number): Sequence => (k >= s.ids.length ? s : {
        ids: s.ids.slice(0, k), offsets: s.offsets.slice(0, 2 * k), wordIds: s.wordIds.slice(0, k),
        tokens: s.tokens.slice(0, k),
      });
      a = cut(a, k1);
      if (b) b = cut(b, k2);
    }
    return this.post.apply(a, b);
  }
}
