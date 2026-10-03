// Normalizers of HF tokenizers 0.22.2 (BertNormalizer, NFC, NFKC, Strip, Replace,
// Precompiled, Sequence) with the alignment to the original text kept per
// normalized character (see normalized.ts). Ported from tokenizers/src/normalizers
// (Apache-2.0) and huggingface/spm_precompiled (Apache-2.0).

import { fromText, matchRegex, replaceMatches, transform, utf8Length, type Normalized } from './normalized.ts';

export type { Normalized } from './normalized.ts';

export interface BertNormalizerConfig {
  cleanText: boolean;
  handleChineseChars: boolean;
  stripAccents: boolean | null;
  lowercase: boolean;
}

const OTHER = /^[\p{Cc}\p{Cf}\p{Cs}\p{Co}]$/u; // Rust is_other as tokenizers 0.22.2 has it: unassigned (Cn) stays
const WHITE_SPACE = /^\p{White_Space}$/u;
const MARK_NONSPACING = /^\p{Mn}$/u;

export function isWhitespace(c: string): boolean {
  const code = c.charCodeAt(0);
  if (code < 128) return code === 32 || (code >= 9 && code <= 13);
  return WHITE_SPACE.test(c);
}

// Format chars assigned after the Unicode tables of tokenizers 0.22.2: that build sees them as
// unassigned and keeps them (measured with every scalar value against BertNormalizer).
function isNewFormatChar(cp: number): boolean {
  return cp === 0x890 || cp === 0x891 || cp === 0x8e2 || cp === 0x110cd || (cp >= 0x13430 && cp <= 0x1343f);
}

// Control in the BERT sense: category Cc, Cf, Cs, Co except tab, line feed, carriage return.
export function isControl(c: string): boolean {
  if (c === '\t' || c === '\n' || c === '\r') return false;
  return OTHER.test(c) && !isNewFormatChar(c.codePointAt(0) as number);
}

export function isChineseChar(cp: number): boolean {
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)
    || (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f)
    || (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b920 && cp <= 0x2ceaf)
    || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
}

export function parseBertNormalizer(json: Record<string, unknown>): BertNormalizerConfig {
  return {
    cleanText: json.clean_text !== false,
    handleChineseChars: json.handle_chinese_chars !== false,
    stripAccents: (json.strip_accents ?? null) as boolean | null,
    lowercase: json.lowercase !== false,
  };
}

// text is a piece of the original string that starts at index base.
export function normalizeBert(
  text: string, base: number, cfg: BertNormalizerConfig,
): Normalized {
  return bertNormalize(fromText(text, base), cfg);
}

export function bertNormalize(n: Normalized, cfg: BertNormalizerConfig): Normalized {
  let chars: string[] = [];
  let starts: number[] = [];
  let ends: number[] = [];
  for (let i = 0; i < n.chars.length; i += 1) {
    const c = n.chars[i];
    // clean_text: drop NUL, U+FFFD and controls, then whitespace becomes a space
    if (cfg.cleanText && (c === '\0' || c === '�' || isControl(c))) continue;
    chars.push(cfg.cleanText && isWhitespace(c) ? ' ' : c);
    starts.push(n.starts[i]);
    ends.push(n.ends[i]);
  }
  if (cfg.handleChineseChars) {
    const c2: string[] = [];
    const s2: number[] = [];
    const e2: number[] = [];
    for (let i = 0; i < chars.length; i += 1) {
      const ch = chars[i];
      if (isChineseChar(ch.codePointAt(0) as number)) {
        // inserted spaces take the alignment of the character itself
        c2.push(' ', ch, ' ');
        s2.push(starts[i], starts[i], starts[i]);
        e2.push(ends[i], ends[i], ends[i]);
      } else {
        c2.push(ch);
        s2.push(starts[i]);
        e2.push(ends[i]);
      }
    }
    chars = c2; starts = s2; ends = e2;
  }
  if (cfg.stripAccents ?? cfg.lowercase) {
    const c2: string[] = [];
    const s2: number[] = [];
    const e2: number[] = [];
    for (let i = 0; i < chars.length; i += 1) {
      for (const d of chars[i].normalize('NFD')) {
        if (MARK_NONSPACING.test(d)) continue;
        c2.push(d);
        s2.push(starts[i]);
        e2.push(ends[i]);
      }
    }
    chars = c2; starts = s2; ends = e2;
  }
  if (cfg.lowercase) {
    const c2: string[] = [];
    const s2: number[] = [];
    const e2: number[] = [];
    for (let i = 0; i < chars.length; i += 1) {
      // per code point like Rust char::to_lowercase (no final-sigma context)
      for (const l of chars[i].toLowerCase()) {
        c2.push(l);
        s2.push(starts[i]);
        e2.push(ends[i]);
      }
    }
    chars = c2; starts = s2; ends = e2;
  }
  return { chars, starts, ends, origin: n.origin };
}

export type NormalizerFn = (n: Normalized) => Normalized;

// Canonical combining class, as far as the composition needs it. JS has no accessor, so the two
// questions the algorithm asks are answered with NFD of probe strings: is ccc greater than 0, and
// is ccc(a) smaller than ccc(b). U+0345 has the highest class (240), which the first probe cannot see.
const isMarkMemo = new Map<string, boolean>();
const lessMemo = new Map<string, boolean>();

function isMark(c: string): boolean {
  let v = isMarkMemo.get(c);
  if (v === undefined) {
    v = c === '\u0345' || `x\u0345${c}`.normalize('NFD') !== `x\u0345${c}`;
    isMarkMemo.set(c, v);
  }
  return v;
}

// ccc(a) < ccc(b): NFD puts a before b after reordering only when b was first and had the larger class
function cccLess(a: string, b: string): boolean {
  const key = a + b;
  let v = lessMemo.get(key);
  if (v === undefined) {
    v = `x${b}${a}`.normalize('NFD') !== `x${b}${a}`;
    lessMemo.set(key, v);
  }
  return v;
}

const pairMemo = new Map<string, string | null>();

function composePair(a: string, b: string): string | null {
  const key = a + '\u0000' + b;
  let v = pairMemo.get(key);
  if (v === undefined) {
    const n = (a + b).normalize('NFC');
    v = [...n].length === 1 ? n : null;
    pairMemo.set(key, v);
  }
  return v;
}

// NFC and NFKC of unicode-normalization-alignments, as used by tokenizers: decompose every source char
// (changes 0 for the first part, 1 for the rest), put the marks in canonical order with their changes,
// compose with the standard algorithm; a composed char gets the sum of the changes minus one per merge.
// The result goes through transform().
function composeNormalize(n: Normalized, form: 'NFC' | 'NFKC'): Normalized {
  const decomposition = form === 'NFC' ? 'NFD' : 'NFKD';
  const d: Array<[string, number]> = [];
  for (const c of n.chars) {
    [...c.normalize(decomposition)].forEach((x, i) => d.push([x, i === 0 ? 0 : 1]));
  }
  for (let i = 0; i < d.length;) {
    if (!isMark(d[i][0])) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < d.length && isMark(d[j][0])) j += 1;
    for (let a = i + 1; a < j; a += 1) {
      const cur = d[a];
      let b = a - 1;
      while (b >= i && cccLess(cur[0], d[b][0])) {
        d[b + 1] = d[b];
        b -= 1;
      }
      d[b + 1] = cur;
    }
    i = j;
  }
  const out: Array<[string, number]> = [];
  let starter = -1;
  // class of the last char kept: null is "above every class" (a leading mark), a string is that char
  let last: string | null = null;
  let lastIsStarter = false;
  for (const [x, change] of d) {
    const xMark = isMark(x);
    if (starter >= 0) {
      const composite = composePair(out[starter][0], x);
      const allowed = lastIsStarter || (last !== null && xMark && cccLess(last, x));
      if (composite !== null && allowed) {
        out[starter] = [composite, out[starter][1] + change - 1];
        continue;
      }
    }
    if (!xMark) starter = out.length;
    last = x;
    lastIsStarter = !xMark;
    out.push([x, change]);
    if (out.length === 1 && xMark) {
      last = null;
      starter = -1;
    }
  }
  return transform(n, out);
}

// Normalizer Strip: the count of leading and trailing White_Space chars is cut.
function stripNormalize(n: Normalized, left: boolean, right: boolean): Normalized {
  const count = n.chars.length;
  let lead = 0;
  if (left) while (lead < count && isWhitespace(n.chars[lead])) lead += 1;
  let trail = 0;
  if (right) while (trail < count && isWhitespace(n.chars[count - 1 - trail])) trail += 1;
  if (lead === 0 && trail === 0) return n;
  const a = Math.min(lead, count);
  const b = Math.max(a, count - trail);
  return {
    chars: n.chars.slice(a, b), starts: n.starts.slice(a, b), ends: n.ends.slice(a, b), origin: n.origin,
  };
}

// Oniguruma \w is Letter, Mark, Decimal_Number, Connector_Punctuation; \s is White_Space.
const ONIG_WORD = '\\p{L}\\p{M}\\p{Nd}\\p{Pc}';
const ONIG_SPACE = '\\p{White_Space}';

// Translates the escapes whose meaning differs between Oniguruma and JavaScript with the u flag.
export function oniguruma(pattern: string): string {
  let out = '';
  let inClass = false;
  for (let i = 0; i < pattern.length; i += 1) {
    const c = pattern[i];
    if (c === '\\') {
      const e = pattern[i + 1];
      i += 1;
      const plain: Record<string, string> = { w: ONIG_WORD, s: ONIG_SPACE };
      const negated: Record<string, string> = { W: ONIG_WORD, S: ONIG_SPACE };
      if (e in plain) out += inClass ? plain[e] : `[${plain[e]}]`;
      else if (e in negated) {
        if (inClass) throw new Error(`Replace: unsupported \\${e} inside a character class in ${pattern}`);
        out += `[^${negated[e]}]`;
      } else if (e === 'd' || e === 'D') {
        throw new Error(`Replace: unsupported \\${e} in ${pattern}`);
      } else if ((e === 'b' || e === 'B') && !inClass) {
        // JavaScript's \b is ASCII, Oniguruma's is Unicode; no safe translation without lookarounds
        throw new Error(`Replace: unsupported \\${e} outside a character class in ${pattern}`);
      } else out += c + e;
    } else {
      if (c === '[') inClass = true;
      else if (c === ']') inClass = false;
      out += c;
    }
  }
  return out;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

class Precompiled {
  private trie: Uint32Array;
  private blob: Uint8Array;
  private cache = new Map<string, string | null>();
  private encoder = new TextEncoder();
  private decoder = new TextDecoder();

  constructor(base64: string) {
    const raw = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const trieBytes = view.getUint32(0, true);
    this.trie = new Uint32Array(trieBytes / 4);
    for (let i = 0; i < this.trie.length; i += 1) this.trie[i] = view.getUint32(4 + 4 * i, true);
    this.blob = raw.subarray(4 + trieBytes);
  }

  // spm_precompiled Precompiled::transform: the first (shortest) prefix match of chunk.
  transform(chunk: string): string | null {
    const hit = this.cache.get(chunk);
    if (hit !== undefined) return hit;
    const index = this.firstPrefix(this.encoder.encode(chunk));
    let out: string | null = null;
    if (index !== null) {
      let end = index;
      while (end < this.blob.length && this.blob[end] !== 0) end += 1;
      out = this.decoder.decode(this.blob.subarray(index, end));
    }
    this.cache.set(chunk, out);
    return out;
  }

  // DoubleArray::common_prefix_search, first result only.
  private firstPrefix(key: Uint8Array): number | null {
    const a = this.trie;
    const offset = (u: number) => (u >>> 10) << ((u & 0x200) >> 6);
    let node = 0;
    let unit = a[node];
    node ^= offset(unit);
    for (const c of key) {
      if (c === 0) break;
      node ^= c;
      unit = a[node];
      if (unit === undefined || ((unit & 0x800000ff) >>> 0) !== c) return null;
      node ^= offset(unit);
      if ((unit >>> 8) & 1) return a[node] & 0x7fffffff;
    }
    return null;
  }

  normalize(n: Normalized, segmenter: Intl.Segmenter): Normalized {
    const tr: Array<[string, number]> = [];
    let modified = false;
    const push = (oldPart: string, newPart: string) => {
      const oldCount = [...oldPart].length;
      const cs = [...newPart];
      const diff = cs.length - oldCount;
      for (const c of cs) tr.push([c, 0]);
      if (diff > 0) {
        for (let k = 0; k < diff; k += 1) tr[tr.length - 1 - k][1] = 1;
      } else if (diff < 0 && tr.length) {
        tr[tr.length - 1][1] += diff;
      }
    };
    const text = n.chars.join('');
    for (const { segment } of segmenter.segment(text)) {
      if (utf8Bytes(segment) < 6) {
        const norm = this.transform(segment);
        if (norm !== null) {
          modified = true;
          push(segment, norm);
          continue;
        }
      }
      for (const c of segment) {
        const norm = this.transform(c);
        if (norm !== null) {
          modified = true;
          push(c, norm);
        } else {
          tr.push([c, 0]);
        }
      }
    }
    return modified ? transform(n, tr) : n;
  }
}

function utf8Bytes(s: string): number {
  let n = 0;
  for (const c of s) n += utf8Length(c);
  return n;
}

export function parseNormalizer(json: Record<string, unknown> | null): NormalizerFn | null {
  if (json === null || json === undefined) return null;
  const type = String(json.type);
  switch (type) {
    case 'Sequence': {
      const fns = (json.normalizers as Array<Record<string, unknown>>)
        .map((j) => parseNormalizer(j)) as NormalizerFn[];
      return (n) => fns.reduce((acc, f) => f(acc), n);
    }
    case 'BertNormalizer': {
      const cfg = parseBertNormalizer(json);
      return (n) => bertNormalize(n, cfg);
    }
    case 'NFC':
    case 'NFKC':
      return (n) => composeNormalize(n, type as 'NFC' | 'NFKC');
    case 'Strip': {
      const left = json.strip_left !== false;
      const right = json.strip_right !== false;
      return (n) => stripNormalize(n, left, right);
    }
    case 'Replace': {
      const pattern = json.pattern as { String?: string; Regex?: string };
      const source = pattern.String !== undefined ? escapeRegExp(pattern.String)
        : oniguruma(pattern.Regex as string);
      if (source === '') throw new Error('Replace: empty pattern');
      const re = new RegExp(source, 'gu');
      const content = json.content as string;
      return (n) => replaceMatches(n, matchRegex(n.chars, re), content);
    }
    case 'Precompiled': {
      const pre = new Precompiled(json.precompiled_charsmap as string);
      const segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' });
      return (n) => pre.normalize(n, segmenter);
    }
    default:
      throw new Error(`tokenizer.json: unsupported normalizer type ${type}`);
  }
}
