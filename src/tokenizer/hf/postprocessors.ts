// Post-processors of HF tokenizers 0.22.2: TemplateProcessing, RobertaProcessing and
// ByteLevel (tokenizers/src/processors, Apache-2.0). They put special tokens and type
// ids around one or two sequences.

import { isWhitespace } from './normalizers.ts';

// One tokenized sequence. offsets holds [start, end] pairs flat; wordIds -1 for none.
// tokens is the token text, needed by the post-processors that trim offsets.
export interface Sequence {
  ids: number[];
  offsets: number[];
  wordIds: number[];
  tokens: string[];
}

export interface PostProcessor {
  addedTokens(pair: boolean): number;
  apply(a: Sequence, b: Sequence | null): Encoding;
}

export interface Encoding {
  ids: number[];
  typeIds: number[];
  attentionMask: number[];
  // [start, end] per token in UTF-16 units of the input text; special tokens (0, 0)
  offsets: Array<[number, number]>;
  // index of the pre-token inside its sequence; -1 for special tokens
  wordIds: number[];
  // 0 first text, 1 second text, -1 special tokens
  sequenceIds: number[];
  specialTokensMask: number[];
}

type Piece =
  | { kind: 'sequence'; which: 0 | 1; typeId: number }
  | { kind: 'special'; ids: number[]; typeId: number };

export class TemplateProcessing implements PostProcessor {
  private single: Piece[];
  private pair: Piece[];

  private constructor(single: Piece[], pair: Piece[]) {
    this.single = single;
    this.pair = pair;
  }

  static fromJson(json: Record<string, unknown>): TemplateProcessing {
    const specials = (json.special_tokens ?? {}) as Record<string, { ids: number[] }>;
    const parse = (items: unknown): Piece[] => (items as Array<Record<string, Record<string, unknown>>>)
      .map((item) => {
        if (item.Sequence) {
          const id = item.Sequence.id;
          if (id !== 'A' && id !== 'B') throw new Error(`TemplateProcessing: sequence id ${String(id)}`);
          return { kind: 'sequence', which: id === 'A' ? 0 : 1, typeId: item.Sequence.type_id as number } as Piece;
        }
        if (item.SpecialToken) {
          const tok = specials[item.SpecialToken.id as string];
          if (!tok) throw new Error(`TemplateProcessing: unknown special token ${String(item.SpecialToken.id)}`);
          return { kind: 'special', ids: tok.ids, typeId: item.SpecialToken.type_id as number } as Piece;
        }
        throw new Error(`TemplateProcessing: template item ${Object.keys(item).join(',')}`);
      });
    return new TemplateProcessing(parse(json.single), parse(json.pair));
  }

  // Number of special tokens the template adds.
  addedTokens(pair: boolean): number {
    let n = 0;
    for (const p of pair ? this.pair : this.single) if (p.kind === 'special') n += p.ids.length;
    return n;
  }

  apply(a: Sequence, b: Sequence | null): Encoding {
    const template = b ? this.pair : this.single;
    const out: Encoding = {
      ids: [], typeIds: [], attentionMask: [], offsets: [], wordIds: [], sequenceIds: [],
      specialTokensMask: [],
    };
    for (const piece of template) {
      if (piece.kind === 'special') {
        for (const id of piece.ids) {
          out.ids.push(id);
          out.typeIds.push(piece.typeId);
          out.attentionMask.push(1);
          out.offsets.push([0, 0]);
          out.wordIds.push(-1);
          out.sequenceIds.push(-1);
          out.specialTokensMask.push(1);
        }
        continue;
      }
      const seq = piece.which === 0 ? a : b;
      if (!seq) throw new Error('TemplateProcessing: template needs a second sequence');
      for (let i = 0; i < seq.ids.length; i += 1) {
        out.ids.push(seq.ids[i]);
        out.typeIds.push(piece.typeId);
        out.attentionMask.push(1);
        out.offsets.push([seq.offsets[2 * i], seq.offsets[2 * i + 1]]);
        out.wordIds.push(seq.wordIds[i]);
        out.sequenceIds.push(piece.which);
        out.specialTokensMask.push(0);
      }
    }
    return out;
  }
}

const SPACE_CHAR = '\u0120'; // the byte-level char of 0x20

// process_offsets of tokenizers/src/processors/byte_level.rs: leading and trailing spaces
// of a token are cut from its offsets, except the one the pre-tokenizer added at the start.
function trimOffsets(seq: Sequence, addPrefixSpace: boolean): number[] {
  const offsets = seq.offsets.slice();
  for (let i = 0; i < seq.ids.length; i += 1) {
    const token = [...seq.tokens[i]];
    let leading = 0;
    while (leading < token.length && (token[leading] === SPACE_CHAR || isWhitespace(token[leading]))) leading += 1;
    let trailing = 0;
    while (trailing < token.length
      && (token[token.length - 1 - trailing] === SPACE_CHAR || isWhitespace(token[token.length - 1 - trailing]))) {
      trailing += 1;
    }
    if (leading === 0 && trailing === 0) continue;
    let start = offsets[2 * i];
    let end = offsets[2 * i + 1];
    if (leading > 0) {
      const isFirst = i === 0 || start === 0;
      if (isFirst && addPrefixSpace && leading === 1) leading = 0;
      start = Math.min(start + leading, end);
    }
    if (trailing > 0 && end >= trailing) end = Math.max(end - trailing, start);
    offsets[2 * i] = start;
    offsets[2 * i + 1] = end;
  }
  return offsets;
}

function special(out: Encoding, id: number) {
  out.ids.push(id);
  out.typeIds.push(0);
  out.attentionMask.push(1);
  out.offsets.push([0, 0]);
  out.wordIds.push(-1);
  out.sequenceIds.push(-1);
  out.specialTokensMask.push(1);
}

function body(out: Encoding, seq: Sequence, offsets: number[], which: 0 | 1, typeId: number) {
  for (let i = 0; i < seq.ids.length; i += 1) {
    out.ids.push(seq.ids[i]);
    out.typeIds.push(typeId);
    out.attentionMask.push(1);
    out.offsets.push([offsets[2 * i], offsets[2 * i + 1]]);
    out.wordIds.push(seq.wordIds[i]);
    out.sequenceIds.push(which);
    out.specialTokensMask.push(0);
  }
}

function emptyEncoding(): Encoding {
  return {
    ids: [], typeIds: [], attentionMask: [], offsets: [], wordIds: [], sequenceIds: [],
    specialTokensMask: [],
  };
}

// RobertaProcessing: <s> A </s> for one text, <s> A </s></s> B </s> for two; every type id is 0.
export class RobertaProcessing implements PostProcessor {
  private sep: number;
  private cls: number;
  private trim: boolean;
  private addPrefixSpace: boolean;

  constructor(sep: number, cls: number, trim: boolean, addPrefixSpace: boolean) {
    this.sep = sep;
    this.cls = cls;
    this.trim = trim;
    this.addPrefixSpace = addPrefixSpace;
  }

  static fromJson(json: Record<string, unknown>): RobertaProcessing {
    return new RobertaProcessing(
      (json.sep as [string, number])[1], (json.cls as [string, number])[1],
      json.trim_offsets !== false, json.add_prefix_space !== false);
  }

  addedTokens(pair: boolean): number {
    return pair ? 4 : 2;
  }

  apply(a: Sequence, b: Sequence | null): Encoding {
    const out = emptyEncoding();
    special(out, this.cls);
    body(out, a, this.trim ? trimOffsets(a, this.addPrefixSpace) : a.offsets, 0, 0);
    special(out, this.sep);
    if (b) {
      special(out, this.sep);
      body(out, b, this.trim ? trimOffsets(b, this.addPrefixSpace) : b.offsets, 1, 0);
      special(out, this.sep);
    }
    return out;
  }
}

// ByteLevel as post-processor: no special tokens, offsets trimmed, type ids 0 and 1.
export class ByteLevelProcessing implements PostProcessor {
  private trim: boolean;
  private addPrefixSpace: boolean;

  constructor(trim: boolean, addPrefixSpace: boolean) {
    this.trim = trim;
    this.addPrefixSpace = addPrefixSpace;
  }

  static fromJson(json: Record<string, unknown>): ByteLevelProcessing {
    return new ByteLevelProcessing(json.trim_offsets !== false, json.add_prefix_space !== false);
  }

  addedTokens(): number {
    return 0;
  }

  apply(a: Sequence, b: Sequence | null): Encoding {
    const out = emptyEncoding();
    body(out, a, this.trim ? trimOffsets(a, this.addPrefixSpace) : a.offsets, 0, 0);
    if (b) body(out, b, this.trim ? trimOffsets(b, this.addPrefixSpace) : b.offsets, 1, 1);
    return out;
  }
}

export function parsePostProcessor(json: Record<string, unknown>): PostProcessor {
  const type = String(json?.type);
  switch (type) {
    case 'TemplateProcessing':
      return TemplateProcessing.fromJson(json);
    case 'RobertaProcessing':
      return RobertaProcessing.fromJson(json);
    case 'ByteLevel':
      return ByteLevelProcessing.fromJson(json);
    default:
      throw new Error(`tokenizer.json: unsupported post_processor type ${type}`);
  }
}
