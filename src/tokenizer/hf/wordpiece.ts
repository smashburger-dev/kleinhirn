// WordPiece model: longest match first, continuation pieces with a prefix.

import { asDict, Vocab } from '../bin.ts';

export interface PieceSpan {
  id: number;
  // half-open range of characters inside the word
  start: number;
  end: number;
  // token text, only where a post-processor needs it (BPE)
  value?: string;
}

export class WordPiece {
  private vocab: Vocab;
  private unkToken: string;
  private prefix: string;
  private maxInputCharsPerWord: number;

  constructor(
    vocab: Vocab, unkToken: string, prefix: string, maxInputCharsPerWord: number,
  ) {
    this.vocab = vocab;
    this.unkToken = unkToken;
    this.prefix = prefix;
    this.maxInputCharsPerWord = maxInputCharsPerWord;
    if (!vocab.has(unkToken)) throw new Error(`WordPiece: unk token ${unkToken} is not in the vocab`);
  }

  static fromJson(model: Record<string, unknown>): WordPiece {
    const vocab = new Vocab(asDict(model.vocab));
    return new WordPiece(
      vocab,
      (model.unk_token as string | undefined) ?? '[UNK]',
      (model.continuing_subword_prefix as string | undefined) ?? '##',
      (model.max_input_chars_per_word as number | undefined) ?? 100,
    );
  }

  id(token: string): number | undefined {
    return this.vocab.id(token);
  }

  // chars: the code points of one word. Counted in code points, not UTF-16 units.
  tokenize(chars: string[]): PieceSpan[] {
    const unk: PieceSpan = { id: this.vocab.id(this.unkToken) as number, start: 0, end: chars.length };
    if (chars.length > this.maxInputCharsPerWord) return [unk];
    const out: PieceSpan[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found: PieceSpan | null = null;
      while (start < end) {
        const piece = (start > 0 ? this.prefix : '') + chars.slice(start, end).join('');
        const id = this.vocab.id(piece);
        if (id !== undefined) {
          found = { id, start, end };
          break;
        }
        end -= 1;
      }
      if (!found) return [unk];
      out.push(found);
      start = end;
    }
    return out;
  }
}
