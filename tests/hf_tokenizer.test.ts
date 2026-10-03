import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JsonTokenizer } from '../src/tokenizer/hf/index.ts';
import { truncateLengths } from '../src/tokenizer/hf/truncation.ts';

const VOCAB = [
  '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world', 'un', '##aff', '##able', 'cafe',
  ',', '!', 'a', 'b', '中', '文', 'i', '##\u0307',
];

function tinyJson(lowercase: boolean, extra: Record<string, unknown> = {}) {
  const model: Record<string, unknown> = {
    type: 'WordPiece', unk_token: '[UNK]', continuing_subword_prefix: '##',
    max_input_chars_per_word: 12,
    vocab: Object.fromEntries(VOCAB.map((t, i) => [t, i])),
  };
  return {
    version: '1.0', truncation: null, padding: null,
    added_tokens: ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'].map((content, id) => ({
      id, content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true,
    })),
    normalizer: {
      type: 'BertNormalizer', clean_text: true, handle_chinese_chars: true,
      strip_accents: null, lowercase,
    },
    pre_tokenizer: { type: 'BertPreTokenizer' },
    post_processor: {
      type: 'TemplateProcessing',
      single: [
        { SpecialToken: { id: '[CLS]', type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } },
        { SpecialToken: { id: '[SEP]', type_id: 0 } }],
      pair: [
        { SpecialToken: { id: '[CLS]', type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } },
        { SpecialToken: { id: '[SEP]', type_id: 0 } }, { Sequence: { id: 'B', type_id: 1 } },
        { SpecialToken: { id: '[SEP]', type_id: 1 } }],
      special_tokens: {
        '[CLS]': { id: '[CLS]', ids: [2], tokens: ['[CLS]'] },
        '[SEP]': { id: '[SEP]', ids: [3], tokens: ['[SEP]'] },
      },
    },
    model, ...extra,
  };
}

const tok = JsonTokenizer.fromJson(tinyJson(true));

test('single text: ids, offsets, word ids, special tokens', () => {
  const e = tok.encode('Hello, unaffable world!');
  assert.deepEqual(e.ids, [2, 5, 11, 7, 8, 9, 6, 12, 3]);
  assert.deepEqual(e.offsets, [[0, 0], [0, 5], [5, 6], [7, 9], [9, 12], [12, 16], [17, 22], [22, 23], [0, 0]]);
  assert.deepEqual(e.wordIds, [-1, 0, 1, 2, 2, 2, 3, 4, -1]);
  assert.deepEqual(e.sequenceIds, [-1, 0, 0, 0, 0, 0, 0, 0, -1]);
  assert.deepEqual(e.specialTokensMask, [1, 0, 0, 0, 0, 0, 0, 0, 1]);
  assert.deepEqual(e.typeIds, new Array(9).fill(0));
});

test('accents: NFD offsets stay on the original character, strip when lowercase', () => {
  const e = tok.encode('Café');
  assert.deepEqual(e.ids, [2, 10, 3]);
  assert.deepEqual(e.offsets[1], [0, 4]);
  const cased = JsonTokenizer.fromJson(tinyJson(false));
  assert.deepEqual(cased.encode('Café').ids, [2, 1, 3]); // no strip, no lowercase
});

test('CJK gets its own words, offsets in UTF-16 units', () => {
  const e = tok.encode('a中文b');
  assert.deepEqual(e.ids, [2, 13, 15, 16, 14, 3]);
  assert.deepEqual(e.offsets.slice(1, 5), [[0, 1], [1, 2], [2, 3], [3, 4]]);
  const astral = tok.encode('a 🚀 b');
  assert.deepEqual(astral.ids, [2, 13, 1, 14, 3]);
  assert.deepEqual(astral.offsets.slice(1, 4), [[0, 1], [2, 4], [5, 6]]);
});

test('lowercase of a dotted capital I: accent stripping runs first, so it is one i', () => {
  assert.deepEqual(tok.encode('\u0130').ids, [2, 17, 3]);
  // with stripping off the lowercase gives i and a combining dot on one original
  const keep = tinyJson(true);
  (keep.normalizer as Record<string, unknown>).strip_accents = false;
  const e = JsonTokenizer.fromJson(keep).encode('\u0130');
  assert.deepEqual(e.ids, [2, 17, 18, 3]);
  assert.deepEqual(e.offsets.slice(1, 3), [[0, 1], [0, 1]]);
});

test('words over the limit and words without a match become unk', () => {
  const long = 'a'.repeat(13);
  assert.deepEqual(tok.encode(long).ids, [2, 1, 3]);
  assert.deepEqual(tok.encode('unaffx').ids, [2, 1, 3]);
  assert.deepEqual(tok.encode('unaffx').offsets[1], [0, 6]);
});

test('special tokens in the text keep their id and their span, matching is case sensitive', () => {
  const e = tok.encode('hello [SEP] world [mask]');
  assert.deepEqual(e.ids, [2, 5, 3, 6, 1, 1, 1, 3]);
  assert.deepEqual(e.offsets[2], [6, 11]);
  assert.deepEqual(e.wordIds.slice(1, 4), [0, 1, 2]);
});

test('control characters vanish, tab and newline count as spaces', () => {
  const e = tok.encode('hello\u0000\tworld\u000b!');
  assert.deepEqual(e.ids, [2, 5, 6, 12, 3]);
  assert.deepEqual(e.offsets.slice(1, 4), [[0, 5], [7, 12], [13, 14]]);
});

test('pair: type ids, sequence ids, word ids restart', () => {
  const e = tok.encode('hello', 'world !');
  assert.deepEqual(e.ids, [2, 5, 3, 6, 12, 3]);
  assert.deepEqual(e.typeIds, [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(e.sequenceIds, [-1, 0, -1, 1, 1, -1]);
  assert.deepEqual(e.wordIds, [-1, 0, -1, 0, 1, -1]);
});

test('truncation counts the special tokens of the template', () => {
  const single = tok.encode('hello world hello world', null, { maxLength: 5 });
  assert.deepEqual(single.ids, [2, 5, 6, 5, 3]);
  const pair = tok.encode('hello world hello', 'world world', { maxLength: 7 });
  assert.equal(pair.ids.length, 7);
  assert.throws(() => tok.encode('a', 'hello world hello', { maxLength: 4, truncation: 'only_first' }),
    /too short/);
  assert.equal(tok.encode('hello world', null, { maxLength: 3, truncation: false }).ids.length, 4);
});

test('truncateLengths: longer shrinks first, odd token from the shorter, tie from the first', () => {
  assert.deepEqual(truncateLengths(10, 4, 8, 'longest_first'), [4, 4]);
  assert.deepEqual(truncateLengths(10, 10, 9, 'longest_first'), [4, 5]);
  assert.deepEqual(truncateLengths(12, 10, 9, 'longest_first'), [5, 4]);
  assert.deepEqual(truncateLengths(9, 30, 20, 'longest_first'), [9, 11]);
  assert.deepEqual(truncateLengths(10, 6, 12, 'only_first'), [6, 6]);
  assert.deepEqual(truncateLengths(10, 6, 12, 'only_second'), [10, 2]);
});

test('unsupported components throw with the type name', () => {
  const norm = tinyJson(true);
  (norm as Record<string, unknown>).normalizer = { type: 'Lowercase' };
  assert.throws(() => JsonTokenizer.fromJson(norm), /normalizer type Lowercase/);
  const pre = tinyJson(true, { pre_tokenizer: { type: 'Whitespace' } });
  assert.throws(() => JsonTokenizer.fromJson(pre), /pre_tokenizer type Whitespace/);
  const model = tinyJson(true);
  (model.model as Record<string, unknown>).type = 'WordLevel';
  assert.throws(() => JsonTokenizer.fromJson(model), /model type WordLevel/);
  const post = tinyJson(true, { post_processor: { type: 'BertProcessing' } });
  assert.throws(() => JsonTokenizer.fromJson(post), /post_processor type BertProcessing/);
});

test('added tokens with lstrip take the spaces before them', () => {
  const added = tinyJson(true);
  (added.added_tokens[4] as Record<string, unknown>).lstrip = true;
  const e = JsonTokenizer.fromJson(added).encode('hello   [MASK] world');
  assert.deepEqual(e.ids, [2, 5, 4, 6, 3]);
  assert.deepEqual(e.offsets[2], [5, 14]);
});
