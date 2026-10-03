// K28.3b: ByteLevel BPE, Unigram and BPE with Metaspace, NFC, Strip, Replace, RobertaProcessing,
// added tokens with lstrip and normalized, against expectations computed by HF tokenizers 0.22.2
// on tiny tokenizers (convert/k28_tokenizer_fixtures.py). Runs without models/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JsonTokenizer } from '../src/tokenizer/hf/index.ts';
import type { TruncationStrategy } from '../src/tokenizer/hf/index.ts';
import { oniguruma } from '../src/tokenizer/hf/normalizers.ts';

const DIR = new URL('./fixtures/hf-tokenizers/', import.meta.url);
const NAMES = ['bpe-roberta', 'bpe-roberta-prefix', 'bpe-nfc-bytelevel-post', 'unigram-metaspace', 'bpe-metaspace'];

interface Case {
  first: string; second: string | null; max_length: number; truncation: string; ok: boolean; error?: string;
  ids: number[]; type_ids: number[]; offsets: number[][]; word_ids: number[]; sequence_ids: number[];
}

// Python counts code points, JS UTF-16 units
function toUtf16(text: string, cp: number): number {
  let units = 0;
  let i = 0;
  for (const c of text) {
    if (i === cp) break;
    units += c.length;
    i += 1;
  }
  return units;
}

for (const name of NAMES) {
  test(`${name}: every fixture case equals HF tokenizers`, () => {
    const tok = JsonTokenizer.fromString(readFileSync(new URL(`${name}.json`, DIR), 'utf8'));
    const cases: Case[] = JSON.parse(readFileSync(new URL(`${name}.expected.json`, DIR), 'utf8'));
    for (const c of cases) {
      const label = JSON.stringify([c.first, c.second, c.max_length, c.truncation]);
      if (!c.ok) {
        assert.throws(() => tok.encode(c.first, c.second, {
          maxLength: c.max_length, truncation: c.truncation as TruncationStrategy }),
        (e: Error) => (c.error as string).includes(e.message), label);
        continue;
      }
      const e = tok.encode(c.first, c.second, {
        maxLength: c.max_length, truncation: c.truncation as TruncationStrategy });
      const offsets = c.offsets.map(([s, t], k) => {
        if (c.sequence_ids[k] < 0) return [s, t];
        const text = c.sequence_ids[k] === 0 ? c.first : (c.second as string);
        return [toUtf16(text, s), toUtf16(text, t)];
      });
      assert.deepEqual(e.ids, c.ids, label);
      assert.deepEqual(e.typeIds, c.type_ids, label);
      assert.deepEqual(e.offsets, offsets, label);
      assert.deepEqual(e.wordIds, c.word_ids, label);
      assert.deepEqual(e.sequenceIds, c.sequence_ids, label);
    }
  });
}

test('unsupported components throw with the type name', () => {
  const base = JSON.parse(readFileSync(new URL('bpe-roberta.json', DIR), 'utf8'));
  const with_ = (patch: Record<string, unknown>) => JsonTokenizer.fromJson({ ...base, ...patch });
  assert.throws(() => with_({ normalizer: { type: 'Lowercase' } }), /normalizer type Lowercase/);
  assert.throws(() => with_({ pre_tokenizer: { type: 'Whitespace' } }), /pre_tokenizer type Whitespace/);
  assert.throws(() => with_({ model: { type: 'WordLevel' } }), /model type WordLevel/);
  assert.throws(() => with_({ post_processor: { type: 'BertProcessing' } }), /post_processor type BertProcessing/);
  assert.throws(() => with_({ model: { ...base.model, dropout: 0.1 } }), /dropout/);
});

test('oniguruma: word and space classes follow Oniguruma, not JavaScript', () => {
  assert.equal(new RegExp(oniguruma('(\\W)?@usuario(\\W)'), 'u').test('a@usuarió'), false);
  assert.equal(new RegExp(oniguruma('(\\W)url(\\W)'), 'u').test(' url　'), true);
  assert.equal(new RegExp(oniguruma('x\\s'), 'u').test('x\u0085'), true);
  assert.throws(() => oniguruma('[\\W]'), /character class/);
});
