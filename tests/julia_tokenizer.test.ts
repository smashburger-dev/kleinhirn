import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { BpeTokenizer } from '../src/tokenizer/bpe.ts';

function loadJson(path: string) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

const tokPath = 'models/julia-1/f32/tokenizer.json';
// models/ carries the (large, non-git) weights and tokenizer; a
// checkout without it skips the golden checks instead of failing.
const skip = existsSync(tokPath) ? false
  : `${tokPath} not present (models/ is not shipped)`;
const tokenizer = skip ? null : new BpeTokenizer(loadJson(tokPath));

test('julia-1: tokenizer_ids golden', { skip }, () => {
  const golden = loadJson('tests/golden/julia-1/tokenizer_ids.json');
  let checked = 0;
  for (const [text, ids] of Object.entries(golden)) {
    assert.deepEqual(tokenizer!.encodeIds(text), ids, `text ${JSON.stringify(text)}`);
    checked += 1;
  }
  assert.equal(checked, Object.keys(golden).length);
});
