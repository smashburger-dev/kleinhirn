import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PairTable } from '../src/tokenizer/pairs.ts';

test('PairTable keeps exact keys, replaces a repeated pair and grows past its first size', () => {
  const t = new PairTable(1);
  assert.equal(t.get(1, 2), -1);
  t.set(1, 2, 7);
  t.set(2, 1, 8);
  t.set(1, 2, 9); // a later set replaces the value
  assert.equal(t.get(1, 2), 9);
  assert.equal(t.get(2, 1), 8);
  assert.equal(t.get(0, 0), -1);
  for (let i = 0; i < 5000; i += 1) t.set(i, i * 3 + 1, i);
  for (let i = 0; i < 5000; i += 1) assert.equal(t.get(i, i * 3 + 1), i);
  assert.equal(t.get(9, 9), -1);
});
