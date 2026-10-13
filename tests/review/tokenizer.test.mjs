// RED, source-contract cases absent from the stored HF 0.22.2 fixtures.
// No Python or downloaded reference is used. See report for reference limitations.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseNormalizer } from '../../src/tokenizer/hf/normalizers.ts';
import { fromText, join } from '../../src/tokenizer/hf/normalized.ts';

// R6 stage 3: \b and \B translate to lookarounds on Oniguruma's Unicode \w (src/tokenizer/hf/normalizers.ts).
test('T01 Oniguruma word boundary must not replace inside a Unicode word', () => {
  const normalize = parseNormalizer({ type: 'Replace', pattern: { Regex: '\\bfoo\\b' }, content: 'X' });
  // Oniguruma \\w includes é. Consequently neither boundary exists in éfooé.
  assert.equal(join(normalize(fromText('éfooé'))), 'éfooé');
});
