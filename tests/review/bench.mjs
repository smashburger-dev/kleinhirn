// CPU microbenchmarks only; writes JSON when given a local output path.
// node --expose-gc --import ./tests/helpers/node-hooks.mjs tests/review/bench.mjs tests/review/cpu-results.json
import { writeFileSync } from 'node:fs';
import { loadavg, cpus, platform } from 'node:os';
import { summarizeLatency } from '../../bench/metrics.ts';
import { JsonTokenizer } from '../../src/tokenizer/hf/index.ts';
import { Bpe } from '../../src/tokenizer/hf/bpe.ts';
import { Unigram } from '../../src/tokenizer/hf/unigram.ts';
import { BpeTokenizer } from '../../src/tokenizer/bpe.ts';
import { buildUnigram, unigramEncode } from '../../src/tokenizer/unigram.ts';
import { fromText } from '../../src/tokenizer/hf/normalized.ts';
import { parseNormalizer } from '../../src/tokenizer/hf/normalizers.ts';
import { halfBitsToFloat32 } from '../../src/half.ts';
import { Kleinhirn } from '../../src/index.ts';
import { JuliaEngine } from '../../src/julia.ts';
import { model, fixture } from '../helpers/review.mjs';

let sink;
const results = [];
async function measure(name, fn, iterations = 1, samples = 7, warm = 2) {
  for (let i = 0; i < warm; i += 1) sink = await fn();
  const times = [];
  for (let s = 0; s < samples; s += 1) {
    const start = performance.now();
    for (let i = 0; i < iterations; i += 1) sink = await fn();
    times.push((performance.now() - start) / iterations);
  }
  results.push({ name, iterations, samplesMs: times, ...summarizeLatency(times) });
}

const started = { date: new Date().toISOString(), node: process.version, platform: platform(),
  cpu: cpus()[0].model, load: loadavg(), kind: 'CPU-only, synthetic data, not engine latency' };
const bpeOpt = { unkToken: null, continuingSubwordPrefix: null, endOfWordSuffix: null,
  fuseUnk: false, byteFallback: false, ignoreMerges: false };
const bp = new Bpe(new Map([['a', 0], ['aa', 1]]), [['a', 'a']], bpeOpt);
const legacyBpe = new BpeTokenizer({ model: { vocab: { a: 0, aa: 1, '▁': 2 }, merges: [['a', 'a']] }, added_tokens: [] });
for (const n of [512, 2048, 8192]) {
  const chars = Array(n).fill('a'), text = chars.join('');
  await measure(`HF BPE repetitive n=${n}`, () => bp.tokenize(chars), 1, 5);
  await measure(`Julia BPE repetitive n=${n}`, () => legacyBpe.encodeIds(text), 1, 5);
}
for (const max of [32, 256]) {
  const vocab = [['<unk>', -10], ['a', -1], ['z'.repeat(max), -2]];
  const uni = new Unigram(vocab, 0, false), old = buildUnigram(vocab, 0);
  for (const n of [512, 2048]) {
    const chars = Array(n).fill('a'), text = chars.join('');
    await measure(`HF Unigram n=${n} maxPiece=${max}`, () => uni.tokenize(chars), 1, 5);
    await measure(`GLiNER trie n=${n} maxPiece=${max}`, () => unigramEncode(old, text), 10, 5);
  }
}
const normalize = parseNormalizer({ type: 'NFC' });
for (const n of [256, 1024, 4096]) {
  // High class marks followed by low class marks force insertion-sort shifts.
  const marks = fromText('x' + '\u0345'.repeat(n / 2) + '\u0300'.repeat(n / 2));
  await measure(`NFC offset reorder marks=${n}`, () => normalize(marks), 1, 5);
}
const tok = JsonTokenizer.fromJson(fixture('bpe-roberta'));
for (const n of [1024, 8192, 65536]) {
  const text = 'hello '.repeat(Math.ceil(n / 6)).slice(0, n);
  await measure(`encode maxLength128 inputChars=${n}`, () => tok.encode(text, null, { maxLength: 128 }), 1, 5);
}

const singleWord = fixture('bpe-nfc-bytelevel-post');
singleWord.added_tokens = [{ id: 900, content: 'X', normalized: false, special: false, single_word: true }];
const boundaryTok = JsonTokenizer.fromJson(singleWord);
for (const n of [128, 512, 2048]) {
  const text = 'X '.repeat(n);
  await measure(`AddedToken single_word matches=${n}`, () => boundaryTok.encode(text), 1, 5);
}

// No actual 34 MB tokenizer is shipped in this worktree. Generate a sparse-merge
// 256k vocabulary with the same JSON size, not a substitute for Gemma's model.
const large = fixture('bpe-roberta');
large.model.merges = [];
for (let i = 0; i < 256000; i += 1) large.model.vocab[`token${i.toString(36).padStart(6, '0')}_${'x'.repeat(107)}`] = 300 + i;
const target = 34_000_000;
const initial = JSON.stringify(large);
large.reviewPadding = 'x'.repeat(Math.max(0, target - Buffer.byteLength(initial) - 19));
const json = JSON.stringify(large);
const jsonBytes = Buffer.byteLength(json);
globalThis.gc?.();
const heapBefore = process.memoryUsage().heapUsed;
let loaded;
await measure('34MB synthetic JSON.parse', () => JSON.parse(json), 1, 3, 1);
await measure('34MB synthetic JsonTokenizer.fromString', () => { loaded = JsonTokenizer.fromString(json); return loaded; }, 1, 3, 1);
globalThis.gc?.();
const retainedHeapDelta = process.memoryUsage().heapUsed - heapBefore;

for (const precision of ['f32', 'f16']) {
  const { engine } = model('sentence-transformers/all-MiniLM-L6-v2', precision);
  const emb = engine.weights.embeddings;
  const gliner = Reflect.construct(Kleinhirn, [{}, { embeddings: emb }, null, { hiddenSize: 384 }, 1, precision]);
  const julia = Reflect.construct(JuliaEngine, [{}, { embeddings: emb }, null, { hiddenSize: 384 }, precision]);
  for (const L of [128, 1024]) {
    const input = { inputIds: new Int32Array(L), seqLen: L };
    const inputs = Array(16).fill(input);
    await measure(`Encoder wordRows ${precision} B1 L${L}`, () => engine.wordRows([input], L), 50);
    await measure(`Encoder wordRows ${precision} B16 L${L}`, () => engine.wordRows(inputs, L), 5);
    await measure(`GLiNER embeddingRows ${precision} L${L}`, () => gliner.embeddingRows(input, { length: L }), 50);
    await measure(`Julia embeddingRows ${precision} L${L}`, () => julia.embeddingRows(input, { length: L }), 50);
    await measure(`Encoder mask/types B16 L${L}`, () => engine.maskAndTypes(inputs, L), 100);
  }
}
for (const n of [16, 384, 8192 * 28]) {
  const bits = new Uint16Array(n).fill(0x3c00);
  await measure(`half native n=${n}`, () => halfBitsToFloat32(bits, true), 20);
  await measure(`half manual n=${n}`, () => halfBitsToFloat32(bits, false), 20);
}
const { engine } = model();
await measure('queue enqueue + async noop', () => engine.enqueue(async () => 1), 10000);
const report = { ...started, endLoad: loadavg(), large: { jsonBytes, vocabSize: Object.keys(large.model.vocab).length,
  retainedHeapDelta, caveat: 'includes one retained tokenizer and benchmark objects; not total peak memory' }, results };
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, results: results.map(({ name, medianMs, p95Ms }) => ({ name, medianMs, p95Ms })) }, null, 2));
void sink;
