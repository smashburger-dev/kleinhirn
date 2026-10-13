// R6 second entry, dist/tokenizer.js: the HF tokenizers with the synchronous
// JsonTokenizer.fromString and the async JsonTokenizer.fromBytes. The core loads
// this chunk on demand (encoder.ts fetchTokenizer), so a model without a
// tokenizer never downloads it.

export { JsonTokenizer } from './tokenizer/hf/index.ts';
export type { Encoding, EncodeOptions, TruncationStrategy } from './tokenizer/hf/index.ts';
