# Research: encoder classifiers on WebGPU

Status 2026-09-26. Every claim carries a source; own calculations are
marked as such.

## Browser state

| Browser | WebGPU since | Source |
|---|---|---|
| Chrome, Edge (Windows, macOS, ChromeOS) | 113 | https://web.dev/blog/webgpu-supported-major-browsers |
| Firefox Windows | 141 | same source |
| Firefox macOS Tahoe 26, ARM64 | 145 | same source |
| Safari macOS/iOS/iPadOS 26 | 26 | same source |

- `shader-f16`: 93.4 % of browsers tested by Web3D Survey, Safari 100 %
  (https://web3dsurvey.com/webgpu/features/shader-f16). An f32 path
  remains mandatory as a fallback.
- Subgroups: Chrome and Edge from 144 (January 2026), Firefox and Safari
  not (https://webstatus.dev/features/webgpu-subgroups). Subgroup
  kernels are therefore only an optional speedup.

## Existing runtimes and what they measure

| Runtime | What it does | Evidence |
|---|---|---|
| ONNX Runtime Web, WebGPU EP | general ONNX graph on WebGPU, import `onnxruntime-web/webgpu` | https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/web/ep-webgpu.md |
| Transformers.js v3 | 120 architectures, WebGPU via ORT Web, quantizations down to q4f16 | https://huggingface.co/blog/transformersjs-v3 |
| GLiNER small v2.1 via Transformers.js | 233.9 MB download on WebGPU (q4f16), 174.9 MB on WASM (uint8) | https://localmodel.run/browser/gliner-small-v2.1 |
| GLiNER.js | TypeScript inference for GLiNER on onnxruntime-web, WebGPU selectable | npm package `gliner` |
| gliner2.5-multi ONNX for WebGPU (Pastel-Org) | one encoder pass in ORT Web; `model.onnx` 1,121.5 MB; on M2 Max, base and multi died between 3,500 and 3,600 words at `createCommandEncoder`/`bad_alloc`; 4,096 words in 13 windows: base 2.75 s, multi 4.3 s | https://huggingface.co/nicolasembleton/gliner2.5-multi-v1-onnx |
| WebGPT | GPT-2 in pure JS/WGSL; M1: 30 ms per token at 117M parameters (f32) | https://github.com/0hq/WebGPT |
| vibe-infer | MNIST MLP from WGSL compute shaders without a framework, teaching example | https://github.com/vtemian/vibe-infer |

For GLiNER2.5-small on WebGPU there is no published latency number; the
own baseline measurement (`bench/run-ort.mjs`) fills that gap.

## Competitor: Julia 1 (Supersonic Labs)

Sources, retrieved 2026-09-27: https://supersoniclabs.ia.br/julia-1/,
https://huggingface.co/SupersonicLabs/Julia-1,
https://huggingface.co/SupersonicLabs/Julia-1-ONNX, encoder config
https://huggingface.co/jhu-clsp/mmBERT-small.

- Model: 144.3M parameters, fine-tuned from mmBERT-small (ModernBERT: 22
  layers, hidden 384, 6 heads, GeGLU with intermediate 1152, vocabulary
  256,000, RoPE with theta 160,000, local attention with window 128,
  every third layer global, no biases, tied embeddings). The embedding
  table carries about 98M of the parameters. License Apache-2.0.
- Interface: state, question and 2 to 20 options; output a choice
  (choice), a grade (score) or yes/no (noul). This matches GLiNER2
  classification with labels in the input.
- Their numbers (2026-09-24, H200 BF16): Typed Decisions 1,463/2,000
  (73.15 %), AG News 94/100, DAIR Emotion 86/100, Banking77 with
  72-label preselection 64/100, MASSIVE scenarios 71.50 % over 52
  languages. Protocol: Jev benchmark repo, pinned to commit 0d610cc53e79.
- Runtimes: PyTorch CPU on Apple M4 33.15 ms median per decision (100
  words, 4 options), 370.6 MiB RAM; Android with ONNX Runtime on CPU
  203 ms, 393 MB. Browser: ONNX Runtime WebGPU in Brave under Linux, GPU
  unnamed, 75.47 ms per decision in batches of four, weights 551 MB in
  fp32, loading from browser cache 5.84 s, 100/100 agreement with the
  original, maximum logit deviation 0.00225.
- Calculation: per token and layer about 1.9M multiply-adds, over 22
  layers 42M. That is about double GLiNER2.5-small (21M).

Attack surface: their browser path is the generic ORT WebGPU runtime
with fp32 weights. Their 100 test requests with original logits and
`benchmark-webgpu.html` are public, so their path can be measured side
by side with kleinhirn on the same machine.

## Why an own engine

Calculation for GLiNER2.5-small, L128, one head with up to 16 labels:

- Compute: 12 layers of 128 x 384 x (3 x 384 + 384 + 2 x 1536) = 226M
  multiply-adds each, 2.7G total; attention with the two relative terms
  is small on top. About 5.4 GFLOP per call. A 16-core Apple GPU
  delivers about 5.2 TFLOP/s; at full utilization that would be 1.0 ms,
  at a realistic 20 to 40 % utilization 2.6 to 5.2 ms. Shorter buckets
  (L64 for short answers) halve it.
- The rest of the latency is overhead: number of dispatches, bind-group
  creation, buffer copies, the readback path (`mapAsync`), tokenization
  in JS.
- A general runtime pays one dispatch per operator and has to assemble
  the relative DeBERTa attention from many small gather operators. A
  special engine fuses: QKV in one matmul, the relative attention in one
  kernel, bind groups once per bucket at load.
- Constant parts of the DeBERTa attention do not depend on the input.
  `rel_embeddings` after LayerNorm and from it `key_proj`/`query_proj`
  per layer can be precomputed during conversion.
- Bundle: ORT Web ships several MB of WASM even on the WebGPU path. A
  special engine needs JS and WGSL in the two-digit KB range (measured
  in `FINDINGS.md`).

## Memory calculation

| Variant | Parameters | Download (calculation) |
|---|---|---|
| small, all fp32 | 70.9M | 284 MB |
| small, layers fp16, embeddings fp16 | 70.9M | 142 MB |
| small, layers fp16, embeddings int8 | 70.9M | 92 MB |
| small, layers int8, embeddings int8 | 70.9M | 71 MB |
| small, vocabulary trimmed to 30k, int8 | about 33M | about 33 MB |
| multi, layers fp16, embeddings int8 | about 278M | about 364 MB |
| multi, vocabulary trimmed to 40k, fp16 + int8 | about 117M | about 202 MB |

The embedding table needs no GPU memory: at most 256 rows are read per
call. kleinhirn keeps it in JS memory and uploads only the L x hidden
values. Note: `maxStorageBufferBindingSize` is 128 MiB without requested
limits; the small table in fp32 (197 MB) does not fit a standard
binding.

Shards are cut under 95 MB so weight hosting stays simple (GitHub Pages
allows 100 MB per file).

## Vocabulary trimming

A multilingual model carries embeddings for languages a deployment never
sees. Keeping only the tokens that occur in the target corpus, plus all
single characters, shrinks the table strongly at small quality loss.
Evidence: Ushio, Zhou, Camacho-Collados 2023, "Efficient Multilingual
Language Model Compression through Vocabulary Trimming" (Findings of
EMNLP 2023).

Implemented as `convert/trim_vocab.py` on multi (measurements in
`docs/FINDINGS.md`): a corpus (file with texts) determines the kept ids,
plus all single characters, all byte pieces `<0xNN>` and all
special/schema tokens; the tool writes a trimmed HF checkpoint
(tokenizer.json, model.safetensors with gathered embedding rows,
vocab_size in config.json), then `export_weights.py` exports to the
kleinhirn format. Two pitfalls: the used ids must be collected through
the real schema path, because texts in schema context form pieces
without the `▁` prefix that a standalone encode never shows; and the
label strings belong in the corpus, otherwise the markers segment
differently and argmax parity breaks.

## Risks

- Parity of the relative attention: DeBERTa uses logarithmic buckets
  from |i - j| >= 129. That never occurs at L128, it does at L256.
  Goldens must cover both lengths.
- Masked rows: the HF implementation sets fully masked rows to 0 after
  the softmax; other wrappers use -1e4 and get a uniform distribution.
  Only valid positions are compared.
- fp16 overflow in attention scores and LayerNorm variance: accumulation
  in f32, storage in f16.
- Tokenizer: DeBERTa-v3 uses SentencePiece unigram with precompiled
  normalization. A JS reimplementation needs parity tests on German
  text, code and formulas.
- `mapAsync` latency on readback: read only the K probabilities, reuse
  the staging buffer.
- Memory measurement on Apple Silicon: GPU allocations live in the GPU
  process and do not necessarily appear in RSS. The memory metric is
  therefore calibrated beforehand with a known 512-MB buffer.
