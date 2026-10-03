# kleinhirn

Encoder models in the browser, on the GPU, with no server.

kleinhirn runs text encoders in the user's browser: classification,
zero-shot classification over NLI, sentence embeddings, named entities
and reranking. The text stays on the device, there is no inference
server to pay for, and it works offline once the model is cached.

It loads [Hugging Face](https://huggingface.co/) checkpoints of six
encoder families directly from `config.json` and
[safetensors](https://github.com/safetensors/safetensors):
[BERT](https://github.com/google-research/bert) (with
[ELECTRA](https://github.com/google-research/electra) and
[MiniLM](https://github.com/microsoft/unilm/tree/master/minilm)),
[RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta),
[XLM-R](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr),
[DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased),
[DeBERTa-v2/v3](https://github.com/microsoft/DeBERTa) and
[ModernBERT](https://github.com/AnswerDotAI/ModernBERT). Two models with
their own heads run on the same engine:
[GLiNER2.5](https://github.com/fastino-ai/GLiNER2), a zero-shot
classifier that takes any set of labels, and
[Julia 1](https://huggingface.co/SupersonicLabs/Julia-1), a decision
model that picks one of up to 20 options. The engine ships its own GPU
programs ([WGSL](https://www.w3.org/TR/WGSL/) kernels) for
[WebGPU](https://www.w3.org/TR/webgpu/); browsers without WebGPU fall
back to a hand-written WASM-SIMD module on the CPU (today for the
DeBERTa models only).

The usual tool for running such models in a browser is
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (ORT) from
Microsoft. It is the baseline for every speed number below.

## Results

- **Same answers as the original.** 88 of the 89 checkpoints on our list
  (the three most downloaded per family and task, up to 200M parameters)
  make the same decisions as the [PyTorch](https://github.com/pytorch/pytorch)
  reference: 100 % in full
  precision (f32), at least 99.5 % in half precision (f16) of the
  decisions that f16 rounding cannot flip, embeddings with cosine at
  least 0.9999 and 0.999. The 89th passes in f32 and the
  engine picks f32 for it.
- **Faster than ORT at its best, in every family.** One model per family
  (the most downloaded), inputs of exactly 128 and 512 tokens, f16,
  against ORT's fastest setting we found for each model (graph optimizer
  variants, three ORT builds, graph capture where it starts):

  | Model | 128 tokens kleinhirn / ORT | 512 tokens kleinhirn / ORT |
  |---|---|---|
  | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (BERT) | 3.29 / 5.29 ms | 8.97 / 12.06 ms |
  | [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) (RoBERTa) | 13.40 / 18.96 ms | 47.55 / 59.67 ms |
  | [mmarco-mMiniLMv2-L12](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (XLM-R) | 6.07 / 10.09 ms | 17.86 / 24.33 ms |
  | [distilbert-base-uncased-sst-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (DistilBERT) | 6.99 / 9.76 ms | 24.54 / 33.34 ms |
  | [deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (DeBERTa-v3) | 14.54 / 24.35 ms | 57.31 / 95.53 ms |
  | [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) (ModernBERT) | 6.99 / 12.44 ms | 21.35 / 35.81 ms |

  1.26x to 1.78x faster; all three kleinhirn repetitions below all three
  ORT repetitions in each of the twelve cells.
- **Faster again on real texts.** kleinhirn computes only the rows a text
  has; ORT's fixed-shape graphs compute the whole bucket. On the
  1,000-text corpus (median 45 of 128 tokens) GLiNER2.5-small runs 4.4
  ms against 12.2 ms for ORT with graph capture in f16 (2.8x) and 5.4
  against 14.8 ms in f32 (2.7x). Julia 1 decides in 10.4 ms against
  25.5 ms for its upstream ORT pipeline in f16 (2.4x) and 14.0 against
  29.2 ms in f32 (2.1x).
- **Closer to the reference than ORT.** On the measurement inputs above,
  ORT's f16 output deviates 2.9 to 20.5 times more from ORT's own f32
  output than kleinhirn's f16 does. On Julia 1 the largest logit error
  in f16 is 0.109 for kleinhirn and 1.02 for ORT. Every kernel
  accumulates in f32.
- **Less memory.** 52 to 71 % of ORT's peak memory, depending on the
  path (941 vs 1,442 MiB for GLiNER2.5-small in f16, 1,138 vs 1,599 MiB
  for Julia 1). The ORT peaks are from September; kleinhirn's grew by 5
  to 14 % with the new kernels.
- **Small.** The engine is 46 KB gzip, plus 10 KB for the WASM
  fallback. ORT's WebGPU runtime files are 6.3 MB gzip (25.9 MB raw).
  The model weights come on top in both cases.
- **Not done yet.** The WASM fallback covers only the DeBERTa models.
  Phones have not been measured since the kernels changed (October
  2026); the last iPhone run missed the latency target (below).
- **Reproducible.** Every number comes from a script in this repo under
  a fixed protocol: interleaved runs, a quiet machine, the median of
  three repetitions. Details under "How we measure" and in
  `docs/FINDINGS.md`, sections 14 and 15.

## Use cases

- Learning platforms that give feedback on student answers on the
  device. The first user is [argmin](https://github.com/smashburger-dev/argmin),
  a German learning platform that
  classifies misconceptions in learner answers.
- Local-first and offline apps: notes, mail or task tools that tag, sort
  or route text where it was written.
- Semantic search and deduplication with sentence embeddings, computed
  where the text is.
- Browser extensions that classify the current page or the text being
  typed, with no backend.
- Support forms that suggest the right category or queue while the user
  types.
- Moderation by your own rules: zero-shot labels checked in the browser
  before anything is posted.
- Agents and interfaces that need a fast local choice among fixed
  options, which is what Julia 1 is trained for: which tool, which
  step, which intent.

## Status

- Models: the 89 checkpoints of `data/k28/models.json` (six families,
  five tasks: sequence classification, NLI, embeddings, token
  classification, reranking), GLiNER2.5 small, base and multi
  (classification) and Julia 1. GLiNER2.5's entity extraction is not
  implemented yet.
- The speed numbers above are from an M1 Pro in
  [Chromium](https://www.chromium.org/). The new kernels give the same
  decisions in [Playwright](https://github.com/microsoft/playwright)
  [WebKit](https://github.com/WebKit/WebKit) and
  [Firefox](https://github.com/mozilla-firefox/firefox).
- Device matrix from September 2026 (before the kernel rewrite): parity
  with the PyTorch reference holds in every stage that runs in Chromium,
  [Brave](https://github.com/brave/brave-browser), Playwright WebKit,
  [Safari](https://www.apple.com/safari/) 27, Playwright Firefox 153 and
  Chromium without WebGPU, and on an
  [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/)
  ([FINDINGS section 13](docs/FINDINGS.md)). Latency for the 256-token
  bucket missed the targets (p95 of 20 ms on desktop WebGPU, 300 ms for
  WASM) in every row then; Firefox pays about 104 ms for every
  asynchronous GPU readback.
- On an iPhone 16 Pro (iOS 27, Safari and Brave, which is WebKit there)
  f16 ran with full parity, but the 256-token p95 was 166 ms against
  the 60 ms target; the WASM stage also runs there with exact parity
  (L256 p95 775 ms). Peak memory and the tab kill threshold are not
  measured yet.

## Measure your device

The repository holds a static page, `site/`, that runs the engine on the
device that opens it. It loads the weights from Hugging Face, checks
parity against the goldens, measures latency for two length buckets on
every stage the device supports (f16, f32, WASM), and shows one JSON
file you can download. Page address:
https://smashburger-dev.github.io/kleinhirn/

- Nothing is uploaded. The page requests only itself and the weights
  from Hugging Face. The JSON stays on your device until you share it.
- The file holds the user agent, core count, memory class, the WebGPU
  adapter info, features and limits, and what you type into the device
  field. Adapter info and features can roughly identify a device model.
  Read the file before you share it.
- The page keeps a small marker (first visit, last visit, visit count)
  in browser storage to measure how long storage survives on the
  device. The marker holds no other data.
- To contribute, open an issue with the "Device result" form and attach
  the file. A maintainer runs `tools/check_device_result.mjs`, which
  recomputes parity from the logits in the file, and adds the result to
  [`docs/COMMUNITY.md`](docs/COMMUNITY.md).
- Community numbers are separate from the official matrix. Parity is
  verified by script; latency is self-reported and unverified. One tab
  on one device is one sample. The file format is in
  [`docs/device-results.md`](docs/device-results.md).

## Why it is faster

- The encoder matmuls are register-blocked: every GPU thread computes a
  4x4 block of the output from tiles in workgroup memory, and each output
  still sums in a fixed order in f32. Three tile sizes give the same
  bits; each call picks the one that keeps the GPU busy for its row
  count.
- Attention runs as three matmul-shaped kernels (scores, softmax,
  context). DeBERTa's relative terms (content-to-position and
  position-to-content) are matmuls over the relative positions the
  input can reach. Tiles that no valid token reaches are skipped.
- Only the rows a text has are computed; buckets only fix buffer sizes.
- Each sequence bucket gets a fixed plan built at load time: pipelines,
  buffers and bind groups exist before the first call. A call uploads
  the inputs, encodes one command buffer and reads back once.
- The engine requests exactly the WebGPU minimum device limits (256
  invocations per workgroup, 16 KiB workgroup memory, 128 MiB per
  storage binding) and uses no subgroups, so a desktop GPU cannot use
  more than a phone GPU guarantees. ORT asks for more (1,024
  invocations, 32 KiB) and accumulates its f16 matmuls in f16.

## How ORT was set up

For the six families: one ONNX export per model and length with fixed
shape [1, L] and the head inside the graph, then
`onnxruntime.transformers.optimizer` with its default fusions and two
variants (without EmbedLayerNormalization, with MultiHeadAttention), the
optimizer's fp16 conversion, three ORT Web 1.29.0 builds (default,
`webgpu`, `jspi`), graph capture on and off. The fastest combination per
model and length that passes the parity check on the CPU is the
baseline (`data/k28/k28.8-best.json`): the plain export with graph
capture in six cells, the optimized graph with capture in four, and for
DeBERTa the optimized graph without capture (capture does not start
there), always in one of the native WebGPU builds.

For GLiNER2.5 and Julia 1 the plain export goes through the same
optimizer (BERT path). For WebGPU, `convert/capture_surgery.py` replaces
a Squeeze/Unsqueeze pair in GLiNER's attention-mask chain with one
Reshape, so ORT's graph capture can run. ORT also gets its cheapest
timing boundary: inputs in persistent GPU buffers, one readback per
call.

## Roadmap

- WASM for every family, so a model that runs on one path runs on all.
- Phones: the iPhone with the new kernels, peak memory, how the weights
  are loaded.
- Smaller downloads: the word-embedding table is 31 to 82 % of a
  model's weights, but a text needs only a few of its rows; fetch rows
  on demand, and int8 weights.
- Kernel choice per device by the best worst case.
- A public benchmark for encoders in the browser that compares engines
  with a parity column.

## Supported models

- The 89 checkpoints of `data/k28/models.json`, each at a pinned
  revision: per family and task the three most downloaded Hugging Face
  models up to 200M parameters. Convert one with
  `node tools/k28_convert.ts <model-id>` after fetching it
  (`convert/k28_fetch.py`).
- GLiNER2.5 classification models (DeBERTa-v2/v3 encoders):
  `small-upstream`
  ([fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1)),
  `base-upstream`
  ([fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1)),
  `multi-upstream`
  ([fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1)),
  each at a pinned revision (`tools/fetch_models.py`).
- Julia 1 ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)),
  a ModernBERT decision model fine-tuned from
  [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small):
  state + question + 2-20 options in, one decision out.

## Bundle size

The engine bundle (`npm run build`) is `dist/kleinhirn.js` plus the
WASM worker with the embedded `.wasm` module:

| File | Raw | gzip -9 |
|---|---|---|
| kleinhirn.js | 179,317 B | 46,253 B |
| wasm-worker.js incl. deberta.wasm | 35,771 B | 10,000 B |
| total | 215,088 B | 56,253 B |

`kleinhirn.js` embeds a build id (`Date.now()` in base 36), so its gzip
size can differ by a byte between builds. It grew from 20,948 B with the
general engine (the tokenizers for every family alone: 11.5 KB) and the
new kernels (5.5 KB).

For comparison, onnxruntime-web 1.29.0 loads three runtime files on its
WebGPU path (`ort.webgpu.bundle.min.mjs`,
`ort-wasm-simd-threaded.asyncify.mjs` and `.wasm`): 25,917,382 B raw,
6,338,621 B gzip -9. The model weights are a separate download in both
cases.

## Runtime path

Fallback chain (`backend: 'auto'`):

1. WebGPU f16 (when the adapter supports `shader-f16`)
2. WebGPU f32
3. WASM-SIMD in a worker
   ([AssemblyScript](https://github.com/AssemblyScript/assemblyscript),
   no [Emscripten](https://github.com/emscripten-core/emscripten)),
   DeBERTa models only

The engine requests the device with exactly the WebGPU minimum limits
(`limits: 'minimum'`, the default): 256 invocations per workgroup,
16 KiB workgroup memory, 128 MiB per storage binding, etc. All kernels
stay inside these limits, so a desktop adapter cannot silently use more
than a mobile adapter provides.

Sequence buckets: the general engine loads 128 and 512 by default (any
multiple of 4 up to the model's positions); GLiNER uses L64-L1024 with up
to 16 labels per call and a wide L1280/K80 bucket for label sets up to
80 (see `docs/ARCHITECTURE.md`).

## Quickstart

```bash
npm ci
npm run build                       # dist/kleinhirn.js + wasm worker

# Python env for conversion + goldens (uv or venv, Python 3.12)
uv venv .venv && uv pip install --python .venv/bin/python -r convert/requirements.txt

# Julia-1 scripts (fetch_datasets, julia_golden, export_julia) need
# transformers 5.x in a second env
uv venv .venv-julia && uv pip install --python .venv-julia/bin/python -r convert/requirements-julia.txt

# Pinned model checkpoints -> models/ (gitignored)
.venv/bin/python tools/fetch_models.py small-upstream
.venv/bin/python convert/export_weights.py small-upstream   # -> models/small-upstream/{f32,f16}
# Julia-1: fetch_models.py julia-1, then .venv-julia/bin/python convert/export_julia.py

# Pinned benchmark inputs -> models/k9-data, models/jev-benchmarks
.venv-julia/bin/python tools/fetch_datasets.py

node --test tests/*.test.ts         # tokenizer + limits tests
```

In the browser (the package is not on npm; serve `dist/` or import the
built file directly):

Any model of the list (Python env with `convert/requirements-k28.txt`):

```bash
.venv-k28/bin/python convert/k28_fetch.py sentence-transformers/all-MiniLM-L6-v2
node tools/k28_convert.ts sentence-transformers/all-MiniLM-L6-v2   # -> models/k28/<slug>/{f32,f16}
```

```ts
import { EncoderModel } from './dist/kleinhirn.js';

const model = await EncoderModel.load({
  manifestUrl: '/models/k28/sentence-transformers__all-MiniLM-L6-v2/f16/manifest.json',
});
const [vector] = await model.embed(['The rocket launched successfully.']);
// also: classify(text), zeroShot(text, labels), rerank(query, passages), tokenClassify(text)
```

GLiNER2.5:

```ts
import { loadEngine } from './dist/kleinhirn.js';

const kh = await loadEngine({
  manifestUrl: '/models/small-upstream/f16/manifest.json',
  buckets: [128, 256, 512, 1024],
});
const out = await kh.classify(text, [
  { task: 'topic', labels: ['science', 'sports', 'politics'] },
]);
```

## How we measure

- Reference is PyTorch fp32 on CPU. Goldens are regenerated from the
  public corpus `tests/corpus/texts1000.json` (1,000 texts: 500 from the
  argmin content tree under CC-BY-4.0, 500 generated by
  `tools/synthetic_texts.py`) via `convert/golden.py`.
- Engine parity runs in a visible Playwright Chromium against those
  goldens: `node bench/run-parity.mjs <model> <f32|f16>`.
- A numpy reference forward (`convert/manifest_forward.py`) reads the
  kleinhirn manifest + shards without PyTorch and must match the goldens
  to 1e-4 with 100 % argmax. This validates the weight format
  independent of any browser.
- Accuracy benchmarks follow `docs/BENCHMARKS.md` (K9 suite): every
  dataset is fetched at a pinned revision by `tools/fetch_datasets.py`;
  per-example prediction files contain only `id`, `gold`, `predicted`,
  `probabilities`.
- Latency and peak memory are only reported from official runs on a
  10-core M1 Pro: interleaved ABAB, median of 3 repetitions with the
  range recorded per run; a run is provisional when the 1-minute load
  at its start reaches the pass's bound (4 for the pass against the
  fused ORT graphs, 6 for the earlier passes). The memory metric is
  calibrated against a known 512-MB GPU buffer.
- The ORT side runs on the best configuration we could produce: the
  plain ONNX export (`convert/export_onnx.py`) after
  `onnxruntime.transformers.optimizer` (`convert/optimize_onnx.py`),
  in f32 and in the optimizer's own fp16 conversion. On WebGPU a
  two-node rewrite of the attention-mask chain
  (`convert/capture_surgery.py`) lets ORT's graph capture
  (`enableGraphCapture`) run, with inputs in persistent GPU buffers and
  one readback per call.

## Numbers

The tables below are the September measurements of GLiNER2.5 and Julia 1
(before the kernel rewrite); current numbers are under Results and in
`docs/FINDINGS.md`, sections 14 and 15.

Engine parity on the 1,000-text corpus (WebGPU, minimum limits;
`bench/run-parity.mjs`):

| Model | Precision | Argmax agreement | Max logit diff | Gates |
|---|---|---|---|---|
| small-upstream | f32 | 100 % (968) | 2.10e-5 | passed |
| small-upstream | f16 | 99.90 % (968) | 2.10e-2 | passed |
| base-upstream | f32 | 100 % (968) | 2.93e-5 | passed |
| base-upstream | f16 | 99.90 % (968) | 2.60e-2 | passed |
| multi-upstream | f32 | 100 % (953) | 1.32e-4 | passed |
| multi-upstream | f16 | 99.90 % (953) | 5.15e-2 | passed |

WASM-SIMD parity (f32, `--backend=wasm`, small-upstream): argmax 100 %,
max logit diff 2.25e-5; all gates passed.

Julia 1 (`bench/run-julia-parity.mjs`): f32 100/100 choices, logit diff
1.0e-4; f16 100/100 choices, max probability diff 2.4e-2, above our
1e-2 probability gate (known limitation, see `docs/FINDINGS.md`).

Tokenizer parity (`node --test tests/`): 100 % equal token ids vs the HF
tokenizer on all 1,000 corpus texts, for all three GLiNER models and
Julia 1.

### Latency and memory (official runs)

Measured on a 10-core M1 Pro (16 GB) in Playwright Chromium
151.0.7922.34 (Apple Metal adapter, WebGPU minimum limits), interleaved
ABAB, median of 3 repetitions; ranges show the repetition min-max.
Runner `bench/run-official.mjs`, results under `bench/results/`, full
detail in `docs/FINDINGS.md`. The WebGPU comparison against ONNX Runtime
Web with graph capture comes from a pass at commit 2b7dacd
(`k20gc-official-*`); the WASM pair, Julia 1 and the memory table
come from a pass at commit ba3a2c7 (`k20-official-*`, `k20-mem-p*`),
both with a start-load bound of 4. The bucket sweep and the batch throughput come from
earlier passes with a bound of 6 (commits ccef9a7 and 8535538,
`official-*`).

What the ORT optimizer does to these graphs (onnxruntime 1.29.0, BERT
path, the only one that applies; `bench/results/k20-optimize-*.json`):
it fuses LayerNorm, SkipLayerNorm and Gelu, but not the attention of
either model. DeBERTa-v2 keeps its relative-position terms (c2p, p2c)
as separate gathers, ModernBERT keeps RoPE and the windowed mask
unfused: 0 Attention and 0 MultiHeadAttention nodes in both. The fused
graphs are still faster, mostly in f16, because they drop the casts
that the plain fp16 conversion inserts. ORT's graph capture does not
start on the fused graph: a Memcpy pair around a CPU-placed Squeeze in
the DeBERTa mask chain blocks it. Replacing that Squeeze and the
following Unsqueeze with one Reshape to the fixed shape
(element-identical, same CPU parity) puts every node on WebGPU, and
capture runs.

kleinhirn vs ONNX Runtime Web, small-upstream, L128 (median ms per
inference call on 968 corpus texts; the ORT page serves L128 only):

| Path | Precision | Median ms (range) | p95 ms | Argmax vs PyTorch |
|---|---|---|---|---|
| kleinhirn | f16 | 6.5 (6.5-6.6) | 11.4 | 99.90 % |
| ORT WebGPU, fused graph + graph capture | f16 | 12.5 (12.5-12.5) | 13.2 | 99.59 % |
| kleinhirn | f32 | 7.4 (7.4-7.5) | 13.2 | 100 % |
| ORT WebGPU, fused graph + graph capture | f32 | 16.0 (15.5-16.1) | 16.6 | 100 % |
| kleinhirn WASM | f32 | 69.2 (69.2-69.8) | 120.6 | 100 % |
| ORT wasm, fused graph | f32 | 180.0 (179.9-180.4) | 180.8 | 100 % |

Each kleinhirn row pairs with the ORT row below it (ABAB). The WASM
pair comes from the ba3a2c7 pass; the wasm EP has no graph capture.
In its own ABAB pair the plain export ran 17.4 ms (f16) and 18.2 ms
(f32) against 14.2 and 17.5 ms for the fused graph; graph capture then
took the fused graph from 14.8 to 12.5 ms (f16) and from 18.4 to
15.7 ms (f32). The same path moves by up to 14 % between passes
(kleinhirn f32 6.5 ms in the ba3a2c7 pass, 7.4 ms in the capture
pass), so a factor holds only within its pair. Both ORT fp16
graphs miss our fp16 probability gate (max probability diff 5.3e-2
plain, 3.5e-2 fused, gate 1e-2); kleinhirn f16 stays at 4.2e-3.

kleinhirn across sequence buckets (earlier pass, commit ccef9a7; no
ORT counterpart beyond L128):

| Bucket | f16 median (range) | f32 median (range) | f16 p95 | f32 p95 |
|---|---|---|---|---|
| L128 | 6.8 (6.8-6.9) | 6.9 (6.9-6.9) | 12.1 | 12.5 |
| L256 | 36.2 (35.3-37.4) | 37.4 (37.4-37.4) | 47.9 | 49.7 |
| L512 | 95.9 (95.4-109.5) | 97.5 (97.3-97.8) | 131.8 | 134.4 |
| L1024 | 321.3 (320.8-321.3) | 321.5 (320.0-327.0) | 414.6 | 417.8 |

kleinhirn WASM at L256 in the same pass: 360.9 ms (358.8-361.2), p95
450.4.

Julia 1 against the upstream WebGPU loop on the fused upstream graph.
100 upstream parity requests; every path picks the same option as
PyTorch in 100/100. `bench/upstream-julia/bench-graph.html` runs the
upstream benchmark loop verbatim with the graph file as a parameter.
kleinhirn reports ms per decision, upstream ms per request, the same
unit:

| Pair (ABAB) | kleinhirn ms | upstream ms | Max logit diff vs PyTorch, kleinhirn / upstream |
|---|---|---|---|
| f32, one request per call, tokenization timed on both sides | 24.3 (24.3-24.4) | 24.8 (24.8-24.8) | 1.0e-4 / 1.6e-4 |
| f16, one request per call, tokenization timed on both sides | 24.1 (23.4-24.1) | 21.5 (21.4-22.1) | 0.091 / 1.02 |
| fastest paths, f16: kleinhirn batch API at B1 (pre-tokenized) vs upstream batch 4 | 20.7 (20.5-21.2) | 15.8 (15.7-15.8) | not reported / 1.02 |

Against the unchanged upstream graph (earlier pass, commit ccef9a7)
kleinhirn f16 ran 24.9 ms per decision, upstream batch 4 23.9 and
batch 1 30.2 ms per request.

Download, GPU buffers and peak process memory. Peak memory is the
footprint of the benchmark's own browser process tree, median of 3
browser launches with the launch range in parentheses; footprint
varies up to ~15 % per launch, so a "uses less" claim stands only
where the ranges do not overlap:

| Path | Download MiB | GPU buffer MiB (L128) | Peak MiB, median (range) |
|---|---|---|---|
| kleinhirn small f16 | 152.3 | 51.9 | 850 (790-856) |
| ORT WebGPU f16, fused | 187.7 | not measured | 1442 (1425-1454) |
| kleinhirn small f32 | 296.6 | 103.7 | 1151 (1140-1152) |
| ORT WebGPU f32, fused | 331.6 | not measured | 2321 (2301-2329) |
| kleinhirn WASM f32 | 296.6 | - | 572 (545-737) |
| ORT wasm f32, fused | 331.6 | - | 1280 (1273-1282) |
| kleinhirn julia f16 | 308.0 | 109.9 | 1001 (1000-1002) |
| upstream julia batch 4 / batch 1, fused f16 | not measured (graph files 275.2) | not measured | 1621 (1617-1629) / 1599 (1597-1603) |
| upstream julia batch 4, fused f32 | not measured (graph files 550.2) | not measured | 2497 (2382-2515) |

With graph capture the ORT WebGPU peaks were 1476 MiB (f16) and 2368
MiB (f32), provisional: three of the six launches started above the
load bound.

A load-sensitivity pass found no latency change under pure CPU load
up to a 1-minute load of ~10 on this machine; mixed build and browser
load did affect earlier runs, so the start-load bound stays.

### Second platform (provisional)

On a [Modal](https://modal.com/) NVIDIA L4 under Linux (Chromium 151, Vulkan; gate:
`adapterInfo` reports vendor `nvidia`, architecture `lovelace`, no
fallback adapter) kleinhirn f32 passed every parity gate:
small-upstream 100 % argmax (968 cases) with max logit diff 2.35e-5,
julia-1 all parity gates green. small-upstream f32 at L128 measured
4.5 ms median per call in a single pass (provisional). That Chromium
build does not expose `shader-f16`, so the engine runs in its f32
fallback by design there. `bench/results/nvidia-*.json`.

Read: against the best ORT configuration we could produce (fused
graph plus graph capture), kleinhirn is 1.9x faster than ONNX Runtime
Web in f16 and 2.2x in f32 on WebGPU, and 2.6x on the WASM path
(small-upstream, L128); it peaks at 45-59 % of ORT's memory. For
Julia 1 the fused fp16 graph makes the upstream loop faster than
kleinhirn: 21.5 vs 24.1 ms at equal boundaries, 15.8 vs 20.7 ms
between the fastest paths. That graph deviates up to 1.02 in logits
from PyTorch fp32, kleinhirn f16 by 0.091. In f32 the two are level
(24.3 vs 24.8 ms). kleinhirn's Julia advantage is peak memory (1001
vs 1599-1621 MiB) and fp16 accuracy, not speed. Graph capture was not
tried for Julia: it needs fixed shapes, and the upstream loop calls
with varying lengths.

The batch API (`classifyBatch` / `decideBatch`, batch sizes 1/4/8/16,
stride-packed at 64-token increments) produces outputs bit-identical
to single calls on the same inputs (batch parity B1-B16, f32 and f16,
both models, mixed lengths). Official throughput (ms per decision,
same protocol):

| Path | B1 | B4 | B8 | B16 |
|---|---|---|---|---|
| kleinhirn small f16 L128 | 6.90 | 6.39 | 6.02 | 8.83 |
| kleinhirn julia f16 | 21.7 | 35.8 | 47.0 | 46.2 |

For small-upstream, batching matches the single-call path at B1 and
helps slightly up to B8; B16 regresses. For julia-1 the batch path is
faster than the single path at B1 (21.7 vs 24.9 ms, though the batch
number times pre-tokenized inputs while the single-call number
includes tokenization) but slower per decision from B4 on: in this
throughput pass kleinhirn B4 ran 35.8 ms against an upstream batch-4
25.6 ms from a different pass. The published Julia comparison is the
fused-graph pass above. The API is kept for the parity guarantee and
for hosts that batch anyway.

### Julia 1 re-measurement (PyTorch CPU fp32, K9 suite)

| Task | Published | Measured here | Status |
|---|---|---|---|
| [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,463/2,000 (their GPU BF16 run); 1,451/2,000 (their CPU run) | 1,451/2,000 | matches their CPU run exactly |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | 94/100 | 94/100 | exact |
| [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion) | 86/100 | 86/100 | exact |
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) | 64/100 | ambiguous: 57-69 by router parameters | open, "top-16" not uniquely reconstructable |
| [MASSIVE](https://github.com/alexa/massive) | 71.50 % | en-US 51.71 % | not reproducible; no public request format derives it |

AG News, DAIR Emotion and Banking77 are read from the
[BTZSC](https://huggingface.co/datasets/btzsc/btzsc) copies, sampled as in
[jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks); MASSIVE
from the [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive)
dataset.

kleinhirn's Julia engine (WebGPU f16, minimum limits) scores the same
1,451/2,000 on Typed Decisions as PyTorch CPU; argmax agreement
1,997/2,000 (`bench/results/k9-typed-kleinhirn-julia-f16.json`).

### GLiNER2.5 zero-shot on the K9 tasks (PyTorch CPU, [Jev](https://github.com/AbdelStark/jev-benchmarks) adapter)

| Task | small | base | multi |
|---|---|---|---|
| typed | 36.0 % | 39.7 % | 35.3 % |
| agnews | 47 % | 82 % | 67 % |
| emotiondair | 33 % | 43 % | 36 % |
| banking77 | 47 % | 73 % | 63 % |

Jev reference values under the pinned protocol are reported, not
measured by us: AG News 91/100, DAIR Emotion 48/100 and Banking77
87/100 come from the pinned `btzsc-pilot-v1` report (see
`docs/BENCHMARKS.md`); Typed Decisions 72.70 % is listed on the
[Julia 1 model card](https://huggingface.co/SupersonicLabs/Julia-1)
("Evaluation", column "Jev reference") as a supplied comparison
reference.

## Repository layout

```
src/       engine: device, weights, tokenizer, WGSL graph, WASM fallback
convert/   checkpoint -> weight format, goldens, numpy reference forward
bench/     measurement pages + Playwright runners + k9 suite + results
tests/     Node tests, corpus, goldens (per model)
tools/     corpus builder, pinned model/dataset downloaders
site/      device measurement page (source)
pages/     prebuilt page served by GitHub Pages, no weights
docs/      ARCHITECTURE, RESEARCH, BENCHMARKS, FINDINGS
models/    checkpoints and weights, never in git
```

## Third-party projects

Every project, model and dataset this README names, with the place its
owners publish it. kleinhirn is not affiliated with any of them.

Models measured or supported:

- [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1),
  [-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1),
  [-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) by
  Fastino; library and paper code in
  [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1),
  fine-tuned from [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small)
  ([JHU-CLSP/mmBERT](https://github.com/JHU-CLSP/mmBERT))
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)
- [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- the other 83 checkpoints: Hugging Face IDs and revisions in
  `data/k28/models.json`

Model families:

- BERT: [google-research/bert](https://github.com/google-research/bert)
- ELECTRA: [google-research/electra](https://github.com/google-research/electra)
- MiniLM: [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/minilm)
- RoBERTa and XLM-R: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
- DistilBERT: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- DeBERTa: [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)
- ModernBERT: [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)

Runtimes, formats and tools:

- ONNX Runtime and ONNX Runtime Web: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- PyTorch: [pytorch/pytorch](https://github.com/pytorch/pytorch)
- Hugging Face Hub: [huggingface.co](https://huggingface.co/)
- safetensors: [safetensors/safetensors](https://github.com/safetensors/safetensors)
- WebGPU and WGSL: [W3C WebGPU](https://www.w3.org/TR/webgpu/), [W3C WGSL](https://www.w3.org/TR/WGSL/)
- AssemblyScript: [AssemblyScript/assemblyscript](https://github.com/AssemblyScript/assemblyscript)
- Emscripten (not used, named for contrast): [emscripten-core/emscripten](https://github.com/emscripten-core/emscripten)
- Playwright: [microsoft/playwright](https://github.com/microsoft/playwright)
- GLiNER2.5 CoreML export, from which `convert/gliner2_export.py` is
  adapted: [FluidInference/gliner2-5-small-coreml](https://huggingface.co/FluidInference/gliner2-5-small-coreml)

Browsers and platforms:

- Chromium: [chromium.org](https://www.chromium.org/)
- Brave: [brave/brave-browser](https://github.com/brave/brave-browser)
- WebKit: [WebKit/WebKit](https://github.com/WebKit/WebKit)
- Safari: [apple.com/safari](https://www.apple.com/safari/)
- Firefox: [mozilla-firefox/firefox](https://github.com/mozilla-firefox/firefox)
- NVIDIA L4: [nvidia.com](https://www.nvidia.com/en-us/data-center/l4/), run on [Modal](https://modal.com/)

Datasets and benchmarks:

- Typed Decisions: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- BTZSC: [btzsc/btzsc](https://huggingface.co/datasets/btzsc/btzsc)
- Jev benchmark protocol: [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)
- AG News: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
- DAIR Emotion: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Banking77: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- MASSIVE: [alexa/massive](https://github.com/alexa/massive), data from
  [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive)

Corpus and first user:

- argmin, a German learning platform by the author of kleinhirn:
  [smashburger-dev/argmin](https://github.com/smashburger-dev/argmin),
  app at [smashburger-dev.github.io/argmin](https://smashburger-dev.github.io/argmin/).
  500 of the 1,000 corpus texts come from its content (CC-BY-4.0).

## License

Apache-2.0, copyright Noa Katana (see `LICENSE`). Third-party models,
datasets, adapted code and corpus texts keep their own licenses (see
`NOTICE`).
