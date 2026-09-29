# Architecture

kleinhirn runs small encoder classifiers in the browser on WebGPU. First
model family: GLiNER2.5 (DeBERTa-v2 encoder plus schema classification).
Browsers without WebGPU get the same graph as WASM-SIMD in a worker.

## Principles

- One model, fixed buckets. Sequence length L from {64, 128, 256, 512,
  1024}, up to K = 16 labels per call (wide bucket L1280/K80 for larger
  label sets). All buffers and bind groups of a bucket are created at load
  time; a call only writes inputs, submits one command buffer and reads
  back K values.
- Fallback chain: WebGPU f16, WebGPU f32, then WASM-SIMD
  (`backend: 'auto'` chooses, `'webgpu'` and `'wasm'` force). Goal: every
  browser, including mobile devices.
- The reference is PyTorch fp32. Every engine number is checked against
  goldens from `convert/golden.py`.
- The engine computes; it decides nothing. Thresholds, abstention and the
  texts belong to the application.
- No runtime dependency besides WebGPU or the own WASM module.
  onnxruntime-web is only a measurement baseline in `bench/`, never part
  of the bundle.

## Layout

```
convert/            Python: checkpoint -> weight format, goldens, ONNX export for the baseline
src/
  index.ts          public API
  device.ts         adapter, device, limits, feature detection (shader-f16, subgroups)
  weights.ts        manifest loading, sha256 checks, shard upload
  tokenizer/        SentencePiece unigram, normalization, GLiNER2 schema input
  graph/deberta.ts  encoder plan: buffers, pipelines, bind groups per bucket
  kernels/*.wgsl    matmul, layernorm, GELU, relative attention, gather, softmax
  wasm/deberta.ts   AssemblyScript source of the CPU graph (f32, SIMD); built
                    to wasm/deberta.wasm with the pinned asc compiler
  wasm.ts           WASM backend: shard bytes, transposes, graph setup
  wasm-worker.ts    worker host; the WASM path always runs off the main thread
  wasm-client.ts    client facade with the same API as Kleinhirn
  worker.ts         optional worker host with a message protocol
bench/              measurement pages, Playwright runners, metrics.ts, JSON results
tests/              Node tests (tokenizer, manifest), browser parity tests
models/             weights and large goldens, not in git (tools/fetch_models.py)
tools/              corpus builder, pinned model and dataset downloaders
```

## Weight format

`models/<name>/manifest.json` plus shards `weights-<n>.bin`, each under
95 MB.

```json
{
  "format": "kleinhirn-weights",
  "version": 1,
  "source": { "repo": "fastino/gliner2.5-small-v1", "revision": "7e6f537f10337497069276892a5ef435028252ce", "checkpointSha256": "..." },
  "encoder": { "arch": "deberta-v2", "hiddenSize": 384, "layers": 12, "heads": 6, "intermediateSize": 1536,
               "vocabSize": 128011, "positionBuckets": 256, "maxRelativePositions": 512,
               "attSpan": 256, "relEmbeddingRows": 512,
               "posAttType": ["p2c", "c2p"], "shareAttKey": true, "layerNormEps": 1e-7 },
  "head": { "type": "gliner2-classification", "temperature": 1.0 },
  "tokenizer": "tokenizer.json",
  "tensors": [ { "name": "layers.0.qkv.weight", "dtype": "f16", "shape": [1152, 384], "shard": 0, "offset": 0, "byteLength": 884736 } ],
  "shards": [ { "file": "weights-0.bin", "bytes": 94371840, "sha256": "..." } ]
}
```

Rules:
- dtype per tensor: `f32`, `f16` or `i8` (then with `scale` per output
  channel, symmetric).
- Q, K, V are fused into `qkv` (one matmul).
- Precomputed in the converter: `rel_embeddings` after LayerNorm; per
  layer `pos_key = key_proj(rel)` and `pos_query = query_proj(rel)` (with
  `share_att_key`). The engine never computes these constants at runtime.
- Offsets are 256-byte aligned.
- The embedding table stays on the CPU (see `RESEARCH.md`). It exceeds
  the shard limit and is stored in row chunks: several `tensors` entries
  share a name with `rowStart`/`rowEnd` and `keepOnCpu: true`; the loader
  joins the rows in order.

## Tokenizer and schema input

`src/tokenizer/` reimplements `tokenizer.json` without a runtime
dependency:

- `tokenizer.ts`: the normalizer sequence from the file (whitespace
  collapse, NFC, right strip), added tokens with `normalized=false` (the
  schema tokens are searched only after normalization, since they survive
  it unchanged), a metaspace pretokenizer that splits only at the ASCII
  space. The checkpoint uses no Precompiled-Charsmap normalizer; with one,
  the table would have to be implemented, not guessed.
- `unigram.ts`: Viterbi over the piece scores in a codepoint trie,
  unknown characters as `minScore - 10`, consecutive UNKs merge as in HF
  tokenizers.
- `schema.ts`: builds the GLiNER2 classification input after
  `SchemaTransformer` and `ExtractorCollator` (gliner2 2.0.0): per task
  `( [P] <task> ( [L] <label> ... ) )`, structures joined by
  `[SEP_STRUCT]`, then `[SEP_TEXT]` and the text words split by `WORD_RE`
  in lowercase. Markers are the `[L]` positions (the `[P]` position is
  dropped via `positions[1:]`), `marker_groups` is the structure index.
  Bucket overflow or more than the marker cap throws
  `BucketOverflowError`.

## Encoder pass (DeBERTa-v2)

Per call, bucket (L, K):

1. JS: tokenization, schema input, read embedding rows, pad to L x H,
   `writeBuffer`.
2. LayerNorm on the embeddings, then apply the mask (DeBERTa multiplies
   embeddings by the mask).
3. Per layer:
   - `qkv = x @ Wqkv^T + b` (one matmul).
   - Relative attention in one kernel per (head, query row):
     `s = (q·k^T + c2p + p2c) / sqrt(d * 3)`, then masked softmax, then
     `· v`.
     - `c2p[i, j] = q_i · pos_key[clamp(r(i, j) + span, 0, 2 span - 1)]`
     - `p2c[i, j] = k_j · pos_query[clamp(r(i, j) + span, 0, 2 span - 1)]`.
       HF computes this as a gather with `-r` and transposes afterwards;
       because the bucket function is odd, the same index `r(i, j) +
       span` results.
     - `span = position_buckets = 256`, `r(i, j)` = bucket of `i - j`
       after `make_log_bucket_position` (buckets 256, max 512). For
       |i - j| <= 128, `r = i - j`; above that the logarithmic buckets
       apply. The table `r` per bucket length is built at load time.
     - Scale: `sqrt(64 * 3)` for all three terms.
   - Output projection, residual, LayerNorm (eps 1e-7).
   - FFN 384 -> 1536 with exact GELU (erf), -> 384, residual, LayerNorm.
4. Head: gather states at the label markers, GLiNER2 classifier
   (structure read from the checkpoint), divide by `temperature`, masked
   softmax per task.
5. Read back only the probabilities (and optionally logits) of the valid
   labels.

## Kernel layout (measured, see FINDINGS)

- `matmul.wgsl`: 16x16 tiles, `C = A @ W^T + B` with W in [N,K] row
  layout, f32 accumulation, optional fused ReLU/GELU (erf via
  Abramowitz-Stegun 7.1.26). Tile 64 was slower and was discarded.
- `attention.wgsl`: one workgroup (64 threads) per (head, query row). The
  q row lives in registers, the j loop loads only k/pos_key/pos_query;
  masked query rows write zeros and exit early. Scores live in workgroup
  memory, softmax via tree reduction; masked keys are exactly 0 after
  normalization and are skipped during V accumulation, so stale rows
  (`j >= seqLen`) cannot enter the accumulator as 0*NaN.
- Dispatch sizes follow `seqLen`, not the bucket length: rows beyond
  `seqLen` are never read (mask, markers and gather touch only valid
  positions), so matmuls, layernorms and attention dispatch only
  `ceil(seqLen/16)` or `seqLen` rows. The parity capture path still runs
  the full bucket.
- `layernorm.wgsl`: one mode per use (plain, embedding mask, residual),
  tree reduction over 64 threads, eps from the manifest.
- `gather.wgsl` plus `masklogits.wgsl`: gather marker states, then
  `logit/TEMP` with -1e4 for invalid markers; the rest of the head runs
  through `matmul.wgsl` (fc1 ReLU, fc2), the group softmax in JS.
- Fusion attempts (linear plus LayerNorm, the FFN pair) were slower than
  separate dispatches even with transposed weight layouts; on M-class
  GPUs the separate 16x16 tiled matmul pays off.

## Classification head and markers

Measured on the public checkpoints (small/base/multi-upstream, gliner2
2.0.0):

- `classifier` is `nn.Sequential`: `Linear(h -> 2h)` (`classifier.0`),
  `ReLU` (`classifier.1`), `Dropout(0.1)` (`classifier.2`, inference
  neutral), `Linear(2h -> 1)` (`classifier.3`), i.e. 384/768 for hidden
  384 and 768/1536 for hidden 768 (base, multi). Exactly one logit per
  marker. The engine reads the head width from the manifest
  (`head.hiddenSize = 2 * encoder.hiddenSize`), not from a constant.
- `logit = classifier(state_at_marker) / temperature`, `temperature` from
  `native.boundary_settings.classification_temperature` (1.0 in both
  checkpoints).
- Markers: the collator (`ExtractorCollator`, `PreprocessedBatch`)
  provides `cls_marker_indices` (positions of the label markers in the
  token stream, per task group `schema_special_positions[group][1:]`, the
  group marker at `[0]` is not scored), `cls_marker_mask` (valid markers)
  and `cls_group_index` (schema group index per marker; for pure
  classification schemas that is the task order 0..n-1). Example
  two-task schema `topic` (10 labels) + `sentiment` (3 labels): 13
  markers, `marker_groups = [0]*10 + [1]*3`.
- Softmax runs per group over its valid markers, not over all markers;
  invalid markers get -1e4 before the softmax. `multi_label: false` in
  the goldens (softmax, not sigmoid).

Delicate spots (goldens cover each):
- Log buckets from |i - j| >= 129, i.e. only in L256.
- Fully masked query rows must not produce NaN. Only valid positions are
  compared.
- fp16: accumulation in f32; a finite mask value (-1e4) or masking as a
  skip.
- Several tasks in one pass (GLiNER2 can combine several classifications
  in one sequence): softmax per task group, not over all markers.

### More than 16 labels

The standard path caps a schema at K = 16 markers. For larger label sets
(Banking77: 72) the chosen path is a wide bucket L1280 with up to K = 80
markers, not multiple calls: native scoring of all labels in one
sequence is exactly the protocol the K9 GLiNER reference numbers were
measured with, and splitting into subsets changes the schema layout and
would have to be re-measured for accuracy. The bucket provably stays
within the minimum limits (`tests/limits.test.ts` covers its dispatches
and bindings; f32/f16 parity under `limits: 'minimum'` in FINDINGS).
Cost: GPU buffers grow with L (base f32 about 498 MB) and the sequence
carries the 72-label schema. Applications on memory-constrained devices
can instead call client-side in K-16 subsets; that is the fallback, not
the implemented path. The head matmuls dispatch `ceil(K/16)` row tiles.

## Backend selection and WebGPU limits

`loadEngine(options)` (or `Kleinhirn.load` directly for the GPU path)
selects the backend:

1. `backend: 'auto'` (default): WebGPU if `navigator.gpu` exists and an
   adapter arrives; otherwise WASM-SIMD. A failure while loading the GPU
   path falls back to WASM as well.
2. `backend: 'webgpu'` or `'wasm'`: forces the path (measurements,
   diagnosis).
3. Precision on WebGPU: f16 if the manifest ships f16 and the adapter
   supports `shader-f16`, else f32 (option `precision`).

`limits: 'minimum'` (default): the device is requested with exactly the
limits required by the WebGPU standard (`MINIMUM_LIMITS` in `device.ts`,
including 256 invocations per workgroup, 16 KiB workgroup memory, 128 MiB
per storage binding). A kernel can then never silently demand more on a
desktop adapter than a mobile adapter provides; the parity and latency
runs use this mode. `'default'` leaves the negotiation to the adapter.

All kernels stay within these minimum limits: matmul 16x16 workgroups
(256 invocations), layernorm and attention 64 threads, the attention
workgroup score buffer `array<f32, L>` takes 4 KiB at L1024, the relative
position table is 4 MiB at L1024 and stays under 128 MiB per binding.

`timestamp-query`: requested when the adapter offers it. `Kleinhirn.
profile(input)` then returns GPU times per compute pass, else `null`;
measurements then fall back to the CPU clock. The profile run keeps the
multi-pass structure because timestamps are only written at pass
boundaries.

## WASM-SIMD fallback

`src/wasm/deberta.ts` implements the same DeBERTa-v2 graph in
AssemblyScript (f32, `v128` SIMD), all buckets 64 to 1024 in one
instance. Build: `npm run build:wasm` with the exactly pinned
`assemblyscript` devDependency; no Rust, no Emscripten. Runtime:

- The WasmClient starts a worker (`wasm-worker.ts`) that loads
  `WasmDeberta`: shards over the same `fetchShardBytes` route as the GPU
  path (sha256 check retained), matmul weights are transposed from [N,K]
  to [K,N] at load time so the SIMD kernel can vectorize over output
  columns.
- `forward(seqLen, bucketLen)` computes embedding LayerNorm, per layer
  QKV, relative attention, output projection, FFN, both LayerNorms, then
  the GLiNER2 head, matching the WGSL graph.
- Selection via `loadEngine`: automatic without `navigator.gpu` or
  adapter, or forced via `backend: 'wasm'`. The module is embedded into
  the worker bundle as a data URL via Vite `?url` import; it counts
  toward the bundle size (raw and gzip, see FINDINGS).

## Precision paths

| Path | Storage | Compute | When |
|---|---|---|---|
| f32 | f32 | f32 | parity, fallback without `shader-f16` |
| f16 | f16 | f16 with f32 accumulation | default when `shader-f16` exists |
| wasm | f32 | f32 with v128 SIMD | CPU fallback without WebGPU |
| i8w | int8 per channel, f16 activations | dequant in the kernel | candidate for smaller downloads |

## API

```ts
const kh = await loadEngine({ manifestUrl, buckets: [128, 256, 512, 1024],
                              precision: 'auto', backend: 'auto', limits: 'minimum' });
const out = await kh.classify(text, [{ task: 'topic', labels: ['...', '...'] }]);
// out.tasks[0] = { task, labels: [{ label, probability, logit }] }; out.timings = { tokenizeMs, gpuMs, totalMs }
await kh.profile(preparedInput);  // GPU times per pass, null without timestamp-query
kh.info();     // { precision, adapter, limitsMode, timestamps, buckets, gpuBytes, downloadBytes, loadTiming }
kh.dispose();
```

Interfaces reserved for later, currently only types and documentation:
- `kh.extract(text, entityTypes)` for the GLiNER2.5 boundary head.
- `kh.embed(text)` with mean pooling for cosine matching.

Worker: `src/worker.ts` hosts an instance; messages `load`, `classify`,
`dispose`, replies with `id`. WebGPU in workers runs in Chromium, Safari
26 and Firefox.

## ModernBERT and the Julia head

Sources: `models/julia-1/repo` (SupersonicLabs/Julia-1, pinned revision in
`provenance.json`) and `models/julia-1/onnx` (SupersonicLabs/Julia-1-ONNX
with the 100 reference requests). Executable spec in numpy:
`convert/julia_manifest_forward.py`.

### Encoder (ModernBERT, `encoder/config.json`)

22 layers, hidden 384, 6 heads of 64, `norm_eps` 1e-5, no biases anywhere
in the encoder. Embedding: `word.weight` + `embeddings.norm` (LayerNorm
without bias). Per layer pre-norm: `attn_norm` (layer 0 is identity; the
tensor does not exist in the checkpoint), `Wqkv` without bias
[384 -> 1152], RoPE with theta 160000 and rotate-half convention on
positions `arange(n)`, attention without bias, `Wo`, then `mlp_norm` and
GeGLU: `Wi` [384 -> 2304] is split into input [0:1152] and gate
[1152:2304], output `gelu(input) * gate`, `Wo` [1152 -> 384]. Finally
`final_norm`.

Attention rule (`layer_types` in the config): layer index % 3 == 0 is
global (full attention), otherwise local with window `local_attention`
128, i.e. 64 each side: |i - j| <= 64 is allowed, beyond that the score
is masked. Both types use theta 160000 (`rope_parameters` full and
sliding identical). Scale 64^-0.5.

### Input format (`julia/data.py:sequence`)

Request: `{state, question, options[2..20], type}` with `type` from
`choice` (0), `score` (1), `noul` (2); `noul` requires exactly the
options `[false, true]`. Serialization:

```
[CLS] "<type> question: <question>" [SEP] ([MASK] <option_i>) x k [SEP] <state> [SEP]
```

- Markers: position of each `[MASK]` token; each option at most 48
  tokens after it.
- `head_length` 256 covers question plus options; remaining budget
  `max(8, 256 - options)` for the question, fallback shortens options to
  `max(4, (256-16)/k)` tokens.
- `state` fills the rest up to `max_length` (1024 in their inference, up
  to 8192 configurable).
- Strict encoding (reference requests and goldens): the `[MASK]` token
  may not appear in any input text, options <= 48 tokens, question and
  state must fit their budget losslessly.

### Decision head (`julia/model.py`)

`type_emb[qtype]` is added to the `final_norm` output, then two
`nn.TransformerEncoderLayer` (384 wide, 6 heads, FFN 1536 with ReLU,
`norm_first=True`, bias everywhere, padding mask from `attention_mask`).
Then gather at the marker positions and the scorer
`LayerNorm(384) -> Linear(384->384) -> GELU -> Linear(384->1)`. Invalid
markers are filled with -1e4; the choice is made by argmax/softmax over
the valid options. `type` only affects the question prefix and
`type_emb`, not the computation afterwards. The optional `act_head`
(routing actions) is not needed and is not exported.

### Numerical conditioning

A few positions carry very large activations (measured max |h| about
2500). There every small deviation amplifies: PyTorch's own f32 path
deviates from its own f64 result by up to 6e-2 at exactly those
positions (measured in `convert/julia_manifest_forward.py --check`),
while the relative error stays <= 1.2e-4 everywhere. An absolute 1e-4
gate on layer states therefore cannot be passed by an independent
implementation. The numpy spec gate is: per case, the spec's maximum
deviation to the f64 reference may be at most the f32 reference's
deviation to f64 (`layers.f32.bin` as the band); the spec computes in
f64 and thus stays on the same side of the cliff.

For the f32 WGSL engine the same idea applies with a factor: per case
max |WGSL - ref64| at most 3 x band, because two independent f32 samples
can differ by up to 2 x band (triangle inequality) and factor 3 covers
the spread of a single band value. Additionally the states before the
known jump layers (11, 18) must stay under 1e-4 so a wrong step before
the cliff cannot be hidden by the factor. The logit gate stays absolute:
100/100 equal choices, max 0.00225 against the published logits (f32),
f16 at least 99/100.

### WGSL engine (src/graph/julia.ts, src/julia.ts)

Routing in `index.ts` on `manifest.encoder.arch === "modernbert-julia"`;
the DeBERTa engine stays unchanged. The Julia plan creates buffers,
pipelines and bind groups per bucket (L512, L1024) once at load. New
kernels: `mbattention.wgsl` (one workgroup per (head, row), window
override for local layers, 0 for global), `rope.wgsl` (in-place on the q
and k sections, rotate-half, cos|sin table per position), `geglu.wgsl`
(splits the 2304-wide projection into input|gate, erf-GELU), `add.wgsl`
(residual and type-embedding broadcast). Reused: `layernorm`, `matmul`
(ACT 0/1/2 for none/ReLU/GELU), `gather`, `masklogits`.

Data flow: on the CPU the embedding rows of the input ids are placed
into a buffer (keepOnCpu; 256k x 384 would be 384 MB and exceeds the
128-MiB binding limit), `embeddings.norm` writes to `x`. The 22 layers
run in one pass (multi-pass only for capture/timestamps); layer 0 skips
`attn_norm` (identity, the tensor does not exist in the checkpoint).
`final_norm` writes to `tmp`, not back to `x`: in- and output of the
same buffer in one dispatch is not allowed. From `final_norm` the stream
runs on `tmp` (type-embedding add, two head blocks, marker gather,
scorer, mask logits). The `type_emb` row is copied from the weight
tensor via `copyBufferToBuffer` (the tensor buffers therefore carry
`COPY_SRC`). On the input side `tokenizer/julia-input.ts` builds the
sequence like `julia/data.py:sequence` (strict rules, markers, qtype).

## Measurement protocol

Applies to kleinhirn and every comparison runtime alike. Core:
`bench/metrics.ts`.

Latency:
- Dataset: the 1,000 public corpus texts
  (`tests/corpus/texts1000.json`), task `topic` with the 10 demo labels,
  bucket L128 K16. Texts that do not fit the bucket are skipped and
  counted.
- Warmup: the first 20 texts once, discarded.
- Measurement: all texts sequentially, exactly one call at a time. Clock
  `performance.now()` before tokenization, stop when the probabilities
  are a Float32Array in JS.
- Additionally "model only": clock starting at finished input arrays so
  ORT and kleinhirn compare without tokenizers.
- Metrics: median and p95 after `metrics.ts`.
- Browser: Playwright Chromium visible (not headless), version and
  `adapter.info` in the result. Laptop on mains power, no other heavy
  tabs.
- System load: `uptime` before every official run; the result JSON
  carries the load averages (`sysctl vm.loadavg`). If the 1-minute load
  is above 4, the run counts as provisional and is repeated under calm
  load.

Parity:
- Reference: PyTorch fp32, the same wrapper as the export, the same
  schema.
- `compareLogits` from `metrics.ts`: argmax agreement, maximum and mean
  absolute logit deviation, maximum probability deviation.

Memory:
- `gpuBytes`: exact sum of all GPUBuffer sizes the engine allocates.
- Download: sum of shards, tokenizer and JS bundle, raw and gzip.
- OS-level peak: the runner samples the memory of all Chromium processes
  (renderer and GPU process) every 100 ms, from before load to after the
  measurement. Metric: peak minus value before load.
- Calibration before first use: a test page allocates a 512-MB GPU
  buffer and describes it. The chosen metric (candidates: `footprint -p`,
  `top -stats mem`, `ps -o rss`) must rise by 450 to 600 MB. Metrics that
  do not show this are discarded and the choice is recorded in
  `FINDINGS.md`.

Protocol: every run produces a result JSON under `bench/results/`. The
kleinhirn runner measures the Vite library bundle `dist/kleinhirn.js`;
the dev server can serve a stale transform copy of `dist/` after
`vite build`, so the bundle carries a build id (`__KH_BUILD_ID__`,
visible via `kh.info().buildId`) and `bench/run-kleinhirn.mjs` aborts
when the served file does not contain the id of the file on disk.
Restart the dev server after a build or trust the check.
