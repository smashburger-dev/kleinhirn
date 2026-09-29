# K9 benchmark suite: protocol

The protocols are taken from the source artifacts, not invented. Where
an evaluation loop was not shipped (the classification pilots, MASSIVE),
the reconstruction is documented and justified below; the reproduction
gate (±1 percentage point) checks it.

## Pinned sources

| Source | Pin | Purpose |
|---|---|---|
| `github.com/AbdelStark/jev-benchmarks` | commit `0d610cc53e79bcbec691312b0c4adb4a0e371642` | classification pilots (AG News, DAIR Emotion, Banking77) |
| `huggingface.co/datasets/btzsc/btzsc` | revision `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` | pilot test data |
| `huggingface.co/datasets/LocalLLaMA/typed-decisions` | revision `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`, `all/test-00000-of-00001.parquet`, SHA-256 `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` | Typed Decisions |
| `huggingface.co/datasets/AmazonScience/massive` | parquet conversion `refs/convert/parquet` @ `ed58ac423a2f4121720918bf5301577edce4ffd3`, test split | MASSIVE |
| Julia checkpoint | SHA-256 `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72` | identical to `metrics/accuracy-20260924.json` |
| GLiNER2.5 multi | `fastino/gliner2.5-multi-v1@235cf92d6d4318da9bfca0d08975c8fa7250d13b` | reference revision from `configs/pilot-v1.yaml` |

Evidence: `scripts/reproduce_typed.py` in the Julia-1 repo (revision and
hashes in source), Julia-1 `README.md` (protocol paragraph),
`models/jev-benchmarks/configs/pilot-v1.yaml`,
`models/jev-benchmarks/src/jev_benchmarks/data.py`. All inputs are
fetched by `tools/fetch_datasets.py` at these pins.

## Common interface

Every route is run as a row `(state, question, options, type)`, as in
`julia/data.py:validate_row`:

- `state`: text or JSON object (objects are serialized via `json.dumps`,
  `ensure_ascii=False`, `data.py:77`).
- `question`: text; encoding as `"<type> question: <question>"` behind
  `[CLS]` (`data.py:82`).
- `options`: 2 to 20 non-empty descriptions; each option gets a `[MASK]`
  token prepended, at most 48 tokens per option (`data.py:83-86`).
- `type`: `choice` (qtype 0), `score` (qtype 1), `noul` (qtype 2);
  `noul` has exactly the two options `[false, true]` (`data.py:10,29`).

Historical benchmarks use `max_length=1024` and `head_length=512` with
`strict_encoding=True`: masker injection, overlong options and truncated
states are rejected, not clipped (`reproduce_typed.py:66-68`,
`data.py:78-105`). Rejected or failed requests count as abstention and
thus as an error in the denominator (model card: "Abstentions count as
incorrect"; `accuracy-20260924.json` shows `banking77: ok 99, abstained
1`).

## Typed Decisions (gate 73.15 %)

Source: `scripts/reproduce_typed.py` (fully shipped, no reconstruction
latitude).

- 400 test cases; each carries `state` (JSON), `questions` (named
  questions with `type`, `instructions`, `criteria`) and `gold`.
- Each question becomes a row: `state` unchanged, `question` from
  `instructions`, `options` from the `criteria` values in key order.
  `noul` without `criteria` gets the literals `{'false': 'false',
  'true': 'true'}`; keys `['false', 'true']` (`reproduce_typed.py:47-55`).
- Yields 2,000 requests: 600 choice, 800 score, 600 noul (assert in
  `reproduce_typed.py:62`).
- Engine: `FastEngine(device='cpu', transformer_backend='torch',
  strict_encoding=True, max_length=1024, head_length=512, batch_size=16,
  marker_only_head=False)`, one example per forward
  (`reproduce_typed.py:64-80`).
- Prediction: `keys[answer['index']]` against
  `gold[question_id]['label']`.
- Reference: 1,463/2,000 = 73.15 % (choice 428/600, noul 484/600, score
  551/800; `metrics/accuracy-20260924.json`). Their run was H200 BF16;
  our gate run is CPU FP32.

## Classification pilots after Jev/BTZSC (gates 94/100, 86/100, 64/100)

Model card: "classification pilots follow the pinned Jev benchmark
protocol using BTZSC". Sampling exactly as
`jev-benchmarks/src/jev_benchmarks/data.py`:

- Per dataset `btzsc/btzsc`, config `agnews` | `emotiondair` |
  `banking77`, split `test`, revision `fef2a2ac...`.
- The test rows carry one row per candidate hypothesis per example
  (`text`, `hypothesis`, `labels` 0/1). Class count = first repeat
  boundary of the text (`data.py:13-18`); label descriptions = the
  `hypothesis` strings of the first n classes (`data.py:65`).
- Only examples with exactly one positive label count (`data.py:68-73`);
  Banking77 thereby loses its 200 out-of-scope rows (protocol
  `docs/PROTOCOL.md`).
- Selection: `_balanced_indices(targets, 100, seed=20260917+offset)`
  with `random.Random`, shuffle per class, round-robin over sorted
  classes, return sorted (`data.py:21-38`). Offsets: agnews 0,
  emotiondair 1, banking77 2 (`data.py:52`).
- Verified locally: agnews 4 classes/7,600 valid samples/25-25-25-25,
  emotiondair 6/2,000/17-16, banking77 72/2,880 valid/1 to 2 per class.

Request (Jev adapter `adapters/jev.py:30-34`, mapped to the local
interface): `state = {"text": example.text}`, `question = "Which single
label best describes the input text?"`, `type = choice`, `options` = the
hypothesis strings in dataset order.

Banking77 (72 options > 20 limit). Model card: "a ranking/top-16
shortlist, not a native 72-option call". The evaluation loop is not in
the repo. Sweep over the shipped `Router` mechanism
(`julia/router/router.py`, groups of `width` options, `survivors` per
group, iterate until a final call remains; plus a noul ranking over all
72 hypotheses as an alternative):

| Mechanism | Result | Note |
|---|---|---|
| Router(9,2) | 69/100 | literal top-16 (8 groups of 2 survivors) |
| Router(19,4) | 61/100 | literal top-16 (4 groups of 4) |
| Router(18,4) | 62/100 | literal top-16 |
| Router(20,4) | 57/100 | literal top-16 (4 groups of 4) |
| Router(16,4) | 59/100 | 20 finalists |
| **Router(14,4)** | **64/100** | 22 then 8 finalists, exact match |
| Router(15,3) | 62/100 | |
| Router(14,3), (12,3) | 67/100 | |
| Router(12,4), (13,4) | 68/100 | |
| noul ranking top-16 | 31/100 | 72 noul calls + 16 final call, eliminated |

"top-16" is not unambiguously reconstructable; depending on the router
parameters it measures 57 to 69 of 100. Router(14,4) hits 64 exactly but
was chosen after the result, which is fitting to the target number, not
reproduction. The Banking77 gate is therefore OPEN, not passed.
Hypothesis: the published pipeline used a shortlisting with a different
grouping or option order (the parity cases show per-example permuted
options). The sweep is documented in
`bench/results/k9-banking77-sweep.json`. The published eval also had 1
abstention (denominator 99 answered); none of our runs abstained.
Comparisons on Banking77 only with the mode named: Julia with router,
GLiNER with all 72 labels natively.

## MASSIVE (gate 71.50 %): not reproducible, removed from the suite

- `AmazonScience/massive`, all 52 locales from the dataset card, test
  split 2,974 examples each = 154,648 decisions. Parquet conversion
  `refs/convert/parquet`, scenario names from the ClassLabel metadata.
- Task: scenario classification (18 labels), not intent or slots (model
  card). Reference: 110,573/154,648 = 71.50 % macro over the locales,
  en-US 86.75 %, pt-PT 86.25 %.
- No public evaluation loop anywhere (Julia-1 repo, ONNX repo,
  jev-benchmarks, evals.typesafe.ai, dataset cards). Format
  reconstruction was checked on the en-US TRAIN split only (never tuned
  on the test slice): about 50 variants of state (dict `{"text": utt}`,
  raw text, with locale, annot_utt), question (Jev question, "Which
  scenario/intent best describes this request?" among others) and
  options (scenario names, BTZSC hypothesis style "The example utterance
  is about X.", descriptions from the member intents, intent name
  lists). Best result: 68 % on train (dict + "Which intent best
  describes this request?" + hand-written scenario descriptions).
- Literal reconstruction like the pilots (`{"text": utt}` + Jev question
  + scenario names) measures on the TEST split: en-US 51.71 %, pt-PT
  49.26 % (instead of 86.75 % / 86.25 %). Deviation ~35 points, gate
  missed by far.
- Diagnosis: the published MASSIVE number was produced with a request
  format (option texts, possibly state form or phrasing) that cannot be
  derived from any public artifact; the checkpoint was trained on a
  format that was not shipped. The task was removed from the gated
  suite: no GLiNER or browser measurements on MASSIVE, no further format
  attempts.

## Reproduction gate results

| Task | Published | Local (CPU FP32) | Deviation | Status |
|---|---|---|---|---|
| Typed Decisions | 1,463/2,000 = 73.15 % | 1,451/2,000 = 72.55 % | -0.60 pp | passed |
| AG News | 94/100 | 94/100 | 0 | passed |
| DAIR Emotion | 86/100 | 86/100 | 0 | passed |
| Banking77 | 64/100 | ambiguous: 57-69 by router parameters (14,4 hits 64, chosen afterwards) | - | open |
| MASSIVE | 71.50 % macro | en-US 51.71 % (format unknown) | ~-35 pp | not reproducible, removed |

The typed deviation of 12 examples is within the expected range of CPU
FP32 vs H200 BF16 (published run). Per type: choice 426/600, noul
483/600, score 542/800.

## GLiNER2.5 routes (after the Julia gate passes per task)

Zero-shot classification with the same examples and options:

- Text = state plus question: serialized `state` (for objects
  `json.dumps`) + space + `question`. Options = the label descriptions
  of the same task.
- On BTZSC this matches the Jev adapter `adapters/gliner.py`:
  `ClassificationSchema().single("label", labels)`, exclusive schema,
  labels in dataset order, CPU, batch 1 (`configs/pilot-v1.yaml`).
- Models: `fastino/gliner2.5-small-v1`, `fastino/gliner2.5-base-v1`,
  `fastino/gliner2.5-multi-v1` (multi at revision `235cf92d...` as the
  pilot).
- Prediction: label with the highest probability; failed examples stay
  in the denominator (Jev `metrics.py`).

## kleinhirn Julia in the browser

- The same Typed Decisions rows as above, encoded by the own Gemma
  tokenizer, f16 engine in minimum-limits mode, one example per forward.
- Comparison yardstick: argmax equality and accuracy against the PyTorch
  Julia run of the same day (not against the H200 number).

## Result format

- JSON under `bench/results/`: `k9-<task>-<route>.json` with model,
  route, revisions, hits, count, accuracy, status per example.
- Per-example predictions live in `*.predictions.jsonl` with only
  `id`, `gold`, `predicted`, `probabilities`, no dataset texts.
- `docs/FINDINGS.md`: one row per (task, model, route) with hits/count.
- Timings only with a load note in the JSON; accuracy is
  load-independent.
