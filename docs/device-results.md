# Device result files

The device benchmark page writes one JSON file per run. The file has the
schema id `kleinhirn-device-result/1`. The machine-readable schema is
`device-result.schema.json`, published next to the page.

Parity is verified by script: `tools/check_device_result.mjs` recomputes it
from the logits in the file against the goldens and applies the same gates
as the page. Latency is self-reported: nobody re-runs it, and one browser
tab on one device is one sample.

## Fields

| Field | Meaning |
|---|---|
| `schema` | Always `kleinhirn-device-result/1`. |
| `createdAt` | ISO time of the end of the run. |
| `page` | `url` without query, `buildId` of the page build, `engineCommit` (git short commit the page was built from, `-dirty` if the tree had changes), `engineBuildId` (build id stamped into the engine bundle), `bundleSha256` and `workerSha256` (hash of the exact `kleinhirn.js` and WASM worker the page ran). |
| `weights` | `base` URL of the weights host. Per precision (`f16`, `f32`): `manifestSha256`, the `shards` list from the manifest (`file`, `bytes`, `sha256`), `tokenizerBytes`, and `error` if the host could not be read. |
| `goldens` | `sha256` of the golden sets the run was scored on (checked against the repository goldens), the model, the buckets (`L128`, `L256`) and the sha256 of the source golden files. |
| `device` | What you typed: `model`, `os`, and the four protocol checkboxes (`pluggedIn`, `screenOn`, `lowPowerModeOff`, `otherTabsClosed`). |
| `environment` | What the page collected: user agent and low-entropy client hints, core count, device memory, secure context, `Float16Array`, OPFS, storage quota, and WebGPU (`present`, adapter `info`, sorted `features`, all `limits`). `rangeProbe` is a `Range: bytes=0-1023` request to the first shard on the weights host: status, bytes received, `content-range`, or the error text. It shows whether range requests and CORS work from this browser. `storageMarker` (optional) tells how long browser storage survives on the device. The page keeps a small marker `{firstSeen, lastSeen, visits}` in OPFS (`kh-marker.json`), in Cache Storage (cache `kh-marker`, key `/kh-marker`) and in localStorage. At each load, before measuring, it reads all three and reports per store: `present`, `firstSeen`, `lastSeen`, `visits`, `ageDays` (since `firstSeen`), `readError` and `writeError`. Then it updates the markers. The first visit shows all three absent. The marker holds only timestamps and a counter. |
| `autoStage` | The stage the engine picks when nothing is forced (`backend: auto`, `precision: auto`): `picked`, the manifest it loaded, `error`, and `reusedAsMeasured` (the auto load counted as the measured load of that stage). |
| `stages[]` | One entry per measured stage, in the order `f16`, `f32`, `wasm`. |

## Stage entries

| Field | Meaning |
|---|---|
| `name` | `f16` (WebGPU with shader-f16), `f32` (WebGPU), `wasm` (WASM-SIMD in a worker). All stages run with `limits: minimum`. |
| `ok`, `error`, `errorPhase` | `ok` is false when the stage could not run. `error` is the engine's message verbatim, for example a missing feature or a limit below need. `errorPhase` is `load` or `measure`. |
| `load` | `wallMs` of the load call, the engine's `loadTiming` breakdown, backend, precision, `hasF16`, limits mode, `gpuBytes` (sum of all GPU buffers the engine allocates, null for wasm), `downloadBytes`. |
| `parity` | Per bucket: `summary` (argmax agreement, max and mean absolute logit difference, max probability difference, indices of disagreeing cases), `pass`, `missing` (cases that did not fit), and `perCaseLogits` (logits of every case, rounded to six significant digits). 200 cases per bucket. |
| `parityPass`, `gateRule` | Both buckets pass. `exact` (f32, wasm): argmax 100 % and max logit difference at most 1e-3. `f16`: argmax at least 99.5 % and max probability difference at most 1e-2. |
| `latency` | Per bucket (`L128`, `L256`), for `e2e` and `modelOnly`: `n`, `medianMs`, `p95Ms` (sorted[floor(0.95 n)]), min, max, mean, and `skipped` (items that did not fit). 20 warm-up items are discarded, then every item runs once, one call at a time. `e2e` starts the clock before tokenization and stops with probabilities as a `Float32Array` in JS. `modelOnly` starts from prepared arrays. The result cache is off. |
| `memory` | `gpuBytes`, and the JS heap of the page thread (`jsHeapBaselineBytes`, `jsHeapPeakBytes`) where the browser exposes `performance.memory` (Chromium). Elsewhere `jsHeapMeasurable` is false. iOS has no web API for peak memory. The wasm module runs in a worker and is not part of the page heap. |

## Contributing

Open an issue with the "Device result" form and drag the downloaded file into
the result box. A maintainer runs `node tools/check_device_result.mjs <file> --add`,
which validates the file, recomputes parity, stores it under `community/` and
updates `docs/COMMUNITY.md`.
