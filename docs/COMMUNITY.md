# Community results

Results contributed from real devices through the device benchmark page and a GitHub issue.
These numbers are separate from the official matrix, which is measured by us with a fixed protocol.

- Parity (pass or fail) is verified by script: `tools/check_device_result.mjs` recomputes it from the logits in each file against the goldens.
- Latency (p95, model only) is self-reported by the device and not verified.
- "Auto" is the stage the engine picks on that device when nothing is forced.
- `n/a` means the stage did not run on that device; `-` means it was not selected.

| Device | Browser | Adapter | Auto | f16 | f32 | wasm | L128 p95 ms (f16/f32/wasm) | L256 p95 ms (f16/f32/wasm) | Date |
|---|---|---|---|---|---|---|---|---|---|
