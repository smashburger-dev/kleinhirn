# K28 review evidence

Report: `docs/reviews/2026-10-01-k28-engine-review.md`.
Reviewed source: `dfc4b24`, after the explicitly requested ff-only update from `bf91b9a`.
No production fixes. No browser, GPU, server, downloads or installed dependencies.

From the repository root:

```sh
# Intentionally fails. Tests of fixed findings moved into the normal glob (K28.R, tests/k28r_*.test.mjs);
# the rest are deferred or belong to groups D to F.
node --import ./tests/helpers/node-hooks.mjs --test tests/review/engine.test.mjs tests/review/tokenizer.test.mjs

# Five passing tests: static sweep, numerical controls, stored reference fixtures,
# and T04 characterization, which is NOT verified against HF.
node --import ./tests/helpers/node-hooks.mjs --test tests/review/controls.test.mjs

# CPU-only microbenchmarks. This replaces the stored measurements.
node --expose-gc --import ./tests/helpers/node-hooks.mjs tests/review/bench.mjs tests/review/cpu-results.json

# Offline build analysis. First run the existing build; no server is started.
npm run build
node tests/review/bundle.mjs tests/review/bundle-results.json
```

These `.mjs` tests are deliberately outside the normal `npm test` glob. Failing
contracts must not be mistaken for failures in the existing baseline suite.

`tests/helpers/review.mjs` uses the existing recording GPU mock. It does not execute WGSL.
Kernel numerical cases are CPU transcriptions or explicit arithmetic contracts.
The two synchronous hangs run in child processes with a 1500-ms limit; an stderr
marker confirms that imports finished before the timeout.

`cpu-results.json` was measured on `bf91b9a`, under parallel load. The measured
functions in encoder/index/julia/half/tokenizer did not change in the ff update.
It contains samples and medians, not engine latency or model parity numbers. The
34-MB JSON is generated in memory, not a real model tokenizer. It is not saved.

`bundle-results.json` is regenerated for `dfc4b24`. Final byte attribution uses
source maps plus exact WGSL literal matching. Marginal gzip figures are
non-additive size probes, not executable smaller bundles. Both simplification
experiments modify source only in memory.

`checks-results.json` records the final command outcomes. The new boundary cases
have no freshly generated HF oracle; T01 uses Unicode regex semantics, T03 is an
observed hang, and T04 remains explicitly unverified. See the report's last section.
