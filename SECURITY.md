# Security

kleinhirn runs entirely in the browser: no server, no account, the text
stays on the device. Software still has bugs, and we are glad about
every report.

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability. Use the
confidential report on GitHub instead:

**Security → [Report a vulnerability](https://github.com/smashburger-dev/kleinhirn/security/advisories/new)**

Describe what you found and how to reproduce it. You get an answer
within a few days; the fix comes as fast as possible, and you are named
in it if you want.

## What counts as a security problem

- Code execution or script injection through a model, a manifest, a
  tokenizer file or a result file
- Weights or tokenizer files that pass the sha256 checks although they
  differ from the pinned source
- Reads or writes outside a buffer in the WGSL kernels or the WASM
  module
- Data leaving the device that the README says stays on it, for example
  from the device benchmark page

Wrong answers, slow runs and other bugs go into a
[normal issue](https://github.com/smashburger-dev/kleinhirn/issues/new/choose).

## Supported version

There is one version: the current state of `main`. The device page at
https://smashburger-dev.github.io/kleinhirn/ is published from it.
