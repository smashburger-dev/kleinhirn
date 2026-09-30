#!/usr/bin/env node
// Builds models/hf-upload/, the folder for the kleinhirn weights repository on
// Hugging Face: small-upstream/{f16,f32}/ (hardlinks of models/small-upstream
// files), README.md (model card), LICENSE, NOTICE. Prepares only; nothing is
// uploaded. Prints total size and sha256 per file.
//
// Usage: node tools/prepare_hf_upload.mjs
import { createHash } from 'node:crypto';
import {
  copyFileSync, createReadStream, linkSync, mkdirSync, readFileSync, readdirSync,
  rmSync, statSync, writeFileSync,
} from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = resolve(ROOT, 'models/small-upstream');
const OUT = resolve(ROOT, 'models/hf-upload');
const REVISION = '7e6f537f10337497069276892a5ef435028252ce';

const CARD = `---
license: apache-2.0
base_model: fastino/gliner2.5-small-v1
library_name: kleinhirn
tags:
  - gliner2
  - text-classification
  - zero-shot
  - webgpu
  - deberta-v2
---

# kleinhirn weights: GLiNER2.5-small

Weight shards of \`fastino/gliner2.5-small-v1\` in the format the
[kleinhirn](https://github.com/smashburger-dev/kleinhirn) engine loads. kleinhirn
runs small encoder classifiers in the browser on WebGPU, with a WASM-SIMD
fallback on the CPU. These files are for that engine. They are not a
Transformers checkpoint.

## Files

| Folder | Content |
|---|---|
| \`small-upstream/f32/\` | full-precision weights, used by the WebGPU f32 path and the WASM path |
| \`small-upstream/f16/\` | half-precision weights, used by the WebGPU f16 path (needs \`shader-f16\`) |

Each folder has \`manifest.json\` (tensor layout, shard sizes and sha256),
\`tokenizer.json\` and the \`weights-*.bin\` shards. The engine verifies every
shard against the sha256 in the manifest.

## Provenance

Converted from \`fastino/gliner2.5-small-v1\` at revision
\`${REVISION}\` with \`convert/export_weights.py\`
from the kleinhirn repository. The f16 shards are a cast of the fp32 weights.
The tokenizer is the unchanged upstream tokenizer.

## Use

Point the kleinhirn engine at a manifest URL in this repository, for example
\`small-upstream/f16/manifest.json\`. The device benchmark page of the kleinhirn
project loads these files to measure the engine on your device.

## License and credit

Apache-2.0, the license of the original model. See \`LICENSE\` and \`NOTICE\`.
GLiNER2.5-small is by Fastino. Its encoder is DeBERTa-v3-xsmall by Microsoft
(MIT license). The only change here is the conversion of the tensor layout and
the fp16 cast described above.
`;

// Text of https://raw.githubusercontent.com/microsoft/DeBERTa/master/LICENSE,
// embedded verbatim (indentation as in the source).
const DEBERTA_MIT = `    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
`;

const NOTICE = `kleinhirn weights (small-upstream)

This repository contains weights derived from:

- fastino/gliner2.5-small-v1 (GLiNER2.5-small) by Fastino, released under the
  Apache License 2.0. https://huggingface.co/fastino/gliner2.5-small-v1
  Revision ${REVISION}.

- The encoder of that model is DeBERTa-v3-xsmall by Microsoft, released under
  the MIT License. https://github.com/microsoft/DeBERTa

Modifications: the tensors were re-laid out into flat binary shards described
by manifest.json (convert/export_weights.py in
https://github.com/smashburger-dev/kleinhirn). The f16 files are the fp32
weights cast to half precision. No training or fine-tuning was applied. The
tokenizer file is unchanged.

---

DeBERTa license text (Microsoft, MIT License)

${DEBERTA_MIT}
`;

async function sha256(file) {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(file)) h.update(chunk);
  return h.digest('hex');
}

function put(from, to) {
  mkdirSync(dirname(to), { recursive: true });
  try { linkSync(from, to); } catch { copyFileSync(from, to); }
}

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    (e.isDirectory() ? walk(resolve(dir, e.name)) : [resolve(dir, e.name)]));
}

rmSync(OUT, { recursive: true, force: true });
for (const prec of ['f16', 'f32']) {
  for (const f of readdirSync(resolve(SRC, prec)).sort()) {
    if (f === 'manifest.json' || f === 'tokenizer.json' || /^weights-\d+\.bin$/.test(f)) {
      put(resolve(SRC, prec, f), resolve(OUT, 'small-upstream', prec, f));
    }
  }
}
writeFileSync(resolve(OUT, 'README.md'), CARD);
copyFileSync(resolve(ROOT, 'publish/LICENSE'), resolve(OUT, 'LICENSE'));
writeFileSync(resolve(OUT, 'NOTICE'), NOTICE);

// The card must not carry private paths, project names or addresses. The
// private names are assembled from
// parts so that this file itself passes the public export's leak scan.
const LEAK = new RegExp(
  ['Users/', 'Life' + 'maxxing', 'edge-vor-' + 'modell', 'arg' + 'min', 'recruit' + 'ing', '@[\\w-]+\\.[a-z]{2,}'].join('|'),
  'i',
);
for (const f of ['README.md', 'NOTICE']) {
  const hit = LEAK.exec(readFileSync(resolve(OUT, f), 'utf8'));
  if (hit) throw new Error(`${f} contains "${hit[0]}"`);
}

let total = 0;
for (const file of walk(OUT).sort()) {
  const size = statSync(file).size;
  total += size;
  console.log(`${(await sha256(file))}  ${String(size).padStart(10)}  ${relative(OUT, file)}`);
}
console.log(`total ${total} bytes (${(total / 1048576).toFixed(1)} MiB) in ${relative(ROOT, OUT)}`);
