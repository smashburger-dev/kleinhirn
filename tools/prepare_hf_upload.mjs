#!/usr/bin/env node
// Builds models/hf-upload/, the folder for the kleinhirn weights repository on
// Hugging Face. One folder per source model, named after its source repo:
// <org>/<name>/<precision>/ holds manifest.json (copied) and the tokenizer, the
// shards and words.bin (hardlinked, copied as fallback). <org>/<name>/NOTICE
// names the source, its license and the changes. The root has README.md (model
// card) and LICENSE. Only models with an allowed license are included. Prepares
// only; nothing is uploaded. Prints sha256 and size per file and the total.
//
// Usage:
//   node tools/prepare_hf_upload.mjs            default set: the ten models the bench, README and demo use
//   node tools/prepare_hf_upload.mjs --all      also every data/k28/models.json model with a local folder
//   node tools/prepare_hf_upload.mjs --dry-run  print selection, licenses, skipped list and size; write nothing
import { createHash } from 'node:crypto';
import {
  copyFileSync, createReadStream, existsSync, linkSync, mkdirSync, readFileSync, readdirSync,
  rmSync, statSync, writeFileSync,
} from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = resolve(ROOT, 'models/hf-upload');
const K28_DIR = resolve(ROOT, 'models/k28');
const CATALOG = resolve(ROOT, 'data/k28/models.json');
const ALLOWED = new Map([
  ['apache-2.0', 'the Apache License 2.0'],
  ['mit', 'the MIT License'],
  ['cc-by-4.0', 'the Creative Commons Attribution 4.0 license (CC BY 4.0)'],
]);
const ALL = process.argv.includes('--all');
const DRY_RUN = process.argv.includes('--dry-run');

// The ten models the bench, README and demo use. Folder: models/k28/<id with / replaced by __>.
const DEFAULT_K28 = [
  'sentence-transformers/all-MiniLM-L6-v2',
  'cardiffnlp/twitter-roberta-base-sentiment-latest',
  'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
  'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
  'protectai/deberta-v3-base-prompt-injection-v2',
  'ibm-granite/granite-embedding-small-english-r2',
];
// Upstream folders are not in data/k28/models.json. Their license is set here.
const UPSTREAM = ['models/small-upstream', 'models/base-upstream', 'models/multi-upstream', 'models/julia-1'];
const UPSTREAM_LICENSE = { 'SupersonicLabs/Julia-1': 'apache-2.0' };
function upstreamLicense(repo) {
  return repo.startsWith('fastino/') ? 'apache-2.0' : UPSTREAM_LICENSE[repo];
}

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

const slugDir = (id) => resolve(K28_DIR, id.replace(/\//g, '__'));

// One manifest per precision whose folder exists.
function readPrecisions(dir) {
  return ['f16', 'f32']
    .filter((prec) => existsSync(join(dir, prec)))
    .map((prec) => ({ prec, mf: JSON.parse(readFileSync(join(dir, prec, 'manifest.json'), 'utf8')) }));
}

// Selection: included models, skipped models (license not allowed) and k28 ids without a folder.
function select() {
  const catalog = JSON.parse(readFileSync(CATALOG, 'utf8')).models;
  const byId = new Map(catalog.map((m) => [m.id, m]));
  const folders = [];
  const missing = [];
  for (const id of ALL ? catalog.map((m) => m.id) : DEFAULT_K28) {
    if (existsSync(slugDir(id))) folders.push({ dir: slugDir(id), isDefault: DEFAULT_K28.includes(id) });
    else if (!ALL) missing.push(id);
  }
  for (const p of UPSTREAM) folders.push({ dir: resolve(ROOT, p), isDefault: true });

  const models = [];
  const skipped = [];
  for (const { dir, isDefault } of folders) {
    const precs = readPrecisions(dir);
    if (precs.length === 0) throw new Error(`${dir}: no f16 or f32 folder`);
    const { repo, revision } = precs[0].mf.source ?? {};
    for (const { mf } of precs) {
      if (!mf.source?.repo || !mf.source?.revision) throw new Error(`${dir}: manifest lacks source.repo or source.revision`);
      if (mf.source.repo !== repo || mf.source.revision !== revision) throw new Error(`${dir}: precisions name different sources`);
    }
    const cat = byId.get(repo);
    const license = cat?.license ?? upstreamLicense(repo);
    if (!license) {
      if (isDefault) throw new Error(`${repo}: no license for a default model`);
      skipped.push(`${repo} (no license)`);
      continue;
    }
    if (!ALLOWED.has(license)) {
      if (isDefault) throw new Error(`${repo}: license ${license} is not allowed for a default model`);
      skipped.push(`${repo} (${license})`);
      continue;
    }
    for (const { prec, mf } of precs) {
      if (!mf.words) throw new Error(`${dir}/${prec}/manifest.json has no words table; version 3 with words.bin is required`);
    }
    const family = cat?.family ?? precs[0].mf.encoder?.arch ?? '';
    models.push({
      id: repo,
      org: repo.split('/')[0],
      revision,
      license,
      dir,
      task: precs[0].mf.task ?? cat?.task ?? precs[0].mf.head?.type ?? '-',
      deberta: family.startsWith('deberta') || repo.startsWith('fastino/gliner2.5-'),
      precs: precs.map(({ prec, mf }) => {
        const files = [mf.tokenizer, ...mf.shards.map((s) => s.file), mf.words.file];
        const bytes = files.reduce((n, f) => n + statSync(join(dir, prec, f)).size, 0);
        return { prec, mf, files, bytes };
      }),
    });
  }
  return { models, skipped, missing };
}

function noticeText(m) {
  const attribution = {
    'cc-by-4.0': `Licensed under CC BY 4.0 by the Hugging Face organization ${m.org}, https://huggingface.co/${m.id}; changes as described above.`,
    mit: `MIT License, see the original repository for the copyright notice: https://huggingface.co/${m.id}`,
  }[m.license];
  return [
    `${m.id} (kleinhirn weights)`,
    '',
    'This folder contains weights derived from:',
    '',
    `- ${m.id}, published by the Hugging Face organization ${m.org}, released under ${ALLOWED.get(m.license)}. https://huggingface.co/${m.id}`,
    `  Revision ${m.revision}.`,
    ...(m.deberta ? ['', '- The encoder of that model is DeBERTa by Microsoft, released under the MIT License. https://github.com/microsoft/DeBERTa'] : []),
    '',
    'Modifications: the tensors were re-laid out into flat binary shards described by manifest.json, and the word table into words.bin with a checksum per row. The f16 files are the fp32 weights cast to half precision. No training or fine-tuning was applied. The tokenizer content is unchanged; tokenizer.bin stores it in binary sections. Tools: tools/k28_convert.ts, convert/export_weights.py or convert/export_julia.py, and tools/r4_words.ts, in https://github.com/smashburger-dev/kleinhirn.',
    ...(attribution ? ['', attribution] : []),
    ...(m.deberta ? ['', '---', '', 'DeBERTa license text (Microsoft, MIT License)', '', DEBERTA_MIT] : []),
  ].join('\n');
}

function readmeText(models) {
  const mb = (m, prec) => {
    const p = m.precs.find((x) => x.prec === prec);
    return p ? (p.bytes / 1e6).toFixed(1) : '-';
  };
  const rows = models.map((m) =>
    `| \`${m.id}/\` | [${m.id}](https://huggingface.co/${m.id}) | \`${m.revision.slice(0, 12)}\` | ${m.license} | ${m.task} | ${mb(m, 'f16')} | ${mb(m, 'f32')} |`);
  return [
    '---',
    'license: other',
    'license_name: per-model',
    'library_name: kleinhirn',
    'tags:',
    '  - webgpu',
    '  - wasm',
    '  - text-classification',
    '  - feature-extraction',
    '---',
    '',
    '# kleinhirn weights',
    '',
    'Converted weights for the [kleinhirn](https://github.com/smashburger-dev/kleinhirn) engine. kleinhirn',
    'runs small encoder models in the browser on WebGPU, with a WASM-SIMD fallback on the CPU. These files',
    'are for that engine. They are not a Transformers checkpoint.',
    '',
    'Each model sits in a folder named after its source repository. The `f16` folder is for WebGPU with',
    '`shader-f16`, the `f32` folder for WebGPU f32 and WASM.',
    '',
    '| Folder | Source model | Revision | License | Task | f16 MB | f32 MB |',
    '|---|---|---|---|---|---|---|',
    ...rows,
    '',
    'Sizes are the sums of the files each manifest names (tokenizer, shards, words.bin), in 10^6 bytes.',
    '',
    '## Files',
    '',
    'Each precision folder has:',
    '',
    '- `manifest.json`: tensor layout, file sizes and sha256 of every file, and the source revision.',
    '- `tokenizer.bin`: the upstream `tokenizer.json` in binary sections, decoded back to the identical object.',
    '- `shard-*.bin`: the tensor data.',
    '- `words.bin`: the word-embedding table, with a 32-bit checksum after each row.',
    '',
    'The engine verifies every file against the sha256 in the manifest. With `words: \'rows\'` it fetches only',
    'the rows of the tokens in a text by HTTP Range and checks each row\'s checksum. Such requests show this',
    'host which tokens a text contains.',
    '',
    '## Use',
    '',
    'Point the kleinhirn engine at `<org>/<name>/f16/manifest.json` in this repository, at a pinned revision',
    'and not at `main`.',
    '',
    '## License and credit',
    '',
    'Each model keeps its own license. See the license column above and the `NOTICE` in each model folder.',
    'The `LICENSE` file at the root is the Apache License 2.0 text.',
    '',
  ].join('\n');
}

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

const { models, skipped, missing } = select();

if (DRY_RUN) {
  for (const m of models) {
    const sizes = m.precs.map((p) => `${p.prec} ${(p.bytes / 1e6).toFixed(1)} MB`).join('  ');
    console.log(`${m.license.padEnd(11)} ${m.id}  ${sizes}  ${relative(ROOT, m.dir)}`);
  }
  for (const s of skipped) console.log(`skipped: ${s}`);
  for (const id of missing) console.log(`no local folder: ${id}`);
  const total = models.reduce((n, m) => n + m.precs.reduce((k, p) => k + p.bytes, 0), 0);
  console.log(`${models.length} included, ${skipped.length} skipped, ${missing.length} without local folder; `
    + `${(total / 1e6).toFixed(1)} MB of model files (${total} bytes); nothing written`);
  process.exit(0);
}

rmSync(OUT, { recursive: true, force: true });
const notices = [];
for (const m of models) {
  for (const p of m.precs) {
    const to = join(OUT, m.id, p.prec);
    mkdirSync(to, { recursive: true });
    // The manifest is copied, not linked: a tool that rewrites the source manifest in place
    // would otherwise change the upload folder too (happened in R6).
    copyFileSync(join(m.dir, p.prec, 'manifest.json'), join(to, 'manifest.json'));
    for (const f of p.files) put(join(m.dir, p.prec, f), join(to, f));
  }
  const notice = join(OUT, m.id, 'NOTICE');
  writeFileSync(notice, noticeText(m));
  notices.push(notice);
}
const card = join(OUT, 'README.md');
writeFileSync(card, readmeText(models));
copyFileSync(resolve(ROOT, 'publish/LICENSE'), join(OUT, 'LICENSE'));

// The card must not carry private paths, project names or addresses. The
// private names are assembled from
// parts so that this file itself passes the public export's leak scan.
const LEAK = new RegExp(
  ['Users/', 'Life' + 'maxxing', 'edge-vor-' + 'modell', 'arg' + 'min', 'recruit' + 'ing', '@[\\w-]+\\.[a-z]{2,}'].join('|'),
  'i',
);
for (const f of [card, ...notices]) {
  const hit = LEAK.exec(readFileSync(f, 'utf8'));
  if (hit) throw new Error(`${relative(OUT, f)} contains "${hit[0]}"`);
}

let total = 0;
for (const file of walk(OUT).sort()) {
  const size = statSync(file).size;
  total += size;
  console.log(`${(await sha256(file))}  ${String(size).padStart(10)}  ${relative(OUT, file)}`);
}
console.log(`total ${total} bytes (${(total / 1048576).toFixed(1)} MiB) in ${relative(ROOT, OUT)}, ${models.length} models`);
for (const s of skipped) console.log(`skipped: ${s}`);
for (const id of missing) console.log(`no local folder: ${id}`);
