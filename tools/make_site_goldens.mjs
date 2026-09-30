#!/usr/bin/env node
// Builds site/public/goldens-small-upstream.json for the device page from the
// public golden files. The page and tools/check_device_result.mjs both hash
// the `sets` object (JSON.stringify of the parsed file) and compare it with
// `setsSha256`, so a result JSON names exactly the goldens it was scored on.
//
// Usage: node tools/make_site_goldens.mjs [out.json]
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const GOLDEN_DIR = resolve(ROOT, 'tests/golden/small-upstream');

const SOURCES = [
  { bucket: 'L128', file: 'texts1000_l128k16.json', take: 200 },
  { bucket: 'L256', file: 'long200_l256k16.json', take: 200 },
];
const ITEM_KEYS = [
  'title', 'seq_len', 'input_ids', 'attention_mask',
  'marker_indices', 'marker_mask', 'marker_groups', 'logits',
];

export const sha256 = (data) => createHash('sha256').update(data).digest('hex');

// Returns { sets, sourceSha256 }. Deterministic: same files, same bytes.
export function buildSets(goldenDir = GOLDEN_DIR) {
  const sets = {};
  const sourceSha256 = {};
  for (const { bucket, file, take } of SOURCES) {
    const raw = readFileSync(resolve(goldenDir, file));
    sourceSha256[file] = sha256(raw);
    const g = JSON.parse(raw.toString('utf8'));
    if (g.items.length < take) throw new Error(`${file}: only ${g.items.length} items`);
    sets[bucket] = {
      task: g.task,
      labels: g.labels,
      bucket: g.bucket,
      items: g.items.slice(0, take).map((item) => {
        const out = {};
        for (const k of ITEM_KEYS) {
          if (!(k in item)) throw new Error(`${file}: item lacks ${k}`);
          out[k] = item[k];
        }
        return out;
      }),
    };
  }
  return { sets, sourceSha256 };
}

// Hash of the `sets` object as the page recomputes it.
export const setsHash = (sets) => sha256(JSON.stringify(sets));

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const out = resolve(process.argv[2] ?? resolve(ROOT, 'site/public/goldens-small-upstream.json'));
  const { sets, sourceSha256 } = buildSets();
  const doc = {
    schema: 'kleinhirn-site-goldens/1',
    model: 'small-upstream',
    sourceSha256,
    setsSha256: setsHash(sets),
    sets,
  };
  mkdirSync(dirname(out), { recursive: true });
  const text = JSON.stringify(doc);
  writeFileSync(out, text);
  console.log(`${out}\n  bytes ${text.length}\n  sha256 ${sha256(text)}\n  setsSha256 ${doc.setsSha256}`);
  for (const [f, h] of Object.entries(sourceSha256)) console.log(`  source ${f} ${h}`);
}
