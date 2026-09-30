#!/usr/bin/env node
// Validates a kleinhirn-device-result/1 file and recomputes parity from its
// per-case logits against tests/golden/small-upstream, with the same gates as
// the page (site/gates.ts). No dependencies. Latency is not re-checked: it is
// what the device reported.
//
// Usage: node tools/check_device_result.mjs <file.json> [--add]
//   --add   copy a passing file to publish/community/ (community/ in the
//           public repository) and regenerate docs/COMMUNITY.md
import { createHash } from 'node:crypto';
import {
  copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compareLogits } from '../bench/metrics.ts';
import { gateRule, parityPasses } from '../site/gates.ts';
import { buildSets, setsHash } from './make_site_goldens.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SCHEMA_FILE = resolve(ROOT, 'site/public/device-result.schema.json');
// The private repository keeps the public documents under publish/; the
// exported public repository has them at the top level.
const PUBLISH_ROOT = existsSync(resolve(ROOT, 'publish')) ? resolve(ROOT, 'publish') : ROOT;
const COMMUNITY_DIR = resolve(PUBLISH_ROOT, 'community');
const COMMUNITY_MD = resolve(PUBLISH_ROOT, 'docs/COMMUNITY.md');
const BUCKETS = ['L128', 'L256'];

// ---- minimal JSON Schema interpreter (the subset the schema file uses) ----

const typeOf = (v) => (v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v);

export function validate(value, schema, root, path = '$', errors = []) {
  if (schema.$ref) {
    const target = schema.$ref.replace(/^#\//, '').split('/').reduce((o, k) => o[k], root);
    return validate(value, target, root, path, errors);
  }
  if ('const' in schema && value !== schema.const) errors.push(`${path}: expected ${JSON.stringify(schema.const)}`);
  if (schema.enum && !schema.enum.includes(value)) errors.push(`${path}: not one of ${JSON.stringify(schema.enum)}`);
  if (schema.type) {
    const allowed = [].concat(schema.type);
    const t = typeOf(value);
    const okType = allowed.includes(t) || (t === 'number' && allowed.includes('integer') && Number.isInteger(value));
    if (!okType) { errors.push(`${path}: expected ${allowed.join('|')}, got ${t}`); return errors; }
  }
  if (typeof value === 'string' && schema.pattern && !new RegExp(schema.pattern).test(value)) {
    errors.push(`${path}: does not match ${schema.pattern}`);
  }
  if (typeof value === 'number' && schema.minimum !== undefined && value < schema.minimum) {
    errors.push(`${path}: below minimum ${schema.minimum}`);
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${path}: fewer than ${schema.minItems} items`);
    if (schema.items) value.forEach((v, i) => validate(v, schema.items, root, `${path}[${i}]`, errors));
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    for (const k of schema.required ?? []) if (!(k in value)) errors.push(`${path}: missing ${k}`);
    for (const [k, sub] of Object.entries(schema.properties ?? {})) {
      if (k in value) validate(value[k], sub, root, `${path}.${k}`, errors);
    }
  }
  return errors;
}

// ---- parity recomputation ----

export function checkResult(result, { goldenDir } = {}) {
  const errors = [];
  const schema = JSON.parse(readFileSync(SCHEMA_FILE, 'utf8'));
  validate(result, schema, schema, '$', errors);
  const verdicts = [];
  if (errors.length) return { errors, verdicts };

  // Optional storage-age marker: a present store must carry its fields.
  const marker = result.environment.storageMarker;
  if (marker) {
    for (const store of ["opfs", "cache", "localStorage"]) {
      const m = marker[store];
      if (m.present && (m.firstSeen === null || m.visits === null || m.ageDays === null || m.visits < 1)) {
        errors.push(`environment.storageMarker.${store}: present without firstSeen, visits >= 1 and ageDays`);
      }
      if (!m.present && (m.firstSeen !== null || m.visits !== null)) {
        errors.push(`environment.storageMarker.${store}: absent but carries firstSeen or visits`);
      }
    }
    if (errors.length) return { errors, verdicts };
  }

  const { sets } = buildSets(goldenDir);
  const expected = setsHash(sets);
  if (result.goldens.sha256 !== expected) {
    errors.push(`goldens sha256 mismatch: result ${result.goldens.sha256}, tests/golden gives ${expected}`);
    return { errors, verdicts };
  }
  // limit < 200 (page param) is a diagnostic run: parity is checked on the
  // first `limit` items and the file is not a matrix row.
  const limit = result.protocol?.limit ?? null;
  const diagnostic = limit !== null && limit < 200;
  for (const stage of result.stages) {
    if (!stage.ok) {
      verdicts.push({ name: stage.name, ran: false, error: stage.error });
      continue;
    }
    if (!stage.parity || !stage.latency || !stage.load || !stage.memory) {
      errors.push(`${stage.name}: ok stage without parity, latency, load or memory`);
      continue;
    }
    const v = { name: stage.name, ran: true, rule: gateRule(stage.name), buckets: {}, pass: true, diagnostic };
    for (const key of BUCKETS) {
      const golden = diagnostic ? sets[key].items.slice(0, limit) : sets[key].items;
      const cases = stage.parity[key]?.perCaseLogits;
      if (!cases || cases.length !== golden.length) {
        errors.push(`${stage.name} ${key}: expected ${golden.length} cases, got ${cases?.length ?? 0}`);
        v.pass = false;
        continue;
      }
      const ref = [];
      const cand = [];
      let missing = 0;
      let shapeBad = false;
      cases.forEach((c, i) => {
        if (c === null) { missing += 1; return; }
        const nValid = golden[i].marker_mask.filter((m) => m > 0.5).length;
        if (c.length !== nValid) { shapeBad = true; return; }
        ref.push(golden[i].logits.slice(0, nValid));
        cand.push(c);
      });
      if (shapeBad || !ref.length) {
        errors.push(`${stage.name} ${key}: per-case logits have the wrong shape`);
        v.pass = false;
        continue;
      }
      const summary = compareLogits(ref, cand);
      const pass = missing === 0 && parityPasses(stage.name, summary);
      if (pass !== stage.parity[key].pass) {
        errors.push(`${stage.name} ${key}: file claims pass=${stage.parity[key].pass}, recomputed ${pass}`);
      }
      v.buckets[key] = { summary, pass, missing };
      if (!pass) v.pass = false;
    }
    if (v.pass !== stage.parityPass) errors.push(`${stage.name}: file claims parityPass=${stage.parityPass}, recomputed ${v.pass}`);
    verdicts.push(v);
  }
  return { errors, verdicts };
}

export function verdictLine(v) {
  if (!v.ran) return `${v.name.padEnd(4)} did not run: ${v.error}`;
  const parts = BUCKETS.map((k) => {
    const b = v.buckets[k];
    if (!b) return `${k} n/a`;
    const s = b.summary;
    return `${k} ${b.pass ? 'pass' : 'FAIL'} argmax ${(s.argmaxAgreement * 100).toFixed(1)} % `
      + `maxLogit ${s.maxAbsLogitDiff.toExponential(1)} maxProb ${s.maxAbsProbDiff.toExponential(1)}`
      + (b.missing ? ` missing ${b.missing}` : '');
  });
  return `${v.name.padEnd(4)} ${v.pass ? 'PASS' : 'FAIL'} (${v.rule}) ${parts.join(' | ')}`
    + (v.diagnostic ? ' [diagnostic run, not a matrix row]' : '');
}

// ---- community table ----

function browserOf(ua) {
  const major = (m) => m[1].split('.')[0];
  let m;
  if ((m = /Firefox\/([\d.]+)/.exec(ua))) return `Firefox ${major(m)}`;
  if ((m = /Edg\/([\d.]+)/.exec(ua))) return `Edge ${major(m)}`;
  if ((m = /(?:CriOS|Chrome)\/([\d.]+)/.exec(ua))) return `Chrome ${major(m)}`;
  if ((m = /Version\/([\d.]+).*Safari/.exec(ua))) return `Safari ${major(m)}`;
  return 'unknown';
}

const adapterOf = (env) => {
  const i = env.webgpu?.adapter?.info;
  if (!i) return env.webgpu?.present ? 'no adapter' : 'no WebGPU';
  return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ') || 'unnamed';
};

const cell = (s) => String(s).replace(/\|/g, '/').replace(/\n/g, ' ');

export function renderCommunity(entries) {
  const stages = ['f16', 'f32', 'wasm'];
  const head = ['Device', 'Browser', 'Adapter', 'Auto', 'f16', 'f32', 'wasm',
    'L128 p95 ms (f16/f32/wasm)', 'L256 p95 ms (f16/f32/wasm)', 'Date'];
  const rows = entries.map(({ result, verdicts }) => {
    const byName = Object.fromEntries(result.stages.map((s) => [s.name, s]));
    const verdictOf = (n) => {
      const v = verdicts.find((x) => x.name === n);
      if (!v) return '-';
      return v.ran ? (v.pass ? 'pass' : 'fail') : 'n/a';
    };
    const p95 = (k) => stages.map((n) => {
      const s = byName[n];
      return s?.ok ? s.latency[k].modelOnly.p95Ms.toFixed(1) : '-';
    }).join(' / ');
    return [
      result.device.model + (result.device.os ? `, ${result.device.os}` : ''),
      browserOf(result.environment.userAgent), adapterOf(result.environment),
      result.autoStage.picked ?? '-', ...stages.map(verdictOf), p95('L128'), p95('L256'),
      result.createdAt.slice(0, 10),
    ].map(cell);
  });
  return [
    '# Community results',
    '',
    'Results contributed from real devices through the device benchmark page and a GitHub issue.',
    'These numbers are separate from the official matrix, which is measured by us with a fixed protocol.',
    '',
    '- Parity (pass or fail) is verified by script: `tools/check_device_result.mjs` recomputes it from the logits in each file against the goldens.',
    '- Latency (p95, model only) is self-reported by the device and not verified.',
    '- "Auto" is the stage the engine picks on that device when nothing is forced.',
    '- `n/a` means the stage did not run on that device; `-` means it was not selected.',
    '',
    `| ${head.join(' | ')} |`,
    `|${head.map(() => '---').join('|')}|`,
    ...rows.map((r) => `| ${r.join(' | ')} |`),
    '',
  ].join('\n');
}

export function regenerateCommunity() {
  const files = existsSync(COMMUNITY_DIR) ? readdirSync(COMMUNITY_DIR).filter((f) => f.endsWith('.json')).sort() : [];
  const entries = files.map((f) => {
    const result = JSON.parse(readFileSync(resolve(COMMUNITY_DIR, f), 'utf8'));
    const { errors, verdicts } = checkResult(result);
    if (errors.length) throw new Error(`${f}: ${errors.join('; ')}`);
    return { result, verdicts, file: f };
  });
  entries.sort((a, b) => a.result.createdAt.localeCompare(b.result.createdAt));
  writeFileSync(COMMUNITY_MD, renderCommunity(entries));
  return entries.length;
}

// ---- CLI ----

function main() {
  const args = process.argv.slice(2);
  const add = args.includes('--add');
  const file = args.find((a) => !a.startsWith('--'));
  if (!file) { console.error('usage: check_device_result.mjs <file.json> [--add]'); process.exit(2); }
  const raw = readFileSync(file);
  let result;
  try { result = JSON.parse(raw.toString('utf8')); } catch (e) { console.error(`not JSON: ${e.message}`); process.exit(1); }
  const { errors, verdicts } = checkResult(result);
  for (const v of verdicts) console.log(verdictLine(v));
  for (const e of errors) console.log(`ERROR ${e}`);
  const ranFail = verdicts.some((v) => v.ran && !v.pass);
  const ok = !errors.length && !ranFail && verdicts.some((v) => v.ran);
  console.log(ok ? 'OK' : 'REJECTED');
  if (ok && add) {
    const hash8 = createHash('sha256').update(raw).digest('hex').slice(0, 8);
    const slug = result.device.model.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'device';
    const name = `${slug}-${result.createdAt.slice(0, 10)}-${hash8}.json`;
    mkdirSync(COMMUNITY_DIR, { recursive: true });
    copyFileSync(file, resolve(COMMUNITY_DIR, name));
    console.log(`added ${relative(ROOT, resolve(COMMUNITY_DIR, name))}; COMMUNITY.md rows: ${regenerateCommunity()}`);
  }
  process.exit(ok ? 0 : 1);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
