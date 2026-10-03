// K28 converter CLI: models/k28/<slug>/repo (safetensors) -> f32/ and f16/
// (manifest.json, weights-N.bin, tokenizer.json). The core lives in
// src/convert/ and reads no files; this CLI does the file work.
//
// Usage: node tools/k28_convert.ts <model-id>... | --pilot | --pilot-k28.5 | --pilot-k28.6
//          [--again-root DIR]   write a second copy of every output to DIR and
//                               compare sha256 (gate G4)
//          [--report FILE]      default bench/results/k28-convert-pilot.json
//                               (k28-convert-k28.5.json with --pilot-k28.5,
//                               k28-convert-k28.6.json with --pilot-k28.6)

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { manifestMaxLength, specFromHfConfig, templateOfTokenizer, type HfExtras, type Task } from '../src/plan/hf.ts';
import { F16RangeError, buildManifest, gatherTensors } from '../src/convert/manifest.ts';
import { planNames, type ExtraNames } from '../src/convert/names.ts';
import { parseSafetensors, type StFile } from '../src/convert/safetensors.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PILOT = [
  'daekeun-ml/koelectra-small-v3-nsmc',
  'MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33',
  'sentence-transformers/all-MiniLM-L6-v2',
  'BAAI/bge-small-en-v1.5',
  'cross-encoder/ms-marco-MiniLM-L6-v2',
  'cross-encoder/ms-marco-MiniLM-L4-v2',
  'cross-encoder/ms-marco-MiniLM-L12-v2',
  'dslim/bert-base-NER',
];

const PILOT_K285 = [
  'cardiffnlp/twitter-roberta-base-sentiment-latest',
  'cross-encoder/nli-distilroberta-base',
  'sentence-transformers/all-distilroberta-v1',
  'OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M',
  'cross-encoder/stsb-distilroberta-base',
  'qilowoq/mmarco-mMiniLMv2-L12-H384-v1-en-ru',
  'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli',
  'd0rj/e5-small-en-ru',
  'ukr-models/uk-ner',
  'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
  'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
  'typeform/distilbert-base-uncased-mnli',
  'sentence-transformers/distiluse-base-multilingual-cased-v1',
  'OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M',
  'Amdestya/ce-cat-distilbert',
  'emrecan/distilbert-base-turkish-cased-allnli_tr',
];

const PILOT_K286 = [
  'protectai/deberta-v3-base-prompt-injection-v2',
  'cross-encoder/nli-deberta-v3-small',
  'xushijie/polyBERT',
  'OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M',
  'mixedbread-ai/mxbai-rerank-xsmall-v1',
  'sheltron-ai/prompt-guard-68m',
  'Horizon-Labs/multilingual-zeroshot-small',
  'ibm-granite/granite-embedding-small-english-r2',
  'OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M',
  'hotchpotch/japanese-reranker-xsmall-v2',
  'ibm-granite/granite-embedding-reranker-english-r2',
];

interface Entry {
  id: string; task: Task; revision: string; weightFiles: string[];
  config: Record<string, unknown>; sentenceTransformers?: HfExtras['sentenceTransformers'];
}

const sha256 = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');
const slugOf = (id: string): string => id.replace('/', '__');

function weightsPath(entry: Entry, modelDir: string): string {
  const inRepo = join(modelDir, 'repo', 'model.safetensors');
  if (existsSync(inRepo)) return inRepo;
  const converted = join(modelDir, 'model.safetensors');
  if (!existsSync(converted)) {
    const py = join(ROOT, '.venv-k28/bin/python');
    const r = spawnSync(py, [join(ROOT, 'convert/k28_bin_to_safetensors.py'), entry.id], { stdio: 'inherit' });
    if (r.status !== 0) throw new Error(`${entry.id}: bin to safetensors failed`);
  }
  return converted;
}

// Sentence-Transformers Dense modules: the config and the weight file of every
// module of type ...Dense in modules.json (in order). A module with only
// pytorch_model.bin goes through convert/k28_bin_to_safetensors.py.
function denseModules(entry: Entry, modelDir: string): {
  configs: NonNullable<HfExtras['sentenceTransformers']>['dense']; files: Record<string, StFile>; names: ExtraNames;
} {
  const repo = join(modelDir, 'repo');
  const mods = JSON.parse(readFileSync(join(repo, 'modules.json'), 'utf8')) as { type: string; path: string }[];
  const configs: { in_features: number; out_features: number; bias: boolean; activation_function: string }[] = [];
  const files: Record<string, StFile> = {};
  const names: ExtraNames = {};
  for (const m of mods.filter((x) => x.type.endsWith('.Dense'))) {
    const key = `dense${configs.length}`;
    configs.push(JSON.parse(readFileSync(join(repo, m.path, 'config.json'), 'utf8')));
    let weights = join(repo, m.path, 'model.safetensors');
    if (!existsSync(weights)) {
      weights = join(modelDir, `${key}.safetensors`);
      const py = join(ROOT, '.venv-k28/bin/python');
      const r = spawnSync(py, [join(ROOT, 'convert/k28_bin_to_safetensors.py'), entry.id,
        '--bin', join(repo, m.path, 'pytorch_model.bin'), '--out', weights], { stdio: 'inherit' });
      if (r.status !== 0) throw new Error(`${entry.id}: Dense bin to safetensors failed`);
    }
    files[key] = parseSafetensors(new Uint8Array(readFileSync(weights)));
    names[key] = [...files[key].tensors.keys()];
  }
  return { configs, files, names };
}

interface FileHashes { [file: string]: string }

async function convert(entry: Entry, outRoot: string): Promise<{ hashes: Record<string, FileHashes>; info: object }> {
  const modelDir = join(ROOT, 'models/k28', slugOf(entry.id));
  const repo = join(modelDir, 'repo');
  const config = JSON.parse(readFileSync(join(repo, 'config.json'), 'utf8')) as Record<string, unknown>;
  let st = entry.sentenceTransformers;
  let dense: ReturnType<typeof denseModules> | undefined;
  if (st && st.dense) {
    dense = denseModules(entry, modelDir);
    st = { ...st, dense: dense.configs };
  }
  const tokenizer = [join(repo, 'tokenizer.json'), join(modelDir, 'tokenizer.json')].find(existsSync);
  if (!tokenizer) throw new Error(`${entry.id}: no tokenizer.json (run convert/k28_golden.py first)`);
  const template = templateOfTokenizer(JSON.parse(readFileSync(tokenizer, 'utf8')) as Record<string, unknown>);
  const { spec, head, task } = specFromHfConfig(config, { task: entry.task, sentenceTransformers: st, template });
  const tokCfgPath = join(repo, 'tokenizer_config.json');
  const modelMaxLength = existsSync(tokCfgPath)
    ? (JSON.parse(readFileSync(tokCfgPath, 'utf8')) as { model_max_length?: number }).model_max_length : undefined;
  const maxLength = manifestMaxLength(spec, modelMaxLength, st?.maxSeqLength);
  const weights = weightsPath(entry, modelDir);
  const wbytes = new Uint8Array(readFileSync(weights));
  const ckpt = parseSafetensors(wbytes);
  const plan = planNames(spec, head, [...ckpt.tensors.keys()], dense?.names);
  const tensors = gatherTensors(ckpt, plan, dense?.files);
  const meta = {
    source: { repo: entry.id, revision: entry.revision, checkpointSha256: sha256(wbytes) },
    spec, head, task, labels: config.id2label as Record<string, string>,
    tokenizer: 'tokenizer.json', maxLength, sentenceTransformers: st,
  };
  const hashes: Record<string, FileHashes> = {};
  let f16Skipped: string | undefined;
  for (const dtype of ['f32', 'f16'] as const) {
    let built: Awaited<ReturnType<typeof buildManifest>>;
    try {
      built = await buildManifest(tensors, dtype, meta);
    } catch (e) {
      // A finite weight that overflows f16: the model is f32 only, nothing is written for f16.
      if (dtype !== 'f16' || !(e instanceof F16RangeError)) throw e;
      f16Skipped = e.message;
      console.log(`${entry.id}: f16 manifest skipped, writing f32 only (${e.message})`);
      continue;
    }
    const dir = join(outRoot, slugOf(entry.id), dtype);
    mkdirSync(dir, { recursive: true });
    const text = `${JSON.stringify(built.manifest, null, 1)}\n`;
    writeFileSync(join(dir, 'manifest.json'), text);
    copyFileSync(tokenizer, join(dir, 'tokenizer.json'));
    const h: FileHashes = { 'manifest.json': sha256(text), 'tokenizer.json': sha256(readFileSync(tokenizer)) };
    for (const shard of built.shards) {
      writeFileSync(join(dir, shard.file), shard.bytes);
      h[shard.file] = sha256(shard.bytes);
    }
    hashes[dtype] = h;
  }
  return {
    hashes,
    info: {
      prefix: plan.prefix, tensors: plan.tensors.length, unusedCheckpointTensors: plan.unused,
      checkpointDtypes: [...new Set([...ckpt.tensors.values()].map((t) => t.dtype))], maxLength,
      generatedTensors: plan.tensors.filter((t) => t.generate).map((t) => t.name),
      ...(f16Skipped ? { f16Skipped } : {}),
    },
  };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const opt = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    if (i < 0) return undefined;
    const [, v] = args.splice(i, 2);
    return v;
  };
  const againRoot = opt('--again-root');
  const k285 = args.includes('--pilot-k28.5');
  const k286 = args.includes('--pilot-k28.6');
  const report = opt('--report') ?? join(ROOT, k286 ? 'bench/results/k28-convert-k28.6.json'
    : k285 ? 'bench/results/k28-convert-k28.5.json' : 'bench/results/k28-convert-pilot.json');
  const pilot = args.includes('--pilot');
  const ids = [...(pilot ? PILOT : []), ...(k285 ? PILOT_K285 : []), ...(k286 ? PILOT_K286 : []),
    ...args.filter((a) => !a.startsWith('--'))];
  if (ids.length === 0) throw new Error('give model ids or --pilot');
  const models = (JSON.parse(readFileSync(join(ROOT, 'data/k28/models.json'), 'utf8')) as { models: Entry[] }).models;
  const rows = [];
  let files = 0;
  let differing = 0;
  for (const id of ids) {
    const entry = models.find((m) => m.id === id);
    if (!entry) throw new Error(`${id} is not in data/k28/models.json`);
    const first = await convert(entry, join(ROOT, 'models/k28'));
    const row: Record<string, unknown> = { id, ...first.info, hashes: first.hashes };
    if (againRoot) {
      const second = await convert(entry, againRoot);
      const bad: string[] = [];
      for (const dtype of Object.keys(first.hashes)) {
        for (const [f, h] of Object.entries(first.hashes[dtype])) {
          files += 1;
          if (second.hashes[dtype][f] !== h) bad.push(`${dtype}/${f}`);
        }
      }
      differing += bad.length;
      row.differing = bad;
    }
    rows.push(row);
    console.log(JSON.stringify({ ...row, hashes: undefined }));
  }
  const out: Record<string, unknown> = { rows };
  if (againRoot) Object.assign(out, { filesCompared: files, filesDiffering: differing });
  writeFileSync(report, `${JSON.stringify(out, null, 1)}\n`);
  if (differing) process.exit(1);
}

await main();
