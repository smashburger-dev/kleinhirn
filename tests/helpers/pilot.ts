// The K28 pilot models (eight BERT-row models of K28.2, sixteen RoBERTa, XLM-R
// and DistilBERT models of K28.5): ModelSpec and HeadSpec from the frozen
// configs in data/k28/models.json.

import { readFileSync } from 'node:fs';
import { specFromHfConfig, type HfExtras, type Task } from '../../src/plan/hf.ts';
import type { HeadSpec, ModelSpec } from '../../src/plan/spec.ts';

interface Entry {
  id: string; task: Task; config: Record<string, unknown>;
  sentenceTransformers?: HfExtras['sentenceTransformers'];
}

export const PILOT_IDS = [
  'daekeun-ml/koelectra-small-v3-nsmc',
  'MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33',
  'sentence-transformers/all-MiniLM-L6-v2',
  'BAAI/bge-small-en-v1.5',
  'cross-encoder/ms-marco-MiniLM-L4-v2',
  'cross-encoder/ms-marco-MiniLM-L6-v2',
  'cross-encoder/ms-marco-MiniLM-L12-v2',
  'dslim/bert-base-NER',
];

export const PILOT_K285_IDS = [
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

export const PILOT_K286_IDS = [
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

// 2_Dense/config.json of distiluse-base-multilingual-cased-v1 (models.json
// only records that a Dense module exists).
export const DISTILUSE_DENSE = [{
  in_features: 768, out_features: 512, bias: true,
  activation_function: 'torch.nn.modules.activation.Tanh' }];

const models = (JSON.parse(readFileSync(
  new URL('../../data/k28/models.json', import.meta.url), 'utf8')) as { models: Entry[] }).models;

export function pilotSpec(id: string): { spec: ModelSpec; head: HeadSpec; task: Task } {
  const e = models.find((m) => m.id === id);
  if (!e) throw new Error(`model ${id} not in data/k28/models.json`);
  const st = e.sentenceTransformers;
  const dense = st && st.dense === true ? { ...st, dense: DISTILUSE_DENSE } : st;
  // <s> A </s>: the single template of the RoBERTa and XLM-R tokenizers of the list
  const template = [e.config.bos_token_id as number, null, e.config.eos_token_id as number];
  return specFromHfConfig(e.config, { task: e.task, sentenceTransformers: dense, template });
}
