// Hugging Face config.json -> ModelSpec + HeadSpec (K28 design section 2).
// Covered rows: BERT (bert, electra, MiniLM are model_type bert), RoBERTa and
// XLM-R (model_type roberta, xlm-roberta), DistilBERT, DeBERTa-v2 and v3
// (model_type deberta-v2), ModernBERT. Every other family and every value
// outside the plan throws an error that names the config key. Runs under Node
// and in the browser: no imports from node:, only erasable TypeScript syntax.

import type { Act, ClassifyProblem, HeadSpec, HeadStep, ModelSpec } from './spec.ts';

export type Task =
  | 'sequence-classification' | 'nli' | 'reranking' | 'token-classification' | 'embeddings';

// `sentenceTransformers` mirrors the field of data/k28/models.json: the
// module list, the parsed 1_Pooling/config.json, the Normalize flag and the
// Dense module configs (false when there are none).
export interface HfExtras {
  task: Task;
  // The single template of tokenizer.json in order: the id of a special token, null for the text.
  // RoBERTa and XLM-R need it for the position offset (templateOfTokenizer builds it).
  template?: Array<number | null>;
  sentenceTransformers?: {
    modules?: string[];
    pooling?: Record<string, unknown>;
    normalize?: boolean;
    dense?: boolean | { in_features: number; out_features: number; bias: boolean; activation_function: string }[];
    maxSeqLength?: number;
  };
}

const ACTS: Act[] = ['gelu', 'relu', 'tanh', 'silu'];
const ST_ACTS: Record<string, Act | 'none'> = {
  'torch.nn.modules.activation.Tanh': 'tanh',
  'torch.nn.modules.activation.ReLU': 'relu',
  'torch.nn.modules.activation.GELU': 'gelu',
  'torch.nn.modules.activation.SiLU': 'silu',
  'torch.nn.modules.linear.Identity': 'none',
};

function fail(key: string, why: string): never {
  throw new Error(`config key '${key}': ${why}`);
}

function num(config: Record<string, unknown>, key: string, fallback?: number): number {
  const v = config[key] ?? fallback;
  if (typeof v !== 'number' || !Number.isFinite(v)) fail(key, `expected a number, got ${String(v)}`);
  return v;
}

function poolFromSt(pooling: Record<string, unknown> | undefined): 'mean' | 'cls' | 'max' {
  if (!pooling) fail('pooling', 'embeddings need the 1_Pooling/config.json content');
  const modes = Object.keys(pooling).filter((k) => k.startsWith('pooling_mode_') && pooling[k] === true);
  if (modes.length !== 1) fail('pooling_mode', `exactly one mode expected, got [${modes.join(', ')}]`);
  switch (modes[0]) {
    case 'pooling_mode_mean_tokens': return 'mean';
    case 'pooling_mode_cls_token': return 'cls';
    case 'pooling_mode_max_tokens': return 'max';
    default: return fail(modes[0], 'pooling mode is not supported');
  }
}

function denseSteps(
  extras: HfExtras, hidden: number,
): HeadStep[] {
  const dense = extras.sentenceTransformers?.dense;
  if (dense === true) fail('dense', 'the Dense module config (2_Dense/config.json) is missing');
  if (!dense) return [];
  let width = hidden;
  return dense.map((d, i) => {
    const act = ST_ACTS[d.activation_function];
    if (act === undefined) fail('activation_function', `'${d.activation_function}' is not supported`);
    if (d.in_features !== width) fail('in_features', `Dense ${i} expects ${d.in_features}, input is ${width}`);
    width = d.out_features;
    return { op: 'dense', name: `head.dense${i}`, in: d.in_features, out: d.out_features, act, bias: d.bias };
  });
}

function classCount(config: Record<string, unknown>, task: Task): number {
  const id2label = config.id2label as Record<string, string> | undefined;
  const classes = id2label ? Object.keys(id2label).length : num(config, 'num_labels', 2);
  if (task === 'reranking' && classes !== 1) fail('id2label', `reranking needs one logit, got ${classes}`);
  if (task === 'nli') {
    const entail = Object.values(id2label ?? {}).filter((n) => n.toLowerCase().startsWith('entail'));
    if (entail.length !== 1) fail('id2label', 'NLI needs exactly one label starting with "entail"');
  }
  return classes;
}

const PROBLEMS: Record<string, ClassifyProblem> = {
  single_label_classification: 'single',
  multi_label_classification: 'multi',
  regression: 'regression',
};

// problem_type of a sequence head as the `problem` field. The default of the engine (one class is a
// regression, more classes are single-label) is left out, so heads of configs without an explicit
// deviation keep their shape.
function problemOf(config: Record<string, unknown>, classes: number): { problem?: ClassifyProblem } {
  const raw = config.problem_type;
  if (raw === undefined || raw === null) return {};
  const problem = typeof raw === 'string' ? PROBLEMS[raw] : undefined;
  if (!problem) fail('problem_type', `'${String(raw)}' is outside ${Object.keys(PROBLEMS).join(', ')}`);
  return problem === (classes === 1 ? 'regression' : 'single') ? {} : { problem };
}

const FAMILIES = [
  'bert', 'electra', 'roberta', 'xlm-roberta', 'distilbert', 'deberta-v2', 'modernbert',
] as const;
type HfFamily = (typeof FAMILIES)[number];

// RoBERTa and XLM-R number positions as pad_token_id + count of tokens that are not pad_token_id.
// The first token of a text is the first piece of the single template of tokenizer.json. Normally it
// is a special token or a text token, none of them pad, so position i sits at row i + pad_token_id + 1.
// When that first token has the pad id (d0rj/e5-small-en-ru: <s> and <pad> are both 0), it counts as
// padding and gets row pad_token_id, so position i sits at row i + pad_token_id. A later special token
// with the pad id would shift every position after it, which no row of this list does: that throws.
function robertaOffset(config: Record<string, unknown>, template: Array<number | null> | undefined): number {
  const pad = num(config, 'pad_token_id', 1);
  if (!template || template.length === 0) {
    fail('tokenizer template', 'RoBERTa and XLM-R need the single template of tokenizer.json for the position offset');
  }
  if (template.slice(1).some((id) => id !== null && id === pad)) {
    fail('pad_token_id', `${pad} is the id of a later special token of the single template`);
  }
  return template[0] === pad ? pad : pad + 1;
}

// The single template of a tokenizer.json as robertaOffset reads it.
export function templateOfTokenizer(json: Record<string, unknown>): Array<number | null> {
  const post = json.post_processor as Record<string, unknown> | null;
  if (post === null || post === undefined) return [null];
  switch (post.type) {
    case 'TemplateProcessing': {
      const specials = (post.special_tokens ?? {}) as Record<string, { ids: number[] }>;
      const out: Array<number | null> = [];
      for (const item of post.single as Array<Record<string, { id: string }>>) {
        if (item.Sequence) out.push(null);
        else if (item.SpecialToken) out.push(...specials[item.SpecialToken.id].ids);
        else fail('post_processor', 'unknown template item');
      }
      return out;
    }
    case 'RobertaProcessing':
      return [(post.cls as [string, number])[1], null, (post.sep as [string, number])[1]];
    case 'ByteLevel':
      return [null];
    default:
      return fail('post_processor', `type ${String(post.type)} is not read`);
  }
}

// The shared keys of the four families with BERT-style names, read per family.
interface Dims {
  hidden: number; layers: number; heads: number; intermediate: number; act: Act;
  eps: number; typeVocab: number; positionOffset: number; sinusoidal: boolean;
}

function dimsOf(config: Record<string, unknown>, family: HfFamily, template?: Array<number | null>): Dims {
  const actKey = family === 'distilbert' ? 'activation' : 'hidden_act';
  const act = (config[actKey] ?? 'gelu') as Act;
  if (!ACTS.includes(act)) fail(actKey, `'${String(config[actKey])}' is outside ${ACTS.join(', ')}`);
  if (family === 'distilbert') {
    // LayerNorm eps is fixed to 1e-12 in transformers (no config key), and the
    // model has no type embedding.
    return {
      hidden: num(config, 'dim'), layers: num(config, 'n_layers'), heads: num(config, 'n_heads'),
      intermediate: num(config, 'hidden_dim'), act, eps: 1e-12, typeVocab: 0, positionOffset: 0,
      sinusoidal: config.sinusoidal_pos_embds === true,
    };
  }
  return {
    hidden: num(config, 'hidden_size'), layers: num(config, 'num_hidden_layers'),
    heads: num(config, 'num_attention_heads'), intermediate: num(config, 'intermediate_size'),
    act, eps: num(config, 'layer_norm_eps', 1e-12), typeVocab: num(config, 'type_vocab_size', 2),
    positionOffset: family === 'roberta' || family === 'xlm-roberta' ? robertaOffset(config, template) : 0,
    sinusoidal: false,
  };
}

export function specFromHfConfig(
  config: Record<string, unknown>, extras: HfExtras,
): { spec: ModelSpec; head: HeadSpec; task: Task } {
  const modelType = config.model_type as HfFamily;
  if (!FAMILIES.includes(modelType)) {
    fail('model_type', `'${String(config.model_type)}' is not in the covered rows (${FAMILIES.join(', ')})`);
  }
  if (modelType === 'deberta-v2') return specFromDeberta(config, extras);
  if (modelType === 'modernbert') return specFromModernbert(config, extras);
  // The bidirectional kernels cannot run a decoder or cross-attention.
  if (config.is_decoder === true && modelType !== 'distilbert') fail('is_decoder', 'true: causal attention is not planned');
  if (config.add_cross_attention === true && modelType !== 'distilbert') fail('add_cross_attention', 'true: cross-attention is not planned');
  const positionType = (config.position_embedding_type ?? 'absolute') as string;
  if (positionType !== 'absolute') fail('position_embedding_type', `'${positionType}', only 'absolute' is planned`);
  const { hidden, layers, heads, intermediate, act, eps, typeVocab, positionOffset, sinusoidal } =
    dimsOf(config, modelType, extras.template);
  const hiddenKey = modelType === 'distilbert' ? 'dim' : 'hidden_size';
  const headsKey = modelType === 'distilbert' ? 'n_heads' : 'num_attention_heads';
  if (hidden % heads !== 0) fail(headsKey, `${heads} does not divide ${hiddenKey} ${hidden}`);
  const headDim = hidden / heads;
  if (headDim > 64) fail(`${hiddenKey} / ${headsKey}`, `head dimension ${headDim} exceeds 64`);
  const embeddingSize = modelType === 'electra' ? num(config, 'embedding_size', hidden) : hidden;
  const norm = { eps, bias: true };
  const maxPositions = num(config, 'max_position_embeddings', 512);
  if (maxPositions <= positionOffset) {
    fail('max_position_embeddings', `${maxPositions} rows leave no position after the offset ${positionOffset}`);
  }

  const spec: ModelSpec = {
    family: modelType,
    hidden, layers, heads, headDim, intermediate, vocab: num(config, 'vocab_size'),
    embeddingSize,
    embed: {
      positions: 'absolute', positionOffset, maxPositions,
      ...(modelType === 'roberta' || modelType === 'xlm-roberta' ? { padId: num(config, 'pad_token_id', 1) } : {}),
      typeVocab, norm, maskMultiply: false,
      project: modelType === 'electra' && embeddingSize !== hidden,
      ...(sinusoidal ? { sinusoidal: true } : {}),
    },
    attention: { kind: 'standard', bias: true, scale: headDim ** -0.5 },
    block: { order: 'post', firstNormIdentity: false, finalNorm: false, norm },
    ffn: { kind: 'mlp', act, bias: true },
  };

  const { task } = extras;
  const dense = (name: string, i: number, o: number, a: Act | 'none'): HeadStep =>
    ({ op: 'dense', name, in: i, out: o, act: a, bias: true });
  let head: HeadSpec;
  if (task === 'embeddings') {
    head = embedHead(extras, hidden);
  } else {
    const classes = classCount(config, task);
    if (task === 'token-classification') {
      head = { type: 'token', steps: [dense('head.classifier', hidden, classes, 'none')], classes };
    } else if (modelType === 'electra') {
      head = {
        type: 'classify', pool: 'first', classes, ...problemOf(config, classes),
        steps: [dense('head.dense', hidden, hidden, 'gelu'), dense('head.out_proj', hidden, classes, 'none')],
      };
    } else if (modelType === 'roberta' || modelType === 'xlm-roberta') {
      // RobertaClassificationHead: tanh is fixed, whatever hidden_act says.
      head = {
        type: 'classify', pool: 'first', classes, ...problemOf(config, classes),
        steps: [dense('head.dense', hidden, hidden, 'tanh'), dense('head.out_proj', hidden, classes, 'none')],
      };
    } else if (modelType === 'distilbert') {
      head = {
        type: 'classify', pool: 'first', classes, ...problemOf(config, classes),
        steps: [dense('head.pre_classifier', hidden, hidden, 'relu'),
          dense('head.classifier', hidden, classes, 'none')],
      };
    } else {
      head = {
        type: 'classify', pool: 'first', classes, ...problemOf(config, classes),
        steps: [dense('head.pooler', hidden, hidden, 'tanh'), dense('head.classifier', hidden, classes, 'none')],
      };
    }
  }
  return { spec, head, task };
}

function bool(config: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const v = config[key] ?? fallback;
  if (typeof v !== 'boolean') fail(key, `expected a boolean, got ${String(v)}`);
  return v;
}

function actOf(config: Record<string, unknown>, key: string, fallback: Act): Act {
  const act = (config[key] ?? fallback) as Act;
  if (!ACTS.includes(act)) fail(key, `'${String(config[key])}' is outside ${ACTS.join(', ')}`);
  return act;
}

function embedHead(extras: HfExtras, hidden: number): HeadSpec {
  const st = extras.sentenceTransformers;
  return {
    type: 'embed', pool: poolFromSt(st?.pooling), steps: denseSteps(extras, hidden),
    normalize: st?.normalize === true,
  };
}

// DeBERTa-v2 and v3 (K28.6). Relative attention with the two position types of the engine kernel,
// or, without relative attention, absolute positions through the standard attention (polyBERT).
function specFromDeberta(
  config: Record<string, unknown>, extras: HfExtras,
): { spec: ModelSpec; head: HeadSpec; task: Task } {
  const hidden = num(config, 'hidden_size');
  const layers = num(config, 'num_hidden_layers');
  const heads = num(config, 'num_attention_heads');
  if (hidden % heads !== 0) fail('num_attention_heads', `${heads} does not divide hidden_size ${hidden}`);
  const headDim = hidden / heads;
  if (headDim > 64) fail('hidden_size / num_attention_heads', `head dimension ${headDim} exceeds 64`);
  if (config.attention_head_size !== undefined && config.attention_head_size !== headDim) {
    fail('attention_head_size', `${String(config.attention_head_size)} differs from hidden_size / num_attention_heads`);
  }
  const convKernel = num(config, 'conv_kernel_size', 0);
  if (convKernel > 0) {
    if (convKernel % 2 !== 1) fail('conv_kernel_size', `${convKernel}, only odd kernels keep the length`);
    if (num(config, 'conv_groups', 1) !== 1) fail('conv_groups', 'grouped convolution is not planned');
  }
  const embeddingSize = num(config, 'embedding_size', hidden);
  if (embeddingSize !== hidden) fail('embedding_size', `${embeddingSize} differs from hidden_size ${hidden}`);
  const act = actOf(config, 'hidden_act', 'gelu');
  const eps = num(config, 'layer_norm_eps', 1e-7);
  const norm = { eps, bias: true };
  const maxPositions = num(config, 'max_position_embeddings', 512);
  const positionBiased = bool(config, 'position_biased_input', true);
  const relative = bool(config, 'relative_attention', false);

  let attention: ModelSpec['attention'];
  if (relative) {
    if (!bool(config, 'share_att_key', false)) fail('share_att_key', 'false: separate position projections are not planned');
    const norms = String(config.norm_rel_ebd ?? 'none').toLowerCase().split('|').map((x) => x.trim());
    if (norms.length !== 1 || norms[0] !== 'layer_norm') {
      fail('norm_rel_ebd', `'${String(config.norm_rel_ebd)}', only 'layer_norm' is planned`);
    }
    const types = config.pos_att_type;
    if (!Array.isArray(types) || types.length !== 2 || !types.includes('p2c') || !types.includes('c2p')) {
      fail('pos_att_type', `${JSON.stringify(types)}, the kernel needs exactly c2p and p2c`);
    }
    const buckets = num(config, 'position_buckets', -1);
    if (buckets <= 0) fail('position_buckets', `${buckets}, relative positions without buckets are not planned`);
    const rel = num(config, 'max_relative_positions', -1);
    attention = {
      kind: 'deberta-relative', bias: true, scale: 1 / Math.sqrt(3 * headDim),
      rel: { buckets, maxPositions: rel > 0 ? rel : maxPositions, types: ['c2p', 'p2c'] },
    };
  } else {
    attention = { kind: 'standard', bias: true, scale: headDim ** -0.5 };
  }
  // The type table is bound next to the absolute positions only; without them its rows would be dropped.
  if (!positionBiased && num(config, 'type_vocab_size', 0) > 0) {
    fail('type_vocab_size', 'type embeddings without position_biased_input are not planned (the plan binds no type ids)');
  }
  if (!relative && !positionBiased) fail('position_biased_input', 'false without relative attention leaves no position signal');

  const spec: ModelSpec = {
    family: 'deberta-v2', hidden, layers, heads, headDim,
    intermediate: num(config, 'intermediate_size'), vocab: num(config, 'vocab_size'), embeddingSize,
    embed: {
      positions: positionBiased ? 'absolute' : 'none', positionOffset: 0, maxPositions,
      typeVocab: num(config, 'type_vocab_size', 0), norm,
      // DeBERTa multiplies the embeddings by the mask after the LayerNorm.
      maskMultiply: true, project: false,
    },
    attention,
    block: { order: 'post', firstNormIdentity: false, finalNorm: false, norm },
    ffn: { kind: 'mlp', act, bias: true },
    ...(convKernel > 0 ? { conv: { kernel: convKernel, act: actOf(config, 'conv_act', 'tanh') } } : {}),
  };

  const { task } = extras;
  if (task === 'embeddings') return { spec, head: embedHead(extras, hidden), task };
  const classes = classCount(config, task);
  const dense = (name: string, i: number, o: number, a: Act | 'none'): HeadStep =>
    ({ op: 'dense', name, in: i, out: o, act: a, bias: true });
  if (task === 'token-classification') {
    return { spec, head: { type: 'token', steps: [dense('head.classifier', hidden, classes, 'none')], classes }, task };
  }
  // ContextPooler: first token, dense, pooler_hidden_act; then the classifier.
  const poolerSize = num(config, 'pooler_hidden_size', 768);
  if (poolerSize !== hidden) fail('pooler_hidden_size', `${poolerSize} differs from hidden_size ${hidden}`);
  const poolerAct = actOf(config, 'pooler_hidden_act', 'gelu');
  return {
    spec, task,
    head: {
      type: 'classify', pool: 'first', classes, ...problemOf(config, classes),
      steps: [dense('head.pooler', hidden, hidden, poolerAct), dense('head.classifier', hidden, classes, 'none')],
    },
  };
}

// RoPE thetas of transformers 5.0.0: rope_parameters per layer type, else the old keys
// global_rope_theta and local_rope_theta, else 160000 and 10000.
function modernbertThetas(config: Record<string, unknown>): { global: number; local: number } {
  if (config.rope_scaling !== undefined && config.rope_scaling !== null) {
    fail('rope_scaling', 'scaled RoPE is not planned');
  }
  const params = config.rope_parameters as Record<string, Record<string, unknown>> | undefined | null;
  if (params && !params.full_attention && !params.sliding_attention) {
    fail('rope_parameters', 'expected an entry per layer type (full_attention, sliding_attention)');
  }
  const theta = (type: string, legacy: string, fallback: number): number => {
    const entry = params?.[type];
    if (entry) {
      const ropeType = entry.rope_type ?? 'default';
      if (ropeType !== 'default') fail(`rope_parameters.${type}.rope_type`, `'${String(ropeType)}', only 'default' is planned`);
      const extra = Object.keys(entry).filter((k) => k !== 'rope_type' && k !== 'rope_theta');
      if (extra.length) fail(`rope_parameters.${type}`, `key ${extra[0]} is not planned`);
      if (entry.rope_theta !== undefined) return num(entry, 'rope_theta');
    }
    return num(config, legacy, fallback);
  };
  return {
    global: theta('full_attention', 'global_rope_theta', 160_000),
    local: theta('sliding_attention', 'local_rope_theta', 10_000),
  };
}

// ModernBERT (K28.6): pre-norm, RoPE, GeGLU, alternating global and local layers.
function specFromModernbert(
  config: Record<string, unknown>, extras: HfExtras,
): { spec: ModelSpec; head: HeadSpec; task: Task } {
  const hidden = num(config, 'hidden_size');
  const layers = num(config, 'num_hidden_layers');
  const heads = num(config, 'num_attention_heads');
  if (hidden % heads !== 0) fail('num_attention_heads', `${heads} does not divide hidden_size ${hidden}`);
  const headDim = hidden / heads;
  if (headDim > 64) fail('hidden_size / num_attention_heads', `head dimension ${headDim} exceeds 64`);
  if (headDim % 2 !== 0) fail('hidden_size / num_attention_heads', `head dimension ${headDim} is odd, RoPE rotates pairs`);
  if (config.is_causal === true) fail('is_causal', 'causal attention is not planned');
  if (config.causal_mask === true) fail('causal_mask', 'causal attention is not planned');
  const mlpAct = (config.hidden_activation ?? 'gelu') as string;
  if (mlpAct !== 'gelu') fail('hidden_activation', `'${mlpAct}', the GeGLU kernel computes gelu`);
  const eps = num(config, 'norm_eps', 1e-5);
  const normBias = bool(config, 'norm_bias', false);
  const attentionBias = bool(config, 'attention_bias', false);
  const mlpBias = bool(config, 'mlp_bias', false);
  const norm = { eps, bias: normBias };

  const every = num(config, 'global_attn_every_n_layers', 3);
  if (every < 1) fail('global_attn_every_n_layers', `${every}`);
  const local = num(config, 'local_attention', 128);
  const types = config.layer_types;
  if (types !== undefined && types !== null) {
    const want = Array.from({ length: layers }, (_, i) => (i % every ? 'sliding_attention' : 'full_attention'));
    if (!Array.isArray(types) || types.length !== layers || types.some((t, i) => t !== want[i])) {
      fail('layer_types', 'differs from the pattern of global_attn_every_n_layers');
    }
  }
  const thetas = modernbertThetas(config);

  const spec: ModelSpec = {
    family: 'modernbert', hidden, layers, heads, headDim,
    intermediate: num(config, 'intermediate_size'), vocab: num(config, 'vocab_size'),
    embeddingSize: hidden,
    embed: {
      positions: 'none', positionOffset: 0, maxPositions: num(config, 'max_position_embeddings', 8192),
      typeVocab: 0, norm, maskMultiply: false, project: false,
    },
    attention: {
      kind: 'standard', bias: attentionBias, scale: headDim ** -0.5,
      rope: { thetaGlobal: thetas.global, thetaLocal: thetas.local },
      window: { half: Math.floor(local / 2), globalEvery: every },
    },
    block: { order: 'pre', firstNormIdentity: true, finalNorm: true, norm },
    ffn: { kind: 'geglu', act: 'gelu', bias: mlpBias },
  };

  const { task } = extras;
  if (task === 'embeddings') return { spec, head: embedHead(extras, hidden), task };
  const classes = classCount(config, task);
  // ModernBertPredictionHead (dense, activation, LayerNorm) then the classifier, which always has a bias.
  const steps: HeadStep[] = [
    { op: 'dense', name: 'head.dense', in: hidden, out: hidden,
      act: actOf(config, 'classifier_activation', 'gelu'), bias: bool(config, 'classifier_bias', false) },
    { op: 'norm', name: 'head.norm', eps, bias: normBias },
    { op: 'dense', name: 'head.classifier', in: hidden, out: classes, act: 'none', bias: true },
  ];
  if (task === 'token-classification') return { spec, head: { type: 'token', steps, classes }, task };
  const pooling = (config.classifier_pooling ?? 'cls') as string;
  if (pooling !== 'cls' && pooling !== 'mean') fail('classifier_pooling', `'${pooling}', expected cls or mean`);
  return { spec, head: { type: 'classify', pool: pooling === 'cls' ? 'first' : 'mean', steps, classes, ...problemOf(config, classes) }, task };
}

// Longest token sequence the model takes: the position rows after the offset,
// capped by the tokenizer's model_max_length (ignored when it is the
// "unlimited" placeholder, 100000 or more) and, for embeddings, by the
// max_seq_length of sentence-transformers.
export function manifestMaxLength(
  spec: ModelSpec, modelMaxLength?: number | null, maxSeqLength?: number | null,
): number {
  let max = spec.embed.maxPositions - spec.embed.positionOffset;
  if (typeof modelMaxLength === 'number' && modelMaxLength < 100_000) max = Math.min(max, modelMaxLength);
  if (typeof maxSeqLength === 'number' && maxSeqLength > 0) max = Math.min(max, maxSeqLength);
  return max;
}
