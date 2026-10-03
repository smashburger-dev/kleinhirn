// Checkpoint tensor names -> canonical manifest names: the post-norm rows with
// absolute positions (BERT, electra, MiniLM, RoBERTa, XLM-R, DistilBERT), DeBERTa-v2 and v3
// (relative attention, or absolute positions as polyBERT) and ModernBERT.
// Canonical names follow DEBERTA_TENSORS and MODERNBERT_TENSORS in src/plan/spec.ts.
// Weights stay [out, in] as in PyTorch; nothing transposes.
// The plan lists every canonical tensor the description needs and every
// checkpoint tensor it consumes; a checkpoint tensor that is neither used nor
// known to be unused is an error, and so is a missing required tensor.

import type { HeadSpec, ModelSpec } from '../plan/spec.ts';

export interface CanonicalTensor {
  name: string;
  sources: string[]; // checkpoint names, concatenated along rows in this order
  shape: number[];   // expected shape of the (fused) tensor
  // Sources live in this additional file (Dense modules) instead of the checkpoint.
  file?: string;
  // No source: the converter computes the tensor (sinusoidal position table).
  generate?: 'sinusoidal';
  // DeBERTa relative attention (share_att_key): LayerNorm of the relative embeddings, then the
  // key or query projection of one layer. The converter computes it in float64 (gatherTensors).
  derive?: RelProjection;
  // The source is a Conv1d weight [out, in, k]; the stored tensor is [out, k * in] with column
  // t * in + ci = W[c, ci, t], so the matmul kernel can apply it to im2col rows.
  transform?: 'conv1d';
}

export interface RelProjection {
  rel: string;       // checkpoint name of rel_embeddings.weight
  relNormW: string;  // encoder.LayerNorm.weight
  relNormB: string;
  rows: number;      // the first 2 * position_buckets rows
  eps: number;
  projW: string;     // key_proj or query_proj of the layer
  projB: string;
}

export interface NamePlan {
  prefix: string;
  tensors: CanonicalTensor[];
  unused: string[];
}

// Additional tensor files by key, with their tensor names (Dense modules:
// key 'dense0' for the first Dense module of the sentence-transformers chain).
export type ExtraNames = Record<string, string[]>;

// Names (after the prefix is stripped) that may sit in a checkpoint without
// being read: MLM and NSP heads, position id buffers, an ELECTRA
// generator/discriminator head, the DistilBERT MLM head, and the pooler when
// the task does not use it (RoBERTa and BERT poolers exist in every base
// checkpoint, sequence heads of RoBERTa never read one).
const KNOWN_UNUSED = [
  /^cls\./, /(^|\.)position_ids$/, /(^|\.)embeddings\.token_type_ids$/,
  /^discriminator_predictions\./, /^generator_predictions\./, /^generator_lm_head\./,
  /^lm_head\./, /^vocab_(transform|layer_norm|projector)\./,
  // DeBERTa pretraining heads
  /^lm_predictions\./, /^mask_predictions\./,
];
// ModernBERT: the MLM head and decoder (the pooled and per-row heads that a task reads are
// consumed first, so only the rest lands here).
const MODERNBERT_UNUSED = [/^decoder\./, /^head\./, /^classifier\./];

type Row = 'bert' | 'roberta' | 'distilbert' | 'deberta' | 'modernbert';
const rowOf = (family: ModelSpec['family']): Row => {
  switch (family) {
    case 'bert': case 'electra': return 'bert';
    case 'roberta': case 'xlm-roberta': return 'roberta';
    case 'distilbert': return 'distilbert';
    case 'deberta-v2': return 'deberta';
    case 'modernbert': return 'modernbert';
    default: throw new Error(`names table covers bert, electra, roberta, xlm-roberta, distilbert, deberta-v2, modernbert, not ${family}`);
  }
};

const PREFIXES: Record<Row, string[]> = {
  bert: ['bert.', 'electra.', ''],
  roberta: ['roberta.', ''],
  distilbert: ['distilbert.', ''],
  deberta: ['deberta.', ''],
  modernbert: ['model.', ''],
};
// Checkpoint name of the word embedding table per row (after the prefix).
const WORD_TABLE: Record<Row, string> = {
  bert: 'embeddings.word_embeddings.weight', roberta: 'embeddings.word_embeddings.weight',
  distilbert: 'embeddings.word_embeddings.weight', deberta: 'embeddings.word_embeddings.weight',
  modernbert: 'embeddings.tok_embeddings.weight',
};

// The encoder prefix is decided once from the word embedding table.
export function detectPrefix(names: string[], row: Row = 'bert'): string {
  const found = PREFIXES[row].filter((p) => names.includes(`${p}${WORD_TABLE[row]}`));
  if (found.length !== 1) {
    throw new Error(`cannot detect the encoder prefix: word_embeddings.weight under [${found.join(', ')}]`);
  }
  return found[0];
}

// Old checkpoints call LayerNorm parameters gamma and beta. DistilBERT's
// layer norms are named sa_layer_norm and output_layer_norm, so the rule is
// "weight or bias of a norm", given by the caller.
function alias(name: string): string[] {
  if (!/(LayerNorm|layer_norm)\./.test(name)) return [name];
  return [name, name.replace(/\.weight$/, '.gamma').replace(/\.bias$/, '.beta')];
}

// Checkpoint-side name of a head step, per row. The pooler and the DistilBERT
// encoder live under the encoder prefix, the classifier never does.
function headSource(row: Row, step: string): { src: string; encoder: boolean } {
  switch (step) {
    // BERT keeps the pooler inside the model, DeBERTa's ContextPooler sits next to the classifier.
    case 'head.pooler': return { src: 'pooler.dense', encoder: row !== 'deberta' };
    case 'head.classifier': return { src: 'classifier', encoder: false };
    case 'head.dense': return { src: row === 'modernbert' ? 'head.dense' : 'classifier.dense', encoder: false };
    case 'head.norm':
      if (row !== 'modernbert') break;
      return { src: 'head.norm', encoder: false };
    case 'head.out_proj': return { src: 'classifier.out_proj', encoder: false };
    case 'head.pre_classifier':
      if (row !== 'distilbert') break;
      return { src: 'pre_classifier', encoder: false };
    default: break;
  }
  throw new Error(`no checkpoint name for head step ${step} in the ${row} row`);
}

export function planNames(
  spec: ModelSpec, head: HeadSpec, checkpointNames: string[], extra: ExtraNames = {},
): NamePlan {
  const row = rowOf(spec.family);
  const prefix = detectPrefix(checkpointNames, row);
  const have = new Set(checkpointNames);
  const H = spec.hidden;
  const E = spec.embeddingSize;
  const I = spec.intermediate;
  const tensors: CanonicalTensor[] = [];
  const used = new Set<string>();

  const need = (checkpoint: string, file?: string): string => {
    const pool = file === undefined ? have : new Set(extra[file] ?? []);
    for (const cand of alias(checkpoint)) {
      if (pool.has(cand)) {
        if (file === undefined) used.add(cand);
        return cand;
      }
    }
    throw new Error(`${file ?? 'checkpoint'} lacks required tensor ${checkpoint}`);
  };
  const add = (name: string, sources: string[], shape: number[], file?: string): void => {
    const t: CanonicalTensor = { name, sources: sources.map((s) => need(s, file)), shape };
    if (file !== undefined) t.file = file;
    tensors.push(t);
  };
  const pair = (name: string, src: string, out: number, inp: number, file?: string, bias = true): void => {
    add(`${name}.weight`, [`${src}.weight`], [out, inp], file);
    if (bias) add(`${name}.bias`, [`${src}.bias`], [out], file);
  };
  const p = (s: string): string => `${prefix}${s}`;

  // ModernBERT (pre-norm): embedding norm, per layer Wqkv, Wo, MLP Wi and Wo, the norms
  // (no attn_norm in layer 0), the final norm. Biases only where the config has them.
  const addModernbert = (): void => {
    const normBias = spec.block.norm.bias;
    const norm = (canon: string, src: string): void => {
      add(`${canon}.weight`, [`${src}.weight`], [H]);
      if (normBias) add(`${canon}.bias`, [`${src}.bias`], [H]);
    };
    const lin = (canon: string, src: string, out: number, inp: number, bias: boolean): void =>
      pair(canon, src, out, inp, undefined, bias);
    norm('embeddings.norm', p('embeddings.norm'));
    for (let l = 0; l < spec.layers; l += 1) {
      const b = p(`layers.${l}`);
      if (!(spec.block.firstNormIdentity && l === 0)) norm(`layers.${l}.attn_norm`, `${b}.attn_norm`);
      lin(`layers.${l}.wqkv`, `${b}.attn.Wqkv`, 3 * H, H, spec.attention.bias);
      lin(`layers.${l}.attn_out`, `${b}.attn.Wo`, H, H, spec.attention.bias);
      norm(`layers.${l}.mlp_norm`, `${b}.mlp_norm`);
      lin(`layers.${l}.mlp_in`, `${b}.mlp.Wi`, 2 * I, H, spec.ffn.bias);
      lin(`layers.${l}.mlp_out`, `${b}.mlp.Wo`, H, I, spec.ffn.bias);
    }
    norm('final_norm', p('final_norm'));
  };

  add('embeddings.word.weight', [p(WORD_TABLE[row])], [spec.vocab, E]);
  const positionShape = [spec.embed.maxPositions, E];
  const positionName = p('embeddings.position_embeddings.weight');
  if (spec.embed.positions !== 'absolute') {
    // DeBERTa with relative attention and ModernBERT carry no position table.
  } else if (spec.embed.sinusoidal && !have.has(positionName)) {
    tensors.push({ name: 'embeddings.position.weight', sources: [], shape: positionShape,
      generate: 'sinusoidal' });
  } else {
    add('embeddings.position.weight', [positionName], positionShape);
  }
  if (spec.embed.typeVocab > 0) {
    add('embeddings.type.weight', [p('embeddings.token_type_embeddings.weight')],
      [spec.embed.typeVocab, E]);
  }
  if (row === 'modernbert') {
    addModernbert();
  } else {
    add('embeddings.LayerNorm.weight', [p('embeddings.LayerNorm.weight')], [E]);
    add('embeddings.LayerNorm.bias', [p('embeddings.LayerNorm.bias')], [E]);
  }
  if (spec.embed.project) pair('embeddings.project', p('embeddings_project'), H, E);
  if (spec.conv) {
    const k = spec.conv.kernel;
    tensors.push({ name: 'conv.weight', sources: [need(p('encoder.conv.conv.weight'))], shape: [H, k * H],
      transform: 'conv1d' });
    add('conv.bias', [p('encoder.conv.conv.bias')], [H]);
    add('conv.ln.weight', [p('encoder.conv.LayerNorm.weight')], [H]);
    add('conv.ln.bias', [p('encoder.conv.LayerNorm.bias')], [H]);
  }

  for (let l = 0; l < spec.layers && row !== 'modernbert'; l += 1) {
    if (row === 'distilbert') {
      const b = p(`transformer.layer.${l}`);
      add(`layers.${l}.qkv.weight`,
        ['q_lin', 'k_lin', 'v_lin'].map((k) => `${b}.attention.${k}.weight`), [3 * H, H]);
      add(`layers.${l}.qkv.bias`,
        ['q_lin', 'k_lin', 'v_lin'].map((k) => `${b}.attention.${k}.bias`), [3 * H]);
      pair(`layers.${l}.attn_out`, `${b}.attention.out_lin`, H, H);
      for (const [c, src] of [['attn_ln', `${b}.sa_layer_norm`],
        ['ffn_ln', `${b}.output_layer_norm`]] as const) {
        add(`layers.${l}.${c}.weight`, [`${src}.weight`], [H]);
        add(`layers.${l}.${c}.bias`, [`${src}.bias`], [H]);
      }
      pair(`layers.${l}.ffn_in`, `${b}.ffn.lin1`, I, H);
      pair(`layers.${l}.ffn_out`, `${b}.ffn.lin2`, H, I);
      continue;
    }
    const b = p(`encoder.layer.${l}`);
    const self = `${b}.attention.self`;
    // DeBERTa calls the projections query_proj, key_proj and value_proj.
    const qkv = row === 'deberta' ? ['query_proj', 'key_proj', 'value_proj'] : ['query', 'key', 'value'];
    add(`layers.${l}.qkv.weight`, qkv.map((k) => `${self}.${k}.weight`), [3 * H, H]);
    add(`layers.${l}.qkv.bias`, qkv.map((k) => `${self}.${k}.bias`), [3 * H]);
    pair(`layers.${l}.attn_out`, `${b}.attention.output.dense`, H, H);
    for (const [c, src] of [['attn_ln', `${b}.attention.output.LayerNorm`],
      ['ffn_ln', `${b}.output.LayerNorm`]] as const) {
      add(`layers.${l}.${c}.weight`, [`${src}.weight`], [H]);
      add(`layers.${l}.${c}.bias`, [`${src}.bias`], [H]);
    }
    pair(`layers.${l}.ffn_in`, `${b}.intermediate.dense`, I, H);
    pair(`layers.${l}.ffn_out`, `${b}.output.dense`, H, I);
    if (spec.attention.kind === 'deberta-relative') {
      const rel = spec.attention.rel!;
      const derive = (proj: 'key' | 'query'): RelProjection => ({
        rel: need(p('encoder.rel_embeddings.weight')),
        relNormW: need(p('encoder.LayerNorm.weight')), relNormB: need(p('encoder.LayerNorm.bias')),
        rows: 2 * rel.buckets, eps: spec.block.norm.eps,
        projW: need(`${self}.${proj}_proj.weight`), projB: need(`${self}.${proj}_proj.bias`),
      });
      tensors.push({ name: `layers.${l}.pos_key`, sources: [], shape: [2 * rel.buckets, H], derive: derive('key') });
      tensors.push({ name: `layers.${l}.pos_query`, sources: [], shape: [2 * rel.buckets, H], derive: derive('query') });
    }
  }

  if (head.type === 'classify' || head.type === 'token') {
    for (const step of head.steps) {
      const { src, encoder } = headSource(row, step.name);
      if (step.op === 'norm') {
        add(`${step.name}.weight`, [`${src}.weight`], [H]);
        if (step.bias) add(`${step.name}.bias`, [`${src}.bias`], [H]);
        continue;
      }
      pair(step.name, encoder ? p(src) : src, step.out, step.in, undefined, step.bias);
    }
  } else if (head.type === 'embed') {
    head.steps.forEach((step, i) => {
      if (step.op !== 'dense') throw new Error(`head step ${step.name}: only dense steps are supported`);
      // A sentence-transformers Dense module: linear.weight and linear.bias.
      const key = `dense${i}`;
      if (step.name !== `head.${key}`) throw new Error(`Dense step ${step.name} is not head.${key}`);
      pair(step.name, 'linear', step.out, step.in, key, step.bias);
    });
  } else {
    throw new Error(`names table has no head type ${head.type}`);
  }

  const unused: string[] = [];
  for (const name of checkpointNames) {
    if (used.has(name)) continue;
    const bare = prefix && name.startsWith(prefix) ? name.slice(prefix.length) : name;
    const isPooler = /^pooler\./.test(bare);
    if (KNOWN_UNUSED.some((re) => re.test(bare)) || isPooler
      || (row === 'modernbert' && MODERNBERT_UNUSED.some((re) => re.test(name)))) {
      unused.push(name);
      continue;
    }
    throw new Error(`checkpoint tensor ${name} is neither used nor known to be unused (prefix '${prefix}')`);
  }
  return { prefix, tensors, unused };
}
