// Model description (K28 design section 2) and the two adapters that turn
// today's manifest `encoder` field into one. The plan builder depends only on
// this description; the family decides which tensor names are read.

export type Family =
  | 'bert' | 'electra' | 'roberta' | 'xlm-roberta' | 'distilbert'
  | 'deberta-v2' | 'modernbert';
export type Act = 'gelu' | 'relu' | 'tanh' | 'silu';
export type ClassifyProblem = 'single' | 'multi' | 'regression';
export interface Norm { eps: number; bias: boolean }

export interface ModelSpec {
  family: Family;
  hidden: number;
  layers: number;
  heads: number;
  headDim: number;
  intermediate: number;
  vocab: number;
  embeddingSize: number;
  embed: {
    positions: 'absolute' | 'none';
    positionOffset: number;
    // RoBERTa and XLM-R: pad_token_id. The engine numbers positions as offset + index and cannot
    // follow transformers, which counts the ids that are not the pad id (see padIdIndex).
    padId?: number;
    maxPositions: number;
    typeVocab: number;
    norm: Norm;
    maskMultiply: boolean;
    project: boolean;
    // DistilBERT with sinusoidal_pos_embds: the position table is computed, not
    // learned; the converter generates it when the checkpoint lacks it.
    sinusoidal?: boolean;
  };
  attention: {
    kind: 'standard' | 'deberta-relative';
    bias: boolean;
    // Informational. The builder computes the kernel constant with the
    // expressions of the engine before K28, because a different but
    // equivalent expression can change the last bit of the override.
    scale: number;
    rel?: { buckets: number; maxPositions: number; types: ('c2p' | 'p2c')[] };
    rope?: { thetaGlobal: number; thetaLocal: number };
    // half is the manifest's localAttention, handed to the kernel as WINDOW.
    window?: { half: number; globalEvery: number };
  };
  block: {
    order: 'post' | 'pre';
    firstNormIdentity: boolean;
    finalNorm: boolean;
    norm: Norm;
  };
  ffn: { kind: 'mlp' | 'geglu'; act: Act; bias: boolean };
  // DeBERTa-v2 convolution after layer 0 (conv_kernel_size > 0): a 1-D convolution over the
  // encoder input, activation, added to the layer 0 output, LayerNorm. One group only.
  conv?: { kernel: number; act: Act };
}

// Index of the first id the position rule of RoBERTa and XLM-R cannot number, or -1. The engine
// uses row offset + index; transformers uses pad + the count of ids that are not the pad id.
// Both agree when no id is the pad id. With positionOffset equal to the pad id (the first token
// of the template is itself a pad id, d0rj/e5-small-en-ru) they agree when id 0 is the pad id
// and no other id is. A spec without padId is not checked.
export function padIdIndex(embed: ModelSpec['embed'], ids: ArrayLike<number>): number {
  const pad = embed.padId;
  if (pad === undefined) return -1;
  const first = embed.positionOffset === pad;
  for (let i = 0; i < ids.length; i += 1) {
    if (i === 0 && first) {
      if (ids[0] !== pad) return 0;
    } else if (ids[i] === pad) return i;
  }
  return -1;
}

// One step of a pooled or per-row head. `name` is the tensor prefix in the
// manifest ('head.pooler', 'head.classifier', 'head.dense', 'head.out_proj').
export type HeadStep =
  | { op: 'dense'; name: string; in: number; out: number; act: Act | 'none'; bias: boolean }
  | { op: 'norm'; name: string; eps: number; bias: boolean };

export type HeadSpec =
  | { type: 'gliner2'; hidden: number; temperature: number; markers: number }
  | { type: 'julia'; layers: number; ffn: number; options: number }
  // sequence classification, NLI and reranking: pool, then the steps; the last
  // dense step has out = classes. problem is the output semantics of problem_type:
  // single (softmax), multi (sigmoid per logit), regression (raw). Absent means the default,
  // regression for one class and single otherwise (the config writer omits it then).
  | { type: 'classify'; pool: 'first' | 'mean'; steps: HeadStep[]; classes: number; problem?: ClassifyProblem }
  // token classification: the steps run on every row
  | { type: 'token'; steps: HeadStep[]; classes: number }
  // embeddings: pooling as in sentence-transformers, optional Dense modules
  | { type: 'embed'; pool: 'mean' | 'cls' | 'max'; steps: HeadStep[]; normalize: boolean };

// Tensor names one family reads for its encoder layers. Absent entries mean
// the tensor does not exist (bias-free layers bind the shared zero buffer).
export interface LayerTensors {
  qkvW: string; qkvB?: string;
  posKey?: string; posQuery?: string;
  attnOutW: string; attnOutB?: string;
  attnNormW?: string; attnNormB?: string;
  ffnInW: string; ffnInB?: string;
  ffnOutW: string; ffnOutB?: string;
  ffnNormW?: string; ffnNormB?: string;
}

export interface TensorNames {
  embedNormW: string;
  embedNormB?: string;
  finalNormW?: string;
  finalNormB?: string;
  layer(l: number): LayerTensors;
  // Convolution after layer 0: weight [H, kernel * H] (column t * H + ci = W[c, ci, t]), bias, LayerNorm.
  conv?: { w: string; b: string; normW: string; normB: string };
}

export const DEBERTA_TENSORS: TensorNames = {
  embedNormW: 'embeddings.LayerNorm.weight',
  embedNormB: 'embeddings.LayerNorm.bias',
  layer: (l) => ({
    qkvW: `layers.${l}.qkv.weight`, qkvB: `layers.${l}.qkv.bias`,
    posKey: `layers.${l}.pos_key`, posQuery: `layers.${l}.pos_query`,
    attnOutW: `layers.${l}.attn_out.weight`, attnOutB: `layers.${l}.attn_out.bias`,
    attnNormW: `layers.${l}.attn_ln.weight`, attnNormB: `layers.${l}.attn_ln.bias`,
    ffnInW: `layers.${l}.ffn_in.weight`, ffnInB: `layers.${l}.ffn_in.bias`,
    ffnOutW: `layers.${l}.ffn_out.weight`, ffnOutB: `layers.${l}.ffn_out.bias`,
    ffnNormW: `layers.${l}.ffn_ln.weight`, ffnNormB: `layers.${l}.ffn_ln.bias`,
  }),
};

export const DEBERTA_CONV_TENSORS = {
  w: 'conv.weight', b: 'conv.bias', normW: 'conv.ln.weight', normB: 'conv.ln.bias',
};

export const MODERNBERT_TENSORS: TensorNames = {
  embedNormW: 'embeddings.norm.weight',
  finalNormW: 'final_norm.weight',
  layer: (l) => ({
    qkvW: `layers.${l}.wqkv.weight`,
    attnOutW: `layers.${l}.attn_out.weight`,
    attnNormW: `layers.${l}.attn_norm.weight`,
    ffnInW: `layers.${l}.mlp_in.weight`,
    ffnOutW: `layers.${l}.mlp_out.weight`,
    ffnNormW: `layers.${l}.mlp_norm.weight`,
  }),
};

// ModernBERT from a Hugging Face config: the Julia names plus the bias tensors the config has
// (attention_bias, mlp_bias, norm_bias); a missing bias binds the zero buffer in the plan.
export function modernbertTensors(
  bias: { attention: boolean; mlp: boolean; norm: boolean },
): TensorNames {
  return {
    embedNormW: 'embeddings.norm.weight',
    embedNormB: bias.norm ? 'embeddings.norm.bias' : undefined,
    finalNormW: 'final_norm.weight',
    finalNormB: bias.norm ? 'final_norm.bias' : undefined,
    layer: (l) => ({
      qkvW: `layers.${l}.wqkv.weight`, qkvB: bias.attention ? `layers.${l}.wqkv.bias` : undefined,
      attnOutW: `layers.${l}.attn_out.weight`, attnOutB: bias.attention ? `layers.${l}.attn_out.bias` : undefined,
      attnNormW: `layers.${l}.attn_norm.weight`, attnNormB: bias.norm ? `layers.${l}.attn_norm.bias` : undefined,
      ffnInW: `layers.${l}.mlp_in.weight`, ffnInB: bias.mlp ? `layers.${l}.mlp_in.bias` : undefined,
      ffnOutW: `layers.${l}.mlp_out.weight`, ffnOutB: bias.mlp ? `layers.${l}.mlp_out.bias` : undefined,
      ffnNormW: `layers.${l}.mlp_norm.weight`, ffnNormB: bias.norm ? `layers.${l}.mlp_norm.bias` : undefined,
    }),
  };
}

// BERT, ELECTRA, RoBERTa, XLM-R and DistilBERT: the same names as DeBERTa without the relative-position
// projections (K28.2 converter writes the BERT row under these names).
const BERT_TENSORS: TensorNames = {
  ...DEBERTA_TENSORS,
  layer: (l) => {
    const { posKey: _k, posQuery: _q, ...rest } = DEBERTA_TENSORS.layer(l);
    return rest;
  },
};

export function tensorNamesFor(family: Family, spec?: ModelSpec): TensorNames {
  if (family === 'bert' || family === 'electra' || family === 'roberta'
    || family === 'xlm-roberta' || family === 'distilbert') return BERT_TENSORS;
  if (family === 'deberta-v2') {
    return spec?.conv ? { ...DEBERTA_TENSORS, conv: DEBERTA_CONV_TENSORS } : DEBERTA_TENSORS;
  }
  if (family === 'modernbert') {
    if (!spec) return MODERNBERT_TENSORS;
    return modernbertTensors({
      attention: spec.attention.bias, mlp: spec.ffn.bias, norm: spec.block.norm.bias });
  }
  throw new Error(`no tensor names for family ${family}`);
}

// Manifest `encoder` fields as written by convert/export_weights.py and
// convert/export_julia.py.
export interface EncoderSpec {
  hiddenSize: number;
  layers: number;
  heads: number;
  intermediateSize: number;
  positionBuckets: number;
  maxRelativePositions: number;
  layerNormEps: number;
  vocabSize?: number;
}

export interface JuliaSpec {
  layers: number;
  hiddenSize: number;
  heads: number;
  intermediate: number;
  normEps: number;
  ropeTheta: number;
  localAttention: number;
  globalEvery: number;
  headLayers: number;
  headFfn: number;
  options: number;
  maxPos?: number;
  vocab?: number;
}

export function specFromGlinerManifest(
  enc: EncoderSpec,
  head: { temperature: number; hiddenSize?: number },
  markers = 16,
): { spec: ModelSpec; head: HeadSpec } {
  const headDim = enc.hiddenSize / enc.heads;
  const norm = { eps: enc.layerNormEps, bias: true };
  return {
    spec: {
      family: 'deberta-v2',
      hidden: enc.hiddenSize, layers: enc.layers, heads: enc.heads, headDim,
      intermediate: enc.intermediateSize, vocab: enc.vocabSize ?? 0,
      embeddingSize: enc.hiddenSize,
      embed: {
        positions: 'none', positionOffset: 0, maxPositions: 0, typeVocab: 0,
        norm, maskMultiply: true, project: false,
      },
      attention: {
        kind: 'deberta-relative', bias: true, scale: 1 / Math.sqrt(3 * headDim),
        rel: {
          buckets: enc.positionBuckets, maxPositions: enc.maxRelativePositions,
          types: ['c2p', 'p2c'],
        },
      },
      block: { order: 'post', firstNormIdentity: false, finalNorm: false, norm },
      ffn: { kind: 'mlp', act: 'gelu', bias: true },
    },
    head: {
      type: 'gliner2', hidden: Number(head.hiddenSize) || 768,
      temperature: head.temperature, markers,
    },
  };
}

export function specFromJuliaManifest(
  enc: JuliaSpec,
): { spec: ModelSpec; head: HeadSpec } {
  const headDim = enc.hiddenSize / enc.heads;
  const norm = { eps: enc.normEps, bias: false };
  return {
    spec: {
      family: 'modernbert',
      hidden: enc.hiddenSize, layers: enc.layers, heads: enc.heads, headDim,
      intermediate: enc.intermediate, vocab: enc.vocab ?? 0,
      embeddingSize: enc.hiddenSize,
      embed: {
        positions: 'none', positionOffset: 0, maxPositions: enc.maxPos ?? 0,
        typeVocab: 0, norm, maskMultiply: false, project: false,
      },
      attention: {
        kind: 'standard', bias: false, scale: headDim ** -0.5,
        rope: { thetaGlobal: enc.ropeTheta, thetaLocal: enc.ropeTheta },
        window: { half: enc.localAttention, globalEvery: enc.globalEvery },
      },
      block: { order: 'pre', firstNormIdentity: true, finalNorm: true, norm },
      ffn: { kind: 'geglu', act: 'gelu', bias: false },
    },
    head: {
      type: 'julia', layers: enc.headLayers, ffn: enc.headFfn, options: enc.options,
    },
  };
}
