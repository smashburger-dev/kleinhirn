// Layer templates (K28 design section 3). Two shapes cover the models the
// engine runs today and are reused by K28.4 to K28.6: post-norm (BERT
// family, DeBERTa) and pre-norm (ModernBERT, Julia's head layers). Buffer
// ids are fixed by convention: x, tmp, normed, qkv, ctx, attnOut, mid, gate,
// ffnOut, mask, relidx, cossin, zero, dummy.

import type { LayerTensors } from './spec.ts';
import type { Op } from './ir.ts';

export interface LayerDims {
  hidden: number;
  heads: number;
  headDim: number;
  intermediate: number;
  rows: number;    // packed rows of the plan: length * batch
  length: number;  // bucket length (per sequence)
  eps: number;
}

const w = (name: string | undefined, zero = 'zero'): string =>
  name === undefined ? zero : `w:${name}`;
const req = (name: string | undefined, what: string): string => {
  if (name === undefined) throw new Error(`layer tensor ${what} missing`);
  return `w:${name}`;
};
const c16 = (n: number): number => Math.ceil(n / 16);

// The residual add strides over its elements (grid-stride loop in add.wgsl), so its dispatch
// is capped at the 65535 workgroups WebGPU allows along one dimension.
export const MAX_WORKGROUPS = 65535;
export const addWorkgroups = (rows: number, hidden: number): number =>
  Math.min(Math.ceil((rows * hidden) / 64), MAX_WORKGROUPS);

// Encoder matmuls go through the register-blocked kernel (32x64 tiles, K27) when K fits its
// 32-step, otherwise through the 16x16 kernel. Both sum k in order in f32: same bits.
function matmul(
  name: string, d: LayerDims, n: number, k: number, act: number, bind: string[],
): Op {
  const constants = { M: d.rows, N: n, K: k, ACT: act };
  return k % 32 === 0
    ? { name, kernel: 'mmtile', constants, bind, dispatch: [Math.ceil(n / 64), 'rows32'],
      // smaller tiles while the larger would start too few workgroups (all bit-equal, K27):
      // 8x32 while 16x32 would start fewer than 128, 16x32 while 32x64 would start fewer than 96
      alts: [
        { kernel: 'mmtile8', dispatch: [Math.ceil(n / 32), 'rows8'], maxRows: 16 * Math.floor(127 / Math.ceil(n / 32)) },
        { kernel: 'mmtile16', dispatch: [Math.ceil(n / 32), 'rows16'], maxRows: 32 * Math.floor(95 / Math.ceil(n / 64)) },
      ] }
    : { name, kernel: 'matmul', constants, bind, dispatch: [c16(n), 'rows16'] };
}

// Standard attention (no relative terms), K27: three matmul-shaped kernels (scores, softmax,
// context) when D % 32 == 0, L % 32 == 0 and the f32 scores fit one minimum-size binding;
// else the flash kernel (D % 4 == 0, D <= 64, L % 32 == 0); else one workgroup per query row.
const SCORES_MAX_BYTES = 134217728;
const matmulAttention = (d: LayerDims): boolean => d.headDim % 32 === 0 && d.length % 32 === 0
  && d.rows * d.heads * d.length * 4 <= SCORES_MAX_BYTES;

// DeBERTa relative attention as matmuls (K27): c2p = q . pos_key and p2c = k . pos_query over
// the nm position rows from moff, then scores, the softmax with the gathered relative terms
// and the context. Same score as the attention kernel: (q.k + c2p + p2c) / scale.
function relativeAttention(
  d: LayerDims, scale: number, rel: { moff: number; nm: number }, t: LayerTensors,
): Op[] {
  const base = { L: d.length, H: d.heads, D: d.headDim };
  const relC = { H: d.heads, D: d.headDim, ROWS: d.rows, NM: rel.nm, MOFF: rel.moff, L: d.length };
  const tiles = d.heads * Math.ceil(rel.nm / 64);
  return [
    { name: 'relC', kernel: 'attrel', constants: { ...relC, PART: 0 },
      bind: ['qkv', req(t.posKey, 'posKey'), 'kinfo', 'relidx', 'c2p'], dispatch: [tiles, 'rows32'] },
    { name: 'relP', kernel: 'attrel', constants: { ...relC, PART: 1 },
      bind: ['qkv', req(t.posQuery, 'posQuery'), 'kinfo', 'relidx', 'p2c'], dispatch: [tiles, 'rows32'] },
    { name: 'attS', kernel: 'attscore', constants: { ...base, SCALE: 1 / scale, ROWS: d.rows },
      bind: ['qkv', 'kinfo', 'scores'], dispatch: [d.heads * Math.ceil(d.length / 64), 'rows32'] },
    { name: 'attP', kernel: 'attsoftrel',
      constants: { L: d.length, H: d.heads, NM: rel.nm, MOFF: rel.moff, INVSCALE: 1 / scale },
      bind: ['mask', 'relidx', 'c2p', 'p2c', 'scores'], dispatch: [d.heads, 'rows'] },
    { name: 'attV', kernel: 'attpv', constants: { ...base, ROWS: d.rows },
      bind: ['scores', 'qkv', 'mask', 'kinfo', 'ctx'], dispatch: [d.heads * (d.headDim / 32), 'rows32'] },
  ];
}

function standardAttention(d: LayerDims, scale: number, window: number): Op[] {
  const base = { L: d.length, H: d.heads, D: d.headDim };
  if (matmulAttention(d)) {
    return [
      { name: 'attS', kernel: 'attscore', constants: { ...base, SCALE: scale, ROWS: d.rows },
        bind: ['qkv', 'kinfo', 'scores'], dispatch: [d.heads * Math.ceil(d.length / 64), 'rows32'] },
      { name: 'attP', kernel: 'attsoftmax', constants: { L: d.length, H: d.heads, WINDOW: window },
        bind: ['mask', 'scores'], dispatch: [d.heads, 'rows'] },
      { name: 'attV', kernel: 'attpv', constants: { ...base, ROWS: d.rows },
        bind: ['scores', 'qkv', 'mask', 'kinfo', 'ctx'], dispatch: [d.heads * (d.headDim / 32), 'rows32'] },
    ];
  }
  const constants = { ...base, SCALE: scale, WINDOW: window };
  return d.headDim % 4 === 0 && d.headDim <= 64 && d.length % 32 === 0
    ? [{ name: 'attn', kernel: 'mbflash', constants: { ...constants, ROWS: d.rows },
      bind: ['qkv', 'mask', 'ctx'], dispatch: [d.heads, 'rows32'] }]
    : [{ name: 'attn', kernel: 'mbattention', constants, bind: ['qkv', 'mask', 'ctx'], dispatch: [d.heads, 'rows'] }];
}

function layernorm(name: string, d: LayerDims, mode: number, bind: string[]): Op {
  return {
    name, kernel: 'layernorm', constants: { N: d.hidden, MODE: mode, EPS: d.eps },
    bind, dispatch: ['rows', 1],
  };
}

function add(name: string, d: LayerDims, bind: string[]): Op {
  return {
    name, kernel: 'add',
    constants: { TOTAL: d.rows * d.hidden, N: d.hidden, MODE: 0, L: d.length },
    bind, dispatch: [addWorkgroups(d.rows, d.hidden), 1],
  };
}

// qkv, attn, attnOut, lnA, ffn1, ffn2, lnF; the residual is fused into the
// LayerNorm (mode 2). x holds the stream, tmp the mid-layer state.
export function postNormLayer(
  d: LayerDims, t: LayerTensors,
  opts: { act: number; attnScale: number; standard?: boolean; rel?: { moff: number; nm: number } },
): Op[] {
  const H = d.hidden;
  const I = d.intermediate;
  return [
    matmul('qkv', d, 3 * H, H, 0,
      ['x', req(t.qkvW, 'qkvW'), w(t.qkvB), 'qkv']),
    ...(opts.standard
      ? standardAttention(d, opts.attnScale, 0)
      : opts.rel && matmulAttention(d)
        ? relativeAttention(d, opts.attnScale, opts.rel, t)
      : [<Op>{
        name: 'attn', kernel: 'attention',
        constants: { L: d.length, H: d.heads, D: d.headDim, SCALE: opts.attnScale },
        bind: ['qkv', req(t.posKey, 'posKey'), req(t.posQuery, 'posQuery'),
          'relidx', 'mask', 'ctx'],
        dispatch: [d.heads, 'rows'],
      }]),
    matmul('attnOut', d, H, H, 0,
      ['ctx', req(t.attnOutW, 'attnOutW'), w(t.attnOutB), 'attnOut']),
    layernorm('lnA', d, 2,
      ['x', 'attnOut', req(t.attnNormW, 'attnNormW'), w(t.attnNormB), 'mask', 'tmp']),
    matmul('ffn1', d, I, H, opts.act,
      ['tmp', req(t.ffnInW, 'ffnInW'), w(t.ffnInB), 'mid']),
    matmul('ffn2', d, H, I, 0,
      ['mid', req(t.ffnOutW, 'ffnOutW'), w(t.ffnOutB), 'ffnOut']),
    layernorm('lnF', d, 2,
      ['tmp', 'ffnOut', req(t.ffnNormW, 'ffnNormW'), w(t.ffnNormB), 'mask', 'x']),
  ];
}

export interface PreNormOpts {
  stream: string;              // 'x' in the encoder, 'tmp' in Julia's head
  firstNorm: boolean;          // false: qkv reads the stream directly
  attnScale: number;
  window: number;              // WINDOW of the attention kernel, 0 = global
  rope: boolean;
  ropeTable?: string;          // buffer id of the cos/sin table, default 'cossin'
  ffn: { kind: 'geglu' } | { kind: 'mlp'; width: number; act: number };
}

// lnA (optional), qkv, rope (optional), attn, attnOut, addA, lnF, then either
// ffnIn, geglu, ffnOut or lin1, lin2, and addF. Residuals use the add kernel
// in storage precision.
export function preNormLayer(d: LayerDims, t: LayerTensors, o: PreNormOpts): Op[] {
  const H = d.hidden;
  const I = d.intermediate;
  const ops: Op[] = [];
  if (o.firstNorm) {
    ops.push(layernorm('lnA', d, 0,
      [o.stream, 'dummy', req(t.attnNormW, 'attnNormW'), w(t.attnNormB), 'mask', 'normed']));
  }
  ops.push(matmul('qkv', d, 3 * H, H, 0,
    [o.firstNorm ? 'normed' : o.stream, req(t.qkvW, 'qkvW'), w(t.qkvB), 'qkv']));
  if (o.rope) {
    ops.push({
      name: 'rope', kernel: 'rope',
      constants: { L: d.length, H: d.heads, D: d.headDim },
      bind: ['qkv', o.ropeTable ?? 'cossin'], dispatch: ['rows', 2 * d.heads],
    });
  }
  ops.push(...standardAttention(d, o.attnScale, o.window));
  ops.push(matmul('attnOut', d, H, H, 0,
    ['ctx', req(t.attnOutW, 'attnOutW'), w(t.attnOutB), 'attnOut']));
  ops.push(add('addA', d, [o.stream, 'attnOut']));
  ops.push(layernorm('lnF', d, 0,
    [o.stream, 'dummy', req(t.ffnNormW, 'ffnNormW'), w(t.ffnNormB), 'mask', 'normed']));
  if (o.ffn.kind === 'geglu') {
    ops.push(matmul('ffnIn', d, 2 * I, H, 0,
      ['normed', req(t.ffnInW, 'ffnInW'), w(t.ffnInB), 'mid']));
    ops.push({
      name: 'geglu', kernel: 'geglu', constants: { I },
      bind: ['mid', 'gate'], dispatch: ['rows', 1],
    });
    ops.push(matmul('ffnOut', d, H, I, 0,
      ['gate', req(t.ffnOutW, 'ffnOutW'), w(t.ffnOutB), 'ffnOut']));
  } else {
    ops.push(matmul('lin1', d, o.ffn.width, H, o.ffn.act,
      ['normed', req(t.ffnInW, 'ffnInW'), w(t.ffnInB), 'mid']));
    ops.push(matmul('lin2', d, H, o.ffn.width, 0,
      ['mid', req(t.ffnOutW, 'ffnOutW'), w(t.ffnOutB), 'ffnOut']));
  }
  ops.push(add('addF', d, [o.stream, 'ffnOut']));
  return ops;
}
