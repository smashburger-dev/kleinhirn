// Plan builder (K28 design section 3): ModelSpec + head description -> Plan.
// All numbers that reach the kernels (override constants, dispatch sizes,
// buffer sizes) are computed with the expressions of the engine before K28,
// so the dispatch list and the logit bits stay identical.

import type { BufferDecl, Dim, KernelName, Op, Plan, Segment } from './ir.ts';
import {
  addWorkgroups, postNormLayer, preNormLayer, type LayerDims, type PreNormOpts,
} from './layers.ts';
import {
  tensorNamesFor, type HeadSpec, type LayerTensors, type ModelSpec, type TensorNames,
} from './spec.ts';

export interface BuildOptions {
  length: number;
  batch: number;
  markers: number;
  f16: boolean;
  tensorNames?: TensorNames;
}

function relBucketIndex(i: number, j: number, span: number, maxPos: number): number {
  const rel = i - j;
  const mid = span >> 1;
  const abs = Math.abs(rel);
  let r = rel;
  if (abs > mid) {
    const logPos = Math.ceil(
      Math.log(abs / mid) / Math.log((maxPos - 1) / mid) * (mid - 1)) + mid;
    r = logPos * Math.sign(rel);
  }
  return Math.min(Math.max(r + span, 0), 2 * span - 1);
}

export function relPosTable(
  length: number, span: number, maxPos: number,
): Uint32Array<ArrayBuffer> {
  const t = new Uint32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      t[i * length + j] = relBucketIndex(i, j, span, maxPos);
    }
  }
  return t;
}

export function ropeTable(
  length: number, theta: number, headDim: number,
): Float32Array<ArrayBuffer> {
  const half = headDim / 2;
  const t = new Float32Array(length * 2 * headDim);
  for (let i = 0; i < length; i += 1) {
    for (let d = 0; d < half; d += 1) {
      const f = i * theta ** (-(2 * d) / headDim);
      t[i * 2 * headDim + d] = Math.cos(f);
      t[i * 2 * headDim + headDim + d] = Math.sin(f);
    }
  }
  return t;
}

// Writable bindings per kernel: the last binding, except add and rope which
// update binding 0 in place.
export function writableBindings(kernel: KernelName, bindings: number): number[] {
  return kernel === 'add' || kernel === 'rope' ? [0] : [bindings - 1];
}

// A writable buffer may not be bound a second time in the same dispatch
// (WebGPU usage scope rule). Throws with the name of the operation.
export function validateSegments(segments: Segment[], known: Set<string>): void {
  for (const seg of segments) {
    for (const op of [...(seg.captureOps ?? []), ...seg.ops]) {
      const where = `${seg.prefix}${op.name}`;
      for (const id of op.bind) {
        if (!id.startsWith('w:') && !known.has(id)) {
          throw new Error(`plan op ${where}: unknown buffer ${id}`);
        }
      }
      for (const wb of writableBindings(op.kernel, op.bind.length)) {
        const id = op.bind[wb];
        if (op.bind.some((other, i) => i !== wb && other === id)) {
          throw new Error(`plan op ${where}: writable buffer ${id} is bound twice`);
        }
      }
    }
  }
}

export { MAX_WORKGROUPS, addWorkgroups } from './layers.ts';

const ACT: Record<string, number> = { relu: 1, gelu: 2, tanh: 3, silu: 4 };
function actCode(act: string): number {
  const code = ACT[act];
  if (code === undefined) throw new Error(`activation ${act} has no kernel yet`);
  return code;
}

const elementsWritten = (op: Op): number => {
  switch (op.kernel) {
    case 'matmul': case 'mmtile': case 'mmtile16': case 'mmtile8': case 'layernorm': case 'embln': case 'pool': return op.constants.N;
    case 'geglu': return op.constants.I;
    case 'attention': case 'mbattention': case 'mbflash': case 'attpv': return op.constants.H * op.constants.D;
    case 'gather': return op.constants.D;
    case 'im2col': return op.constants.N * op.constants.KS;
    default: return 0;
  }
};

export function buildPlan(spec: ModelSpec, head: HeadSpec, opts: BuildOptions): Plan {
  const { length, batch, f16 } = opts;
  const names = opts.tensorNames ?? tensorNamesFor(spec.family, spec);
  const H = spec.hidden;
  const E = spec.embeddingSize;
  // Multiples of 4 bytes: copyBufferToBuffer needs it, and a one-logit f16
  // output would have 2. The sizes of GLiNER and Julia are multiples of 4
  // already, so their plans do not change.
  const bytesOf = (el: number): number => Math.ceil((el * (f16 ? 2 : 4)) / 4) * 4;
  const M = length * batch;
  const pooledHead = head.type === 'classify' || head.type === 'token' || head.type === 'embed';
  const markers = head.type === 'julia' ? head.options : pooledHead ? 0 : opts.markers;
  const K = markers * batch;
  const eps = spec.block.norm.eps;
  const dims: LayerDims = {
    hidden: H, heads: spec.heads, headDim: spec.headDim,
    intermediate: spec.intermediate, rows: M, length, eps,
  };
  const ln = (name: string, mode: number, bind: string[], n = H, rows: Dim = 'rows'): Op => ({
    name, kernel: 'layernorm', constants: { N: n, MODE: mode, EPS: eps },
    bind, dispatch: [rows, 1],
  });
  const mm = (
    name: string, rows: number, n: number, k: number, act: number, bind: string[],
    yDim: Dim,
  ): Op => ({
    name, kernel: 'matmul', constants: { M: rows, N: n, K: k, ACT: act },
    bind, dispatch: [Math.ceil(n / 16), yDim],
  });
  const K16 = Math.ceil(K / 16);

  const segments: Segment[] = [];
  let slot = 0;
  const take = (...buffers: string[]) => buffers.map((buffer) => ({ buffer, slot: slot++ }));

  // Embedding.
  const embW = `w:${names.embedNormW}`;
  const embB = names.embedNormB ? `w:${names.embedNormB}` : 'zero';
  if (spec.embed.positions === 'absolute') {
    // Families without a type table (DistilBERT) bind the shared zero buffer:
    // embln reads row 0 of it for every type id 0, so the sum is unchanged.
    const typeTable = spec.embed.typeVocab > 0 ? 'w:embeddings.type.weight' : 'zero';
    // Word + position + type in f32, LayerNorm over E; with a projection the
    // norm writes embE and a matmul takes E to H.
    const dst = spec.embed.project ? 'embE' : 'x';
    const ops: Op[] = [{
      name: 'embed', kernel: 'embln',
      constants: {
        N: E, L: length, OFFSET: spec.embed.positionOffset, MAXPOS: spec.embed.maxPositions,
        EPS: spec.embed.norm.eps, MASKMUL: spec.embed.maskMultiply ? 1 : 0 },
      bind: ['emb', 'w:embeddings.position.weight', typeTable,
        'typeIds', embW, embB, 'mask', dst],
      dispatch: ['rows', 1],
    }];
    if (spec.embed.project) {
      ops.push(mm('project', M, H, E, 0,
        ['embE', 'w:embeddings.project.weight', 'w:embeddings.project.bias', 'x'], 'rows16'));
    }
    segments.push({ name: 'embed', prefix: '', ops, capture: take('x') });
  } else if (spec.embed.maskMultiply) {
    segments.push({
      name: 'embed', prefix: '',
      ops: [ln('embed', 1, ['emb', 'dummy', embW, embB, 'mask', 'x'])],
      captureOps: [ln('embedPlain', 0, ['emb', 'dummy', embW, embB, 'mask', 'tmp'])],
      capture: take('tmp', 'x'),
    });
  } else {
    segments.push({
      name: 'embed', prefix: '',
      ops: [ln('embed', 0, ['emb', 'dummy', embW, embB, 'mask', 'x'])],
      capture: take('x'),
    });
  }

  // Convolution of DeBERTa-v2 (conv_kernel_size > 0). HF computes, after layer 0,
  //   x = LN(layer0_out + act(maskfill(conv1d(emb)))) * mask
  // where emb is the encoder input (the embedding after the mask multiply). Layer 0 overwrites x, so
  // the segment 'convIn' runs before it: im2col of x into cols, then cols @ Wc^T + b with the
  // activation into convOut. After layer 0 the segment 'conv' adds x into convOut (a + b equals b + a
  // bit for bit) and writes LN(convOut) * mask back to x (layernorm MODE 1). HF zeroes the rows
  // with mask 0 before the activation; here those rows hold act(bias) instead, but the mask factor of
  // MODE 1 turns them to 0 afterwards, and no row with mask 1 reads them: im2col takes the
  // neighbours from emb, and im2col itself reads a neighbour with mask 0 as zero (the executor
  // dispatches only the rows below seqLen, so a padding row of x can still hold an earlier call).
  const conv = spec.conv;
  if (conv) {
    if (!names.conv) throw new Error('conv without tensor names');
    segments.push({
      name: 'convIn', prefix: 'ConvIn.',
      ops: [
        {
          name: 'im2col', kernel: 'im2col', constants: { N: H, L: length, KS: conv.kernel },
          bind: ['x', 'mask', 'cols'], dispatch: ['rows', 1],
        },
        mm('conv', M, H, conv.kernel * H, actCode(conv.act),
          ['cols', `w:${names.conv.w}`, `w:${names.conv.b}`, 'convOut'], 'rows16'),
      ],
    });
  }

  // Layers.
  const rel = spec.attention.kind === 'deberta-relative' ? spec.attention.rel : undefined;
  let attnScaleOf: (l: number) => { scale: number; window: number };
  if (spec.attention.kind === 'deberta-relative') {
    if (!rel) throw new Error('deberta-relative attention needs attention.rel');
    // Same expression as the engine before K28 (attention kernel): two position
    // types give 3 = 1 + 2.
    const scale = Math.sqrt((1 + rel.types.length) * (H / spec.heads));
    attnScaleOf = () => ({ scale, window: 0 });
  } else {
    const win = spec.attention.window;
    // Same expression as the engine before K28 (mbattention kernel).
    const scale = (H / spec.heads) ** -0.5;
    attnScaleOf = (l) => ({
      scale, window: !win || l % win.globalEvery === 0 ? 0 : win.half });
  }
  const rope = spec.attention.rope;
  // One table when both thetas agree (Julia), else one per layer type: global
  // layers (no window) read cossinG, windowed layers cossinL.
  const ropeSplit = !!rope && rope.thetaGlobal !== rope.thetaLocal;
  for (let l = 0; l < spec.layers; l += 1) {
    const t = names.layer(l);
    const a = attnScaleOf(l);
    let ops: Op[];
    if (spec.block.order === 'post') {
      ops = postNormLayer(dims, t, {
        act: actCode(spec.ffn.act), attnScale: a.scale,
        standard: spec.attention.kind === 'standard',
        // the position rows the relative buckets of this length reach (K27 matmul attention)
        rel: rel && {
          moff: relBucketIndex(0, length - 1, rel.buckets, rel.maxPositions),
          nm: relBucketIndex(length - 1, 0, rel.buckets, rel.maxPositions)
            - relBucketIndex(0, length - 1, rel.buckets, rel.maxPositions) + 1,
        } });
    } else {
      const identity = spec.block.firstNormIdentity && l === 0;
      const ffn: PreNormOpts['ffn'] = spec.ffn.kind === 'geglu'
        ? { kind: 'geglu' }
        : { kind: 'mlp', width: spec.intermediate, act: actCode(spec.ffn.act) };
      ops = preNormLayer(dims, identity ? { ...t, attnNormW: undefined } : t, {
        stream: 'x', firstNorm: !identity, attnScale: a.scale, window: a.window,
        rope: !!rope, ffn,
        ropeTable: ropeSplit ? (a.window === 0 ? 'cossinG' : 'cossinL') : 'cossin',
      });
    }
    // With a convolution the state after layer 0 is the one after 'conv' (as hidden_states[1] in HF).
    const afterConv = l === 0 && conv;
    segments.push({
      name: `layer${l}`, prefix: `L${l}.`, ops, skippable: true,
      ...(afterConv ? {} : { capture: take('x') }) });
    if (afterConv && names.conv) {
      segments.push({
        name: 'conv', prefix: 'Conv.',
        ops: [
          { name: 'add', kernel: 'add',
            constants: { TOTAL: M * H, N: H, MODE: 0, L: length },
            bind: ['convOut', 'x'], dispatch: [addWorkgroups(M, H), 1] },
          ln('lnConv', 1, ['convOut', 'dummy', `w:${names.conv.normW}`, `w:${names.conv.normB}`, 'mask', 'x']),
        ],
        capture: take('x'),
      });
    }
  }

  let stream = 'x';
  if (spec.block.finalNorm) {
    if (!names.finalNormW) throw new Error('finalNorm without a tensor name');
    segments.push({
      name: 'final', prefix: '',
      ops: [ln('final', 0, ['x', 'dummy', `w:${names.finalNormW}`,
        names.finalNormB ? `w:${names.finalNormB}` : 'zero', 'mask', 'tmp'])],
      capture: take('tmp'),
    });
    stream = 'tmp';
  }

  // Head.
  let rowSelect: Plan['rowSelect'];
  // Buffers of the pooled and per-row heads, declared after the generic ones.
  const headBuffers: { id: string; elements: number; final: boolean }[] = [];
  let outRows = K;
  let outCols = 1;
  let outBuffer = 'logits';
  let needFirst = false;
  if (head.type === 'classify' || head.type === 'token' || head.type === 'embed') {
    const perRow = head.type === 'token';
    const rows = perRow ? M : batch;
    const rows16: Dim = perRow ? 'rows16' : Math.ceil(batch / 16);
    let prev = stream;
    let width = H;
    const ops: Op[] = [];
    if (!perRow) {
      const pool = head.pool === 'cls' ? 'first' : head.pool;
      if (pool === 'first') {
        needFirst = true;
        ops.push({
          name: 'gather', kernel: 'gather', constants: { K: 1, D: H, L: length },
          bind: ['first', stream, 'states'], dispatch: [1, 1],
        });
      } else {
        ops.push({
          name: 'pool', kernel: 'pool',
          constants: { L: length, N: H, MODE: pool === 'mean' ? 0 : 1 },
          bind: [stream, 'mask', 'states'], dispatch: [Math.ceil(H / 64), batch],
        });
      }
      headBuffers.push({ id: 'states', elements: batch * H, final: false });
      prev = 'states';
    }
    head.steps.forEach((step, i) => {
      if (step.op === 'norm') {
        // LayerNorm over the rows of the head (batch rows when pooled, every row for tokens).
        const last = i === head.steps.length - 1;
        const dst = last ? 'out' : `hd${i}`;
        ops.push({
          name: `step${i}`, kernel: 'layernorm',
          constants: { N: width, MODE: 0, EPS: step.eps },
          bind: [prev, 'dummy', `w:${step.name}.weight`,
            step.bias ? `w:${step.name}.bias` : 'zero', 'mask', dst],
          dispatch: [perRow ? 'rows' : batch, 1],
        });
        headBuffers.push({ id: dst, elements: rows * width, final: last });
        prev = dst;
        return;
      }
      if (step.in !== width) {
        throw new Error(`head step ${step.name}: input ${step.in}, previous width ${width}`);
      }
      const last = i === head.steps.length - 1;
      const dst = last ? 'out' : `hd${i}`;
      ops.push(mm(`step${i}`, rows, step.out, step.in,
        step.act === 'none' ? 0 : actCode(step.act),
        [prev, `w:${step.name}.weight`, step.bias ? `w:${step.name}.bias` : 'zero', dst],
        rows16));
      headBuffers.push({ id: dst, elements: rows * step.out, final: last });
      prev = dst;
      width = step.out;
    });
    if (!head.steps.length) {
      if (perRow) throw new Error('a token head needs at least one dense step');
      headBuffers.find((b) => b.id === 'states')!.final = true;
    }
    outBuffer = prev;
    outRows = rows;
    outCols = width;
    segments.push({ name: 'head', prefix: 'head.', ops });
  } else if (head.type === 'gliner2') {
    const hh = head.hidden;
    segments.push({
      name: 'head', prefix: 'head.',
      ops: [
        {
          name: 'gather', kernel: 'gather', constants: { K: markers, D: H, L: length },
          bind: ['packed', stream, 'states'], dispatch: [1, 1],
        },
        mm('fc1', K, hh, H, 1,
          ['states', 'w:head.fc1.weight', 'w:head.fc1.bias', 'h1'], K16),
        mm('fc2', K, 1, hh, 0,
          ['h1', 'w:head.fc2.weight', 'w:head.fc2.bias', 'raw'], K16),
        {
          name: 'maskl', kernel: 'masklogits',
          constants: { K: markers, TEMP: head.temperature },
          bind: ['raw', 'packed', 'logits'], dispatch: [1, 1],
        },
      ],
    });
  } else {
    if (stream !== 'tmp') throw new Error('julia head expects a final norm');
    segments.push({
      name: 'typed', prefix: '',
      ops: [{
        name: 'type', kernel: 'add',
        constants: { TOTAL: M * H, N: H, MODE: 1, L: length },
        bind: ['tmp', 'typeRow'], dispatch: [addWorkgroups(M, H), 1],
      }],
      capture: take('tmp'),
    });
    rowSelect = {
      table: 'w:type_emb.weight', dst: 'typeRow', rowBytes: bytesOf(H) };
    for (let i = 0; i < head.layers; i += 1) {
      const p = `head.${i}`;
      const t: LayerTensors = {
        qkvW: `${p}.in_proj.weight`, qkvB: `${p}.in_proj.bias`,
        attnOutW: `${p}.out_proj.weight`, attnOutB: `${p}.out_proj.bias`,
        attnNormW: `${p}.norm1.weight`, attnNormB: `${p}.norm1.bias`,
        ffnNormW: `${p}.norm2.weight`, ffnNormB: `${p}.norm2.bias`,
        ffnInW: `${p}.linear1.weight`, ffnInB: `${p}.linear1.bias`,
        ffnOutW: `${p}.linear2.weight`, ffnOutB: `${p}.linear2.bias`,
      };
      segments.push({
        name: `head${i}`, prefix: `head${i}.`,
        ops: preNormLayer(dims, t, {
          stream: 'tmp', firstNorm: true, attnScale: (H / spec.heads) ** -0.5,
          window: 0, rope: false,
          ffn: { kind: 'mlp', width: head.ffn, act: 1 },
        }),
        capture: take('tmp'),
      });
    }
    segments.push({
      name: 'scorer', prefix: 'scorer.',
      ops: [
        {
          name: 'gather', kernel: 'gather', constants: { K: markers, D: H, L: length },
          bind: ['packed', 'tmp', 'states'], dispatch: [1, 1],
        },
        ln('ln', 0, ['states', 'dummy', 'w:scorer.norm.weight', 'w:scorer.norm.bias',
          'mask', 'kNormed'], H, K),
        mm('fc1', K, H, H, 2,
          ['kNormed', 'w:scorer.fc1.weight', 'w:scorer.fc1.bias', 'h1'], K16),
        mm('fc2', K, 1, H, 0,
          ['h1', 'w:scorer.fc2.weight', 'w:scorer.fc2.bias', 'raw'], K16),
        {
          name: 'maskl', kernel: 'masklogits', constants: { K: markers, TEMP: 1.0 },
          bind: ['raw', 'packed', 'logits'], dispatch: [1, 1],
        },
      ],
    });
  }

  // Buffers. Intermediate sizes are the maximum over the operations that
  // write them, times their row count.
  const all = segments.flatMap((s) => [...(s.captureOps ?? []), ...s.ops]);
  const cols = new Map<string, number>();
  for (const op of all) {
    const dst = op.bind[writableBindings(op.kernel, op.bind.length)[0]];
    const n = elementsWritten(op);
    if (n > 0) cols.set(dst, Math.max(cols.get(dst) ?? 0, n));
  }
  const kRows = new Set(['states', 'kNormed', 'h1', 'raw']);
  const decls: BufferDecl[] = [];
  const st = (id: string, bytes: number, usage: BufferDecl['usage'] = 'rw',
    init?: ArrayBufferView): void => {
    decls.push(init ? { id, bytes, usage, init } : { id, bytes, usage });
  };
  const rowsBuf = (id: string, usage: BufferDecl['usage'] = 'rw'): void => {
    const c = cols.get(id);
    if (c === undefined) return;
    st(id, bytesOf((kRows.has(id) ? K : M) * c), usage);
  };
  st('emb', bytesOf(M * E));
  if (spec.embed.project) st('embE', bytesOf(M * E));
  if (spec.embed.positions === 'absolute') st('typeIds', M * 4);
  for (const id of ['x', 'tmp']) {
    if (cols.has(id)) st(id, bytesOf(M * (cols.get(id) as number)), 'rwSrc');
  }
  if (conv) {
    st('cols', bytesOf(M * conv.kernel * H));
    st('convOut', bytesOf(M * H));
  }
  for (const id of ['normed', 'qkv', 'ctx', 'attnOut', 'mid', 'gate', 'ffnOut',
    'states', 'kNormed', 'h1', 'raw']) {
    if (!(pooledHead && kRows.has(id))) rowsBuf(id);
  }
  st('mask', M * 4);
  // first and last valid key per sequence, written by the executor at upload (K27)
  if (all.some((o) => o.bind.includes('kinfo'))) st('kinfo', 8 * batch);
  // f32 attention scores [row][head][key] of the matmul attention (K27)
  const scoreOps = all.filter((o) => o.kernel === 'attscore');
  for (const id of ['c2p', 'p2c']) {
    const relOps = all.filter((o) => o.kernel === 'attrel' && o.bind[4] === id);
    if (relOps.length) st(id, Math.max(...relOps.map((o) => o.constants.ROWS * o.constants.H * o.constants.NM * 4)));
  }
  if (scoreOps.length) {
    st('scores', Math.max(...scoreOps.map((o) => o.constants.ROWS * o.constants.H * o.constants.L * 4)));
  }
  if (all.some((o) => o.bind.includes('dummy'))) st('dummy', 4, 'storage');
  if (pooledHead) {
    for (const b of headBuffers) st(b.id, bytesOf(b.elements), b.final ? 'logits' : 'rw');
    // Pooling at position 0 reads row b * L; the marker buffer holds zeros.
    if (needFirst) st('first', 3 * batch * 4, 'rw', new Uint32Array(3 * batch));
  } else {
    st('packed', 3 * K * 4);
    st('logits', K * 4, 'logits');
  }
  if (rel) {
    const table = relPosTable(length, rel.buckets, rel.maxPositions);
    st('relidx', table.byteLength, 'rw', table);
  }
  if (rope && !ropeSplit) {
    const table = ropeTable(length, rope.thetaGlobal, spec.headDim);
    st('cossin', table.byteLength, 'rw', table);
  } else if (rope) {
    const global = ropeTable(length, rope.thetaGlobal, spec.headDim);
    const local = ropeTable(length, rope.thetaLocal, spec.headDim);
    st('cossinG', global.byteLength, 'rw', global);
    st('cossinL', local.byteLength, 'rw', local);
  }
  if (rowSelect) st('typeRow', bytesOf(batch * H));
  const zeroUses = all.filter((o) => o.bind.includes('zero'));
  if (zeroUses.length) {
    const width = Math.max(...zeroUses.map((o) => o.constants.N));
    st('zero', bytesOf(width), 'rw', new Uint8Array(bytesOf(width)));
  }

  validateSegments(segments, new Set(decls.map((d) => d.id)));
  return {
    f16, length, batch, markers, embeddingSize: E, buffers: decls, segments,
    inputs: pooledHead
      ? {
        embeddings: 'emb', mask: 'mask',
        // DeBERTa with relative attention and ModernBERT have no type ids input.
        ...(spec.embed.positions === 'absolute' ? { typeIds: 'typeIds' } : {}),
      }
      : { embeddings: 'emb', mask: 'mask', markers: 'packed' },
    rowSelect,
    output: pooledHead
      ? {
        buffer: outBuffer, bytes: bytesOf(outRows * outCols), dtype: 'storage',
        rows: outRows, cols: outCols }
      : { buffer: 'logits', bytes: K * 4, dtype: 'f32', rows: K, cols: 1 },
    captureSlots: slot,
    captureSlotBytes: length * H * 4,
  };
}
