/// <reference lib="esnext.float16" />
// K28 probe, variant B (in-browser conversion): safetensors header parse,
// name mapping to the manifest names of both models, CPU f32 -> f16 cast and
// fused upload. Mirrors convert/export_weights.py (small-upstream) and
// convert/export_julia.py (julia-1). Provisional research code.

export type ModelName = 'small-upstream' | 'julia-1';

export interface StTensor {
  dtype: string;
  shape: number[];
  off: number; // absolute byte offset in the file
  n: number; // element count
}

export interface StFile {
  buf: ArrayBuffer;
  tensors: Map<string, StTensor>;
}

// One manifest-named output tensor. kind 'pre' tensors (DeBERTa position
// projections) are produced by the GPU precompute, not by a plain cast.
export interface Spec {
  name: string;
  shape: number[];
  kind: 'cpu' | 'gpu' | 'pre';
  // Source pieces in row order; elemOffset is the position in the output.
  parts: { src: string; elemOffset: number }[];
}

export interface PreSources {
  rel: Float32Array;
  lnW: Float32Array;
  lnB: Float32Array;
  layers: { kw: Float32Array; kb: Float32Array; qw: Float32Array; qb: Float32Array }[];
}

export function parseSafetensors(buf: ArrayBuffer): StFile {
  const dv = new DataView(buf);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen)));
  const base = 8 + headerLen;
  const tensors = new Map<string, StTensor>();
  for (const [name, v] of Object.entries(header) as [string, any][]) {
    if (name === '__metadata__') continue;
    const n = (v.shape as number[]).reduce((a, b) => a * b, 1);
    tensors.set(name, { dtype: v.dtype, shape: v.shape, off: base + v.data_offsets[0], n });
  }
  return { buf, tensors };
}

// f32 view onto the file bytes (copies only if the offset is misaligned).
export function f32View(st: StFile, name: string): Float32Array {
  const t = st.tensors.get(name);
  if (!t) throw new Error(`missing tensor ${name}`);
  if (t.dtype !== 'F32') throw new Error(`${name}: dtype ${t.dtype}, expected F32`);
  if (t.off % 4 === 0) return new Float32Array(st.buf, t.off, t.n);
  return new Float32Array(st.buf.slice(t.off, t.off + t.n * 4));
}

function single(st: StFile, name: string, src: string, kind: 'cpu' | 'gpu' = 'gpu'): Spec {
  const t = st.tensors.get(src);
  if (!t) throw new Error(`missing tensor ${src}`);
  return { name, shape: t.shape, kind, parts: [{ src, elemOffset: 0 }] };
}

function fused(st: StFile, name: string, srcs: string[]): Spec {
  const parts: Spec['parts'] = [];
  let off = 0;
  let rows = 0;
  let cols = 0;
  for (const src of srcs) {
    const t = st.tensors.get(src);
    if (!t) throw new Error(`missing tensor ${src}`);
    parts.push({ src, elemOffset: off });
    off += t.n;
    rows += t.shape[0];
    cols = t.shape[1] ?? 0;
  }
  return { name, shape: cols ? [rows, cols] : [rows], kind: 'gpu', parts };
}

export function specsSmall(st: StFile, layers: number): Spec[] {
  const specs: Spec[] = [];
  specs.push(single(st, 'embeddings.word.weight', 'encoder.embeddings.word_embeddings.weight', 'cpu'));
  specs.push(single(st, 'embeddings.LayerNorm.weight', 'encoder.embeddings.LayerNorm.weight'));
  specs.push(single(st, 'embeddings.LayerNorm.bias', 'encoder.embeddings.LayerNorm.bias'));
  const rel = st.tensors.get('encoder.encoder.rel_embeddings.weight')!;
  specs.push({ name: 'rel_embeddings', shape: rel.shape, kind: 'pre', parts: [] });
  for (let i = 0; i < layers; i += 1) {
    const p = `encoder.encoder.layer.${i}`;
    const s = `${p}.attention.self`;
    const q = (k: string) => [`${s}.query_proj.${k}`, `${s}.key_proj.${k}`, `${s}.value_proj.${k}`];
    specs.push(fused(st, `layers.${i}.qkv.weight`, q('weight')));
    specs.push(fused(st, `layers.${i}.qkv.bias`, q('bias')));
    const pairs: [string, string][] = [
      ['attn_out', `${p}.attention.output.dense`], ['attn_ln', `${p}.attention.output.LayerNorm`],
      ['ffn_in', `${p}.intermediate.dense`], ['ffn_out', `${p}.output.dense`],
      ['ffn_ln', `${p}.output.LayerNorm`],
    ];
    for (const [dst, src] of pairs) {
      for (const k of ['weight', 'bias']) specs.push(single(st, `layers.${i}.${dst}.${k}`, `${src}.${k}`));
    }
    const h = st.tensors.get(`${s}.key_proj.weight`)!.shape[0];
    specs.push({ name: `layers.${i}.pos_key`, shape: [rel.shape[0], h], kind: 'pre', parts: [] });
    specs.push({ name: `layers.${i}.pos_query`, shape: [rel.shape[0], h], kind: 'pre', parts: [] });
  }
  for (const [dst, src] of [['head.fc1', 'classifier.0'], ['head.fc2', 'classifier.3']]) {
    for (const k of ['weight', 'bias']) specs.push(single(st, `${dst}.${k}`, `${src}.${k}`));
  }
  return specs;
}

export function preSourcesSmall(st: StFile, layers: number): PreSources {
  const out: PreSources = {
    rel: f32View(st, 'encoder.encoder.rel_embeddings.weight'),
    lnW: f32View(st, 'encoder.encoder.LayerNorm.weight'),
    lnB: f32View(st, 'encoder.encoder.LayerNorm.bias'),
    layers: [],
  };
  for (let i = 0; i < layers; i += 1) {
    const s = `encoder.encoder.layer.${i}.attention.self`;
    out.layers.push({
      kw: f32View(st, `${s}.key_proj.weight`), kb: f32View(st, `${s}.key_proj.bias`),
      qw: f32View(st, `${s}.query_proj.weight`), qb: f32View(st, `${s}.query_proj.bias`),
    });
  }
  return out;
}

export function specsJulia(st: StFile, layers: number): Spec[] {
  const specs: Spec[] = [];
  specs.push(single(st, 'embeddings.word.weight', 'encoder.embeddings.tok_embeddings.weight', 'cpu'));
  specs.push(single(st, 'embeddings.norm.weight', 'encoder.embeddings.norm.weight'));
  for (let i = 0; i < layers; i += 1) {
    const p = `encoder.layers.${i}`;
    if (st.tensors.has(`${p}.attn_norm.weight`)) {
      specs.push(single(st, `layers.${i}.attn_norm.weight`, `${p}.attn_norm.weight`));
    }
    specs.push(single(st, `layers.${i}.wqkv.weight`, `${p}.attn.Wqkv.weight`));
    specs.push(single(st, `layers.${i}.attn_out.weight`, `${p}.attn.Wo.weight`));
    specs.push(single(st, `layers.${i}.mlp_norm.weight`, `${p}.mlp_norm.weight`));
    specs.push(single(st, `layers.${i}.mlp_in.weight`, `${p}.mlp.Wi.weight`));
    specs.push(single(st, `layers.${i}.mlp_out.weight`, `${p}.mlp.Wo.weight`));
  }
  specs.push(single(st, 'final_norm.weight', 'encoder.final_norm.weight'));
  specs.push(single(st, 'type_emb.weight', 'type_emb.weight'));
  for (let i = 0; i < 2; i += 1) {
    const p = `head.layers.${i}`;
    specs.push(single(st, `head.${i}.in_proj.weight`, `${p}.self_attn.in_proj_weight`));
    specs.push(single(st, `head.${i}.in_proj.bias`, `${p}.self_attn.in_proj_bias`));
    for (const [src, dst] of [['self_attn.out_proj', 'out_proj'], ['norm1', 'norm1'],
      ['norm2', 'norm2'], ['linear1', 'linear1'], ['linear2', 'linear2']]) {
      for (const k of ['weight', 'bias']) specs.push(single(st, `head.${i}.${dst}.${k}`, `${p}.${src}.${k}`));
    }
  }
  for (const [dst, src] of [['scorer.norm', 'scorer.0'], ['scorer.fc1', 'scorer.1'],
    ['scorer.fc2', 'scorer.3'], ['act_head.fc1', 'act_head.0'], ['act_head.fc2', 'act_head.2']]) {
    for (const k of ['weight', 'bias']) specs.push(single(st, `${dst}.${k}`, `${src}.${k}`));
  }
  specs.push(single(st, 'temperature', 'temperature'));
  return specs;
}

export function makeSpecs(model: ModelName, st: StFile, layers: number): Spec[] {
  return model === 'small-upstream' ? specsSmall(st, layers) : specsJulia(st, layers);
}

export function padded(byteLength: number): number {
  return Math.ceil(byteLength / 4) * 4; // writeBuffer needs %4, as in src/weights.ts
}

export const STORAGE_USAGE = 0x80 | 0x08 | 0x04; // STORAGE | COPY_DST | COPY_SRC

export interface CpuLoaded {
  tensors: Map<string, GPUBuffer>;
  cpuBytes: Map<string, { elemOffset: number; arr: Float16Array }[]>;
  embeddings: Uint16Array;
  gpuBytes: number;
  castMs: number;
  uploadMs: number;
}

// Variant Bcpu: cast every tensor on the CPU with Float16Array, write fused
// parts at their row offset. 'pre' specs are left to the GPU precompute.
export function loadCpuCast(device: GPUDevice, st: StFile, specs: Spec[]): CpuLoaded {
  const tensors = new Map<string, GPUBuffer>();
  const cpuBytes = new Map<string, { elemOffset: number; arr: Float16Array }[]>();
  let embeddings = new Uint16Array(0);
  let gpuBytes = 0;
  let castMs = 0;
  let uploadMs = 0;
  for (const s of specs) {
    if (s.kind === 'pre') continue;
    const kept: { elemOffset: number; arr: Float16Array }[] = [];
    let buf: GPUBuffer | null = null;
    if (s.kind === 'gpu') {
      const total = s.shape.reduce((a, b) => a * b, 1);
      const bytes = padded(total * 2);
      const t0 = performance.now();
      buf = device.createBuffer({ size: bytes, usage: STORAGE_USAGE });
      uploadMs += performance.now() - t0;
      gpuBytes += bytes;
      tensors.set(s.name, buf);
    }
    for (const part of s.parts) {
      const t0 = performance.now();
      const arr = new Float16Array(f32View(st, part.src));
      castMs += performance.now() - t0;
      if (s.kind === 'cpu') {
        embeddings = new Uint16Array(arr.buffer, 0, arr.length);
        kept.push({ elemOffset: 0, arr });
        continue;
      }
      const t1 = performance.now();
      const byteOff = part.elemOffset * 2;
      if (arr.byteLength % 4 === 0) {
        device.queue.writeBuffer(buf!, byteOff, arr.buffer, 0, arr.byteLength);
      } else {
        const pad = new Uint8Array(padded(arr.byteLength));
        pad.set(new Uint8Array(arr.buffer, 0, arr.byteLength));
        device.queue.writeBuffer(buf!, byteOff, pad);
      }
      uploadMs += performance.now() - t1;
      kept.push({ elemOffset: part.elemOffset, arr });
    }
    cpuBytes.set(s.name, kept);
  }
  return { tensors, cpuBytes, embeddings, gpuBytes, castMs, uploadMs };
}
