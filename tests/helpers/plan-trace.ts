// Canonical dispatch traces for plan classes (K28.1). One trace per
// configuration: the build-time uploads, everything a call encodes, and the
// counters around it, written so that two implementations that submit the
// same GPU work compare equal even if they create buffers, pipelines and bind
// groups in a different order. Model specs are constants here; the test
// needs neither models nor a GPU.

import { createHash } from 'node:crypto';
import { createMockGpu, installGpuGlobals, type MockBuffer, type RawEvent } from './mock-gpu.ts';

export type Shape = 'small' | 'base' | 'julia-1';
export type Mode = 'normal' | 'normal100' | 'capture' | 'skip' | 'pass' | 'dispatch';

export interface TraceConfig {
  id: string;
  shape: Shape;
  f16: boolean;
  batch: number;
  length: number;
  markers: number;
  mode: Mode;
}

export const GLINER_SPECS = {
  small: {
    hiddenSize: 384, layers: 12, heads: 6, intermediateSize: 1536,
    positionBuckets: 256, maxRelativePositions: 512, layerNormEps: 1e-7,
  },
  base: {
    hiddenSize: 768, layers: 12, heads: 12, intermediateSize: 3072,
    positionBuckets: 256, maxRelativePositions: 512, layerNormEps: 1e-7,
  },
};
export const GLINER_HEAD_HIDDEN = { small: 768, base: 1536 };
export const GLINER_TEMPERATURE = 1.0;

export const JULIA_SPEC = {
  layers: 22, hiddenSize: 384, heads: 6, intermediate: 1152, normEps: 1e-5,
  ropeTheta: 160000, localAttention: 64, globalEvery: 3, headLayers: 2,
  headFfn: 1536, options: 20,
};
// The layer-0 attention norm is an Identity in the Julia checkpoint: the
// manifest has no such tensor.
export const JULIA_ABSENT = ['layers.0.attn_norm.weight'];

export function allConfigs(): TraceConfig[] {
  const out: TraceConfig[] = [];
  const shapes: { shape: Shape; sizes: [number, number][] }[] = [
    { shape: 'small', sizes: [[128, 16], [512, 16]] },
    { shape: 'base', sizes: [[128, 16], [512, 16], [1280, 80]] },
    { shape: 'julia-1', sizes: [[128, 20], [512, 20]] },
  ];
  for (const { shape, sizes } of shapes) {
    for (const f16 of [false, true]) {
      for (const batch of [1, 4]) {
        for (const [length, markers] of sizes) {
          const modes: Mode[] = ['normal', 'normal100', 'pass'];
          if (batch === 1) modes.push('dispatch');
          if (batch === 1 && !f16) modes.push('capture');
          if (shape !== 'julia-1') modes.push('skip');
          for (const mode of modes) {
            out.push({
              id: `${shape}-${f16 ? 'f16' : 'f32'}-B${batch}-L${length}K${markers}-${mode}`,
              shape, f16, batch, length, markers, mode,
            });
          }
        }
      }
    }
  }
  return out;
}

// What both implementations have to offer to the tracer.
export interface PlanLike {
  gpuBytes: number;
  upload(input: never): void;
  readLogits(): Promise<Float32Array>;
  readCapture(): Promise<Float32Array>;
  profileForward(
    input: never, granularity: 'pass' | 'dispatch', seqLen: number,
  ): Promise<{ times: Record<string, number> } | null>;
}

export interface Impl {
  build(cfg: TraceConfig, device: GPUDevice, tensors: Map<string, GPUBuffer>): PlanLike;
  submit(
    plan: PlanLike, cfg: TraceConfig,
    opts: { capture: boolean; seqLen: number; skip?: Set<number> },
  ): void;
}

export interface Trace {
  gpuBytesBuild: number;
  gpuBytesEnd: number;
  maps: number;
  profileKeys: string[] | null;
  buffers: Record<string, { size: number; usage: number; init?: string }>;
  events: string[];
}

const sha = (s: string): string => createHash('sha256').update(s).digest('hex');

function makeInput(cfg: TraceConfig): Record<string, unknown> {
  const rows = cfg.length * cfg.batch;
  const hidden = cfg.shape === 'julia-1' ? JULIA_SPEC.hiddenSize
    : GLINER_SPECS[cfg.shape].hiddenSize;
  const k = (cfg.shape === 'julia-1' ? JULIA_SPEC.options : cfg.markers) * cfg.batch;
  const input: Record<string, unknown> = {
    embeddings: cfg.f16 ? new Uint16Array(rows * hidden) : new Float32Array(rows * hidden),
    mask: new Float32Array(rows),
    packedMarkers: new Uint32Array(3 * k),
  };
  if (cfg.shape === 'julia-1') {
    input.qtype = cfg.batch === 1 ? 0 : [0, 1, 2, 0];
  }
  return input;
}

function canonical(log: RawEvent[]): { events: string[]; buffers: Trace['buffers']; maps: number } {
  const names = new Map<MockBuffer, string>();
  const buffers: Trace['buffers'] = {};
  const ref = (b: MockBuffer): string => {
    if (b.label.startsWith('w:')) return b.label;
    let n = names.get(b);
    if (n === undefined) {
      n = `b${names.size}`;
      names.set(b, n);
      const entry: Trace['buffers'][string] = { size: b.size, usage: b.usage };
      if (b.initParts.length) {
        const only = b.initParts.length === 1 && b.initParts[0].startsWith('0:');
        entry.init = only ? b.initParts[0].slice(2) : sha(b.initParts.join(';'));
      }
      buffers[n] = entry;
    }
    return n;
  };
  const pipeKey = (p: { code: string; constants: Record<string, number> }): string => {
    const keys = Object.keys(p.constants).sort();
    const consts = keys.map((k) => `${k}=${p.constants[k]}`).join(',');
    return `${sha(p.code).slice(0, 16)}|${consts}`;
  };
  const events: string[] = [];
  let maps = 0;
  for (const e of log) {
    switch (e.op) {
      case 'write':
        if (e.phase === 'run') events.push(JSON.stringify(['write', ref(e.buf), e.off, e.bytes]));
        break;
      case 'begin': events.push(JSON.stringify(e.ts ? ['begin', e.ts[0], e.ts[1]] : ['begin'])); break;
      case 'pipeline': events.push(JSON.stringify(['pipe', pipeKey(e.pipeline)])); break;
      case 'bind': {
        const bound = [...e.bg.entries].sort((a, b) => a.binding - b.binding)
          .map((x) => [x.binding, ref(x.buffer)]);
        events.push(JSON.stringify(['bind', e.group, bound]));
        break;
      }
      case 'dispatch': events.push(JSON.stringify(['dispatch', e.x, e.y])); break;
      case 'end': events.push('["end"]'); break;
      case 'copy':
        events.push(JSON.stringify(['copy', ref(e.src), e.so, ref(e.dst), e.do, e.size]));
        break;
      case 'resolve':
        events.push(JSON.stringify(['resolve', e.first, e.count, ref(e.dst), e.off]));
        break;
      case 'map': maps += 1; break;
    }
  }
  return { events, buffers, maps };
}

export async function recordTrace(cfg: TraceConfig, impl: Impl): Promise<Trace> {
  installGpuGlobals();
  const gpu = createMockGpu(['timestamp-query']);
  const tensors = gpu.weights(cfg.shape === 'julia-1' ? JULIA_ABSENT : []);
  const plan = impl.build(cfg, gpu.device, tensors);
  gpu.markBuilt();
  const gpuBytesBuild = plan.gpuBytes;
  const input = makeInput(cfg);
  let profileKeys: string[] | null = null;
  const stride = cfg.batch;
  switch (cfg.mode) {
    case 'normal':
    case 'normal100':
    case 'skip':
    case 'capture': {
      plan.upload(input as never);
      const seqLen = cfg.mode === 'normal' ? cfg.length * stride
        : cfg.mode === 'capture' ? 100 : 100 * stride;
      impl.submit(plan, cfg, {
        capture: cfg.mode === 'capture', seqLen,
        skip: cfg.mode === 'skip' ? new Set([1]) : undefined,
      });
      await plan.readLogits();
      if (cfg.mode === 'capture') await plan.readCapture();
      break;
    }
    case 'pass':
    case 'dispatch': {
      const r = await plan.profileForward(
        input as never, cfg.mode, cfg.mode === 'pass' ? cfg.length : 100);
      profileKeys = r ? Object.keys(r.times) : null;
      break;
    }
  }
  const { events, buffers, maps } = canonical(gpu.log);
  return { gpuBytesBuild, gpuBytesEnd: plan.gpuBytes, maps, profileKeys, buffers, events };
}

export function firstDifference(a: Trace, b: Trace): string | null {
  if (a.gpuBytesBuild !== b.gpuBytesBuild) {
    return `gpuBytesBuild frozen ${a.gpuBytesBuild}, now ${b.gpuBytesBuild}`;
  }
  if (a.gpuBytesEnd !== b.gpuBytesEnd) {
    return `gpuBytesEnd frozen ${a.gpuBytesEnd}, now ${b.gpuBytesEnd}`;
  }
  if (a.maps !== b.maps) return `mapAsync calls frozen ${a.maps}, now ${b.maps}`;
  if (JSON.stringify(a.profileKeys) !== JSON.stringify(b.profileKeys)) {
    return `profile keys frozen ${JSON.stringify(a.profileKeys)}, now ${JSON.stringify(b.profileKeys)}`;
  }
  const n = Math.max(a.events.length, b.events.length);
  for (let i = 0; i < n; i += 1) {
    if (a.events[i] !== b.events[i]) {
      return `event ${i}: frozen ${a.events[i] ?? '(end)'}, now ${b.events[i] ?? '(end)'}`;
    }
  }
  for (const key of new Set([...Object.keys(a.buffers), ...Object.keys(b.buffers)])) {
    if (JSON.stringify(a.buffers[key]) !== JSON.stringify(b.buffers[key])) {
      return `buffer ${key}: frozen ${JSON.stringify(a.buffers[key])}, now ${JSON.stringify(b.buffers[key])}`;
    }
  }
  return null;
}
