// CPU-only review helpers. No navigator, fetch, browser or real GPU is used.
import { readFileSync } from 'node:fs';
import { GpuBackend } from '../../src/backend.ts';
import { EncoderModel } from '../../src/encoder.ts';
import { buildPlan } from '../../src/plan/build.ts';
import { PlanExecutor } from '../../src/plan/executor.ts';
import { createMockGpu, installGpuGlobals } from './mock-gpu.ts';
import { pilotSpec } from './pilot.ts';

installGpuGlobals();
export const source = (path) => readFileSync(path, 'utf8');
export const fixture = (name) => JSON.parse(source(`tests/fixtures/hf-tokenizers/${name}.json`));
export const MIN = { binding: 134217728, buffer: 268435456, groups: 65535, shared: 16384 };

export function model(id = 'sentence-transformers/all-MiniLM-L6-v2', precision = 'f32', lengths = [128]) {
  const { spec, head, task } = pilotSpec(id);
  const gpu = createMockGpu();
  Object.assign(gpu.device.limits, { maxStorageBufferBindingSize: MIN.binding });
  const destroyed = new Set();
  const destroyedWrites = [];
  const write = gpu.device.queue.writeBuffer.bind(gpu.device.queue);
  gpu.device.queue.writeBuffer = (buffer, offset, data) => {
    if (destroyed.has(buffer)) destroyedWrites.push({ buffer, bytes: data.byteLength });
    write(buffer, offset, data);
  };
  const create = gpu.device.createBuffer.bind(gpu.device);
  gpu.device.createBuffer = (desc) => {
    const b = create(desc);
    b.destroy = () => destroyed.add(b);
    return b;
  };
  // Private TS constructor is callable in JS. Tiny CPU table, real plan construction.
  const embeddings = precision === 'f32' ? new Float32Array(8 * spec.embeddingSize)
    : new Uint16Array(8 * spec.embeddingSize);
  const weights = { tensors: gpu.weights(), embeddings, manifest: { tensors: [] }, gpuBytes: 0 };
  const engine = Reflect.construct(EncoderModel, [new GpuBackend({ device: gpu.device }, weights), embeddings, spec, head, task, precision]);
  for (const length of lengths) engine.plans.set(length, engine.makePlan(length, 1));
  return { engine, gpu, destroyed, destroyedWrites, spec, head };
}

export function planOf(id, length, batch, f16 = false) {
  const { spec, head } = pilotSpec(id);
  return buildPlan(spec, head, { length, batch, markers: 0, f16 });
}

export function executeMock(plan) {
  const gpu = createMockGpu();
  return { gpu, executor: new PlanExecutor(gpu.device, plan, gpu.weights()) };
}

// The scalar operations and the 64-thread trees of both LayerNorm kernels: the mean, then the mean
// of (v - mean)^2 over the row held in workgroup memory (R05).
export function normVariance(values) {
  const f = Math.fround;
  const tree = (part) => {
    for (let o = 32; o > 0; o >>= 1) for (let t = 0; t < o; t += 1) part[t] = f(part[t] + part[t + o]);
    return part[0];
  };
  const s = Array(64).fill(0), q = Array(64).fill(0);
  for (let t = 0; t < 64; t += 1) for (let i = t; i < values.length; i += 64) s[t] = f(s[t] + f(values[i]));
  const mean = f(tree(s) / values.length);
  for (let t = 0; t < 64; t += 1) {
    for (let i = t; i < values.length; i += 64) {
      const d = f(f(values[i]) - mean);
      q[t] = f(q[t] + f(d * d));
    }
  }
  return f(tree(q) / values.length);
}

export function valueEpilogue(scores, values) {
  // Exactly the j+3<L loop plus the L%4 remainder on acc0 in attention and mbattention, for one
  // output column.
  const acc = [0, 0, 0, 0];
  const L = scores.length;
  for (let j = 0; j + 3 < L; j += 4) {
    for (let k = 0; k < 4; k += 1) {
      if (scores[j + k] !== 0) acc[k] += scores[j + k] * values[j + k];
    }
  }
  for (let j = L - (L % 4); j < L; j += 1) if (scores[j] !== 0) acc[0] += scores[j] * values[j];
  return acc.reduce((a, b) => a + b, 0) / scores.reduce((a, b) => a + b, 0);
}
