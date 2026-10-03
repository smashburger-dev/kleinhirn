// CPU-only review helpers. No navigator, fetch, browser or real GPU is used.
import { readFileSync } from 'node:fs';
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
  const weights = { tensors: gpu.weights(), embeddings, manifest: { tensors: [] } };
  const engine = Reflect.construct(EncoderModel, [{ device: gpu.device }, weights, spec, head, task, precision]);
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

// The scalar operations and the 64-thread tree of both LayerNorm kernels.
export function normVariance(values) {
  const f = Math.fround;
  const s = Array(64).fill(0), q = Array(64).fill(0);
  for (let t = 0; t < 64; t += 1) {
    for (let i = t; i < values.length; i += 64) {
      const v = f(values[i]);
      s[t] = f(s[t] + v);
      q[t] = f(q[t] + f(v * v));
    }
  }
  for (let o = 32; o > 0; o >>= 1) {
    for (let t = 0; t < o; t += 1) {
      s[t] = f(s[t] + s[t + o]);
      q[t] = f(q[t] + q[t + o]);
    }
  }
  const mean = f(s[0] / values.length);
  return f(f(q[0] / values.length) - f(mean * mean));
}

export function valueEpilogue(scores, values) {
  // Exactly the j+3<L loop in attention and mbattention, for one output column.
  const acc = [0, 0, 0, 0];
  for (let j = 0; j + 3 < scores.length; j += 4) {
    for (let k = 0; k < 4; k += 1) {
      if (scores[j + k] !== 0) acc[k] += scores[j + k] * values[j + k];
    }
  }
  return acc.reduce((a, b) => a + b, 0) / scores.reduce((a, b) => a + b, 0);
}
