// Web Worker hosting the WASM-SIMD engine. Loads f32 weights and the
// tokenizer off the main thread and answers classify/runPrepared calls
// with the same result contract as the WebGPU Kleinhirn class.

import { WasmKleinhirn } from './wasm.ts';

let engine: WasmKleinhirn | null = null;

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      engine = await WasmKleinhirn.load(m.options);
      self.postMessage({ type: 'ready', id: m.id, info: engine.info() });
    } else if (m.type === 'classify') {
      if (!engine) throw new Error('engine not loaded');
      const result = await engine.classify(m.text, m.tasks);
      self.postMessage({ type: 'result', id: m.id, result });
    } else if (m.type === 'runPrepared') {
      if (!engine) throw new Error('engine not loaded');
      const result = await engine.runPrepared(m.input, m.capture, m.bucket);
      self.postMessage({ type: 'result', id: m.id, result });
    }
  } catch (err) {
    self.postMessage({
      type: 'error', id: m.id, message: err instanceof Error ? err.message : String(err),
    });
  }
};
