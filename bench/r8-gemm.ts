// R8 matmul micro-bench page (docs/R8_WORKORDER.md Phase 0 step 3, Phase 1): fetches the two
// builds of src/wasm/gemm-probe.ts and tools/ff-jit/loops.wasm, runs them in a worker
// (bench/r8-gemm-worker.ts) and stores the result in window.khR8Gemm. Query: [khpost=<url>]

import { isolation } from './k28/k28-8-common.ts';
import { startPosting } from './kbench/post.ts';

interface Result { stage: string; done?: boolean; error?: string; ua?: string; relaxedValid?: boolean; [k: string]: unknown }

declare global {
  interface Window { khR8Gemm?: Result }
}

// one function: three v128.const and f32x4.relaxed_madd (the probe of bench/kbench.ts)
function relaxedValid(): boolean {
  const c = [0xfd, 0x0c, ...new Array<number>(16).fill(0)];
  const body = [0, ...c, ...c, ...c, 0xfd, 0x85, 0x02, 0x0b];
  return WebAssembly.validate(new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 1, 5, 1, 0x60, 0, 1, 0x7b,
    3, 2, 1, 0, 10, body.length + 2, 1, body.length, ...body]));
}

async function main(): Promise<void> {
  const result: Result = { stage: 'boot', ua: navigator.userAgent };
  window.khR8Gemm = result;
  startPosting(() => window.khR8Gemm);
  try {
    Object.assign(result, isolation());
    result.relaxedValid = relaxedValid();
    const get = async (u: string): Promise<ArrayBuffer> => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); };
    const [plain, relaxed, loops] = await Promise.all([get('/src/wasm/gemm-probe.wasm'),
      result.relaxedValid ? get('/src/wasm/gemm-probe-relaxed.wasm') : Promise.resolve(null), get('/tools/ff-jit/loops.wasm')]);
    result.stage = 'running';
    const w = new Worker(new URL('./r8-gemm-worker.ts', import.meta.url), { type: 'module' });
    const out = await new Promise<Record<string, unknown>>((res, rej) => {
      w.onmessage = (e) => res(e.data as Record<string, unknown>);
      w.onerror = (e) => rej(new Error(e.message));
      w.postMessage({ plain, relaxed, loops });
    });
    w.terminate();
    if (out.error) throw new Error(String(out.error));
    Object.assign(result, out);
    result.stage = 'done';
    result.done = true;
  } catch (e) {
    result.stage = 'error';
    result.error = String(e);
    result.done = true;
  }
}

void main();
