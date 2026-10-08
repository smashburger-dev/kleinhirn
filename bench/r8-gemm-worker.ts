// Worker of the R8 matmul micro-bench (bench/r8-gemm.ts): the probe variants on the plain build,
// on the relaxed build where the browser validates it, and the loops of tools/r2_ff_jit.mjs as
// the clock check of the runner (FINDINGS 54: about 112 ms on an optimizing tier).

import { runProbe, type ProbeExports } from './r8-gemm-core.ts';

const imports = { env: { abort() { throw new Error('abort'); } } };

function loops(bytes: ArrayBuffer): Record<string, number> {
  const ex = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports).exports as unknown as {
    scalar(n: number): number; simd(n: number): number;
  };
  const n = 50_000_000;
  const time = (f: () => unknown): number => {
    const t: number[] = [];
    for (let r = 0; r < 4; r += 1) { const t0 = performance.now(); f(); t.push(performance.now() - t0); }
    t.shift();
    t.sort((a, b) => a - b);
    return t[1];
  };
  return { wasmScalarMs: time(() => ex.scalar(n)), wasmSimdMs: time(() => ex.simd(n)) };
}

onmessage = (e: MessageEvent<{ plain: ArrayBuffer; relaxed: ArrayBuffer | null; loops: ArrayBuffer }>) => {
  try {
    const inst = (b: ArrayBuffer): ProbeExports => new WebAssembly.Instance(new WebAssembly.Module(b), imports).exports as unknown as ProbeExports;
    const now = (): number => performance.now();
    const loopMs = loops(e.data.loops);
    const plain = runProbe(inst(e.data.plain), now);
    const relaxed = e.data.relaxed ? runProbe(inst(e.data.relaxed), now, 5, ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']) : null;
    postMessage({ loops: loopMs, plain, relaxed });
  } catch (err) {
    postMessage({ error: String(err) });
  }
};
