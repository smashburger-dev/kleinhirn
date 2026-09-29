// Julia 1 comparison bench (K8-6): the same 100 parity-cases requests, one
// at a time, timing boundary = before encode to probabilities in JS. Same
// structure as models/julia-1/onnx/benchmark-webgpu.html: load+warm, a
// parity pass over all 100, then 5 timed iterations.
// Query: ?precision=f16|f32&buckets=512,1024&limits=minimum|default

import { JuliaEngine } from '../src/julia.ts';
import type { JuliaRequest } from '../src/tokenizer/julia-input.ts';

interface Case {
  request: JuliaRequest;
  pytorch_logits: number[];
}

declare global {
  interface Window { khJuliaBench?: Record<string, unknown> }
}

function argmaxOf(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const precision = params.get('precision') ?? 'f16';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const buckets = params.get('buckets')?.split(',').map(Number) ?? [512, 1024];
  const out: Record<string, unknown> = { stage: 'boot', precision };
  window.khJuliaBench = out;
  try {
    const t0 = performance.now();
    const kh = await JuliaEngine.load({
      manifestUrl: `/models/julia-1/${precision}/manifest.json`,
      buckets,
      precision: 'auto',
      limits,
      // The timed pass repeats the parity items; caching would serve hits
      // instead of computing, so benchmark pages disable the K16 cache.
      cacheSize: 0,
    });
    const loadMs = performance.now() - t0;
    out.info = { ...kh.info(), loadMs };

    const cases = (await (await fetch(
      '/models/julia-1/onnx/parity-cases.json')).json()) as Case[];

    // Parity pass doubles as warmup, like run(true) in their benchmark.
    out.stage = 'parity';
    let matches = 0;
    let maxError = 0;
    for (const item of cases) {
      const res = await kh.decide(item.request);
      const expected = item.pytorch_logits;
      if (argmaxOf(res.logits) === argmaxOf(expected)) matches += 1;
      for (let j = 0; j < expected.length; j += 1) {
        maxError = Math.max(maxError, Math.abs(res.logits[j] - expected[j]));
      }
    }
    const loadMsWarm = performance.now() - t0;
    out.loadMsWarm = loadMsWarm;

    out.stage = 'timed';
    const iterations: number[] = [];
    const perRequest: number[] = [];
    for (let it = 0; it < 5; it += 1) {
      const s = performance.now();
      for (const item of cases) {
        const t = performance.now();
        const res = await kh.decide(item.request);
        res.probabilities[0]; // boundary: probabilities materialized in JS
        perRequest.push(performance.now() - t);
      }
      iterations.push(performance.now() - s);
      out.stage = `timed ${it + 1}/5`;
    }
    const sorted = (a: number[]) => [...a].sort((x, y) => x - y);
    const it = sorted(iterations);
    const rq = sorted(perRequest);
    const pick = (arr: number[], q: number) =>
      arr[Math.min(arr.length - 1, Math.floor(q * arr.length))];
    out.result = {
      requests: cases.length, batch_size: 1, iterations: 5,
      load_ms: loadMs, load_and_warm_ms: loadMsWarm,
      median_ms: it[2], min_ms: it[0], max_ms: it[4],
      ms_per_request: it[2] / cases.length,
      per_request_median_ms: rq[Math.floor(rq.length / 2)],
      per_request_p95_ms: pick(rq, 0.95),
      matching_predictions: matches,
      max_abs_logit_error_vs_pytorch: maxError,
    };
    out.stage = 'done';
    out.done = true;
    kh.dispose();
  } catch (e) {
    out.stage = 'error';
    out.error = String(e);
    out.done = true;
  }
}

void main();
