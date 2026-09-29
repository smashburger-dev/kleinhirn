// K17 judge gates in the browser. Two modes:
//   mode=parity  WebGPU judge logits against the numpy/PyTorch goldens
//                (tests/golden/search/judge-parity.json, 4,517 states).
//                f32: max |logit| diff <= 1e-5 against logit64 and 100 %
//                alive/dead decision agreement. f16: 100 % agreement.
//   mode=search  the 30 hard countdown tasks through `search` with the
//                WebGPU judge (width 6, budget 200): gate 30/30 like the
//                published run; per-task wall time is reported as a
//                provisional number (loadavg recorded by the runner).
// Query: ?mode=parity|search&precision=f32|f16&limits=minimum|default

import { JudgeEngine } from '../src/search/judge-engine.ts';
import { countdown } from '../src/search/countdown-domain.ts';
import { search } from '../src/search/core.ts';

interface ParityItem { s: number[]; t: number; logit: number; logit64: number }
interface HardProblem {
  numbers: number[]; target: number; solution: string; n_solutions: number;
}

interface Result {
  stage: string;
  mode?: string;
  precision?: string;
  info?: unknown;
  adapterInfo?: unknown;
  n?: number;
  parity?: {
    maxAbsProbDiff: number;
    maxAbsLogitDiff: number;
    decisionAgreement: number;
    saturatedSkipped: number;
  };
  tasks?: {
    solved: number; total: number;
    msPerTask: { median: number; min: number; max: number };
    perTask: { target: number; solved: boolean; cost: number; ms: number }[];
  };
  gates?: Record<string, boolean>;
  debugFirst?: { t: number; s: number[]; p: number; ref: number }[];
  debugBufs?: Record<string, number[]>;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khJudgeResult?: Result }
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const logitOf = (p: number) => Math.log(p / (1 - p));

async function parity(
  engine: JudgeEngine, precision: string, result: Result,
): Promise<void> {
  const golden = (await (await fetch(
    '/tests/golden/search/judge-parity.json')).json()) as {
      items: ParityItem[] };
  // JudgeFn scores states against one target; group golden items by t so
  // each score() call exercises the batch path.
  const byTarget = new Map<number, ParityItem[]>();
  for (const it of golden.items) {
    const l = byTarget.get(it.t) ?? [];
    l.push(it);
    byTarget.set(it.t, l);
  }
  let maxProb = 0;
  let maxLogit = 0;
  let agree = 0;
  let skipped = 0;
  let n = 0;
  let done = 0;
  const first: { t: number; s: number[]; p: number; ref: number }[] = [];
  for (const [t, items] of byTarget) {
    const states = items.map((it) => it.s);
    const probs = await engine.score(states, t);
    for (const [i, it] of items.entries()) {
      const p = probs[i];
      if (first.length < 8) {
        first.push({ t, s: it.s, p, ref: sigmoid(it.logit64) });
      }
      const ref64 = sigmoid(it.logit64);
      maxProb = Math.max(maxProb, Math.abs(p - ref64));
      // Inverting the sigmoid only works away from saturation; outside
      // that range the decision bit is the meaningful comparison anyway.
      if (p > 1e-9 && p < 1 - 1e-9) {
        maxLogit = Math.max(maxLogit, Math.abs(logitOf(p) - it.logit64));
      } else skipped += 1;
      if ((p > 0.5) === (it.logit > 0)) agree += 1;
      n += 1;
    }
    done += items.length;
    result.stage = `parity ${done}/${golden.items.length}`;
  }
  result.n = n;
  result.parity = {
    maxAbsProbDiff: maxProb,
    maxAbsLogitDiff: maxLogit,
    decisionAgreement: agree / n,
    saturatedSkipped: skipped,
  };
  result.debugFirst = first;
  result.gates = precision === 'f32'
    ? {
      'logit-1e-5': maxLogit <= 1e-5,
      'prob-1e-5': maxProb <= 1e-5,
      'decision-100': agree === n,
    }
    : { 'decision-100': agree === n };
}

async function debugDump(
  engine: JudgeEngine, result: Result,
): Promise<void> {
  const probs = await engine.score(
    [[3, 12, 17], [3, 5], [3, 29], [3, 48], [3, 72], [3, 204], [3, 720],
      [3, 12, 60], [5], [12], [17], [60], [29], [48], [72], [204]], 684);
  const plans = (engine as unknown as {
    plans: Map<number, {
      debugRead: (n: string) => Promise<ArrayBuffer>;
    }>;
  }).plans;
  const plan = plans.get(16)!;
  const names = ['feats', 'maskBuf', 'x', 'normed', 'qkv', 'ctx',
    'clsRows', 'kNormed', 'raw'] as const;
  const bufs: Record<string, number[]> = {};
  for (const n of names) {
    const b = await plan.debugRead(n);
    bufs[n] = Array.from(new Float32Array(b));
  }
  result.debugFirst = probs.map((p, i) => ({
    t: 684, s: [3, 12, 17], p, ref: i,
  }));
  result.debugBufs = bufs;
}

async function countdownGate(
  engine: JudgeEngine, result: Result,
): Promise<void> {
  const fx = (await (await fetch(
    '/tests/golden/search/countdown-fixture.json')).json()) as {
      hard_problems: HardProblem[] };
  const domain = countdown(engine.score);
  const perTask: { target: number; solved: boolean; cost: number; ms: number }[] = [];
  for (const [i, problem] of fx.hard_problems.entries()) {
    const t0 = performance.now();
    const r = await search(domain, problem, 200, { width: 6 });
    const ms = performance.now() - t0;
    perTask.push({ target: problem.target, solved: r.solved, cost: r.cost, ms });
    result.stage = `task ${i + 1}/30 ${r.solved ? 'solved' : 'FAILED'}`;
  }
  const times = perTask.map((t) => t.ms).sort((a, b) => a - b);
  const solved = perTask.filter((t) => t.solved).length;
  result.tasks = {
    solved, total: perTask.length,
    msPerTask: {
      median: times[Math.floor(times.length / 2)],
      min: times[0], max: times[times.length - 1],
    },
    perTask,
  };
  result.gates = { 'solved-30-30': solved === 30 };
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const mode = params.get('mode') ?? 'parity';
  const precision = params.get('precision') === 'f16' ? 'f16' : 'f32';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const result: Result = { stage: 'boot', mode, precision };
  window.khJudgeResult = result;
  try {
    const engine = await JudgeEngine.load({
      manifestUrl: `/models/countdown-judge/${precision}/manifest.json`,
      precision, limits,
    });
    result.adapterInfo = null;
    if (mode === 'search') {
      await countdownGate(engine, result);
    } else if (mode === 'debug') {
      await debugDump(engine, result);
    } else {
      await parity(engine, precision, result);
    }
    result.info = {
      gpuBytes: engine.gpuBytes,
      downloadBytes: engine.downloadBytes,
      precision: engine.precision,
    };
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
