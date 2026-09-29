// K17 gates for the Interference Search port. CPU-only, no model files.
// The fixture was produced by tools/gen_search_fixture.py running upstream
// interference-search code, so every comparison is TS vs upstream, not TS
// vs a paraphrase.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PyRandom } from '../src/search/pyrng.ts';
import {
  children, countSolutions, genHardProblem, genProblem, moves,
  reachableStates, solver,
} from '../src/search/countdown.ts';
import { linearSearch, search } from '../src/search/core.ts';
import { countdown } from '../src/search/countdown-domain.ts';
import { encode, N_FEAT, numFeats } from '../src/search/judge.ts';

const fx = JSON.parse(readFileSync('tests/golden/search/countdown-fixture.json', 'utf8'));
const judgeFx = JSON.parse(readFileSync('tests/golden/search/judge-parity.json', 'utf8'));

interface Problem {
  numbers: number[];
  target: number;
  solution: string;
  n_solutions?: number;
}

// Same dynamic program as upstream experiments/countdown/compression.py.
function analyze(numbers: number[], target: number) {
  const solvable = solver(target);
  const memo = new Map<string, number>();
  const completions = (s: number[]): number => {
    if (s.length === 1) return 1;
    const k = s.join(',');
    const hit = memo.get(k);
    if (hit !== undefined) return hit;
    const v = moves(s).reduce((a, n) => a + completions(n), 0);
    memo.set(k, v);
    return v;
  };
  const rows: { depth: number; paths: number; states: number; dead_states: number;
    full_paths_through_dead: number; full_paths: number }[] = [];
  let frontier = new Map<string, { s: number[]; c: number }>();
  const start = [...numbers].sort((a, b) => a - b);
  frontier.set(start.join(','), { s: start, c: 1 });
  for (let depth = 0; ; depth++) {
    const entries = [...frontier.values()];
    const dead = entries.filter((e) => !solvable(e.s));
    rows.push({
      depth,
      paths: entries.reduce((a, e) => a + e.c, 0),
      states: entries.length,
      dead_states: dead.length,
      full_paths_through_dead: dead.reduce((a, e) => a + e.c * completions(e.s), 0),
      full_paths: entries.reduce((a, e) => a + e.c * completions(e.s), 0),
    });
    if (entries.every((e) => e.s.length === 1)) return rows;
    const nxt = new Map<string, { s: number[]; c: number }>();
    for (const e of entries) {
      for (const n of moves(e.s)) {
        const k = n.join(',');
        const ex = nxt.get(k);
        if (ex) ex.c += e.c; else nxt.set(k, { s: n, c: e.c });
      }
    }
    frontier = nxt;
  }
}

// The exact table published in upstream README/results (printf-rounded means
// over 20 instances per size). Keys: n_numbers -> depth -> [paths, states].
const PUBLISHED: Record<number, Record<number, [number, number]>> = {
  4: { 0: [1, 1], 1: [19, 18], 2: [179, 97], 3: [566, 159] },
  5: { 0: [1, 1], 1: [31, 28], 2: [589, 278], 3: [5534, 1159], 4: [17266, 1511] },
  6: {
    0: [1, 1], 1: [47, 43], 2: [1475, 704], 3: [27915, 5262],
    4: [264496, 16479], 5: [831176, 13229],
  },
};

test('pyrng: draws match Python random.Random', () => {
  // Reference values produced by CPython 3.12 on the upstream repo.
  let r = new PyRandom(0);
  assert.deepEqual([...Array(6)].map(() => r.randint(1, 25)), [13, 25, 14, 2, 9, 17]);
  r = new PyRandom(0);
  assert.equal(r.randint(25, 100), 74);
  r = new PyRandom(0);
  const x = [3, 1, 4, 1, 5];
  r.shuffle(x);
  assert.deepEqual(x, [4, 1, 3, 5, 1]);
  r = new PyRandom(0);
  assert.deepEqual(r.sample([0, 1, 2, 3, 4, 5], 2), [3, 5]);
  assert.deepEqual(r.sample([0, 1, 2, 3, 4], 2), [0, 2]);
  assert.deepEqual(r.sample([0, 1, 2], 2), [2, 1]);
  r = new PyRandom(0);
  assert.deepEqual([...Array(6)].map(() => r.choice(['+', '-', '*', '/'])),
    ['/', '/', '+', '*', '/', '/']);
  r = new PyRandom(0);
  assert.deepEqual([...Array(4)].map(() => r.random()),
    [0.8444218515250481, 0.7579544029403025, 0.420571580830845, 0.25891675029296335]);
  r = new PyRandom(0);
  assert.deepEqual(r.choices([10, 20, 30], [0.1, 1.0, 0.5], 5), [30, 30, 20, 20, 20]);
  r = new PyRandom(11);
  assert.deepEqual([...Array(5)].map(() => r.randint(1, 100)), [58, 72, 100, 60, 58]);
});

test('countdown: genProblem regenerates the fixture instances exactly', () => {
  const rng = new PyRandom(0);
  for (const n of [4, 5, 6]) {
    for (let i = 0; i < 20; i++) {
      const want = fx.compression_problems
        .filter((p: Problem & { n_numbers: number }) => p.n_numbers === n)[i];
      const got = genProblem(rng, n);
      assert.equal(got.target, want.target);
      assert.deepEqual(got.numbers, want.numbers);
      assert.equal(got.solution, want.solution);
    }
  }
});

test('countdown: genHardProblem regenerates the 30 hard problems', () => {
  const rng = new PyRandom(11);
  for (const want of fx.hard_problems as Problem[]) {
    const got = genHardProblem(rng);
    assert.equal(got.target, want.target);
    assert.deepEqual(got.numbers, want.numbers);
    assert.equal(got.solution, want.solution);
    assert.equal(got.n_solutions, want.n_solutions);
  }
});

test('countdown: compression analysis matches upstream row for row', () => {
  for (const p of fx.compression_problems as Problem[]) {
    const rows = analyze(p.numbers, p.target);
    assert.deepEqual(rows, (p as any).analyze, `analyze mismatch ${p.numbers}`);
  }
});

test('countdown: reproduced compression means equal the published table', () => {
  for (const n of [4, 5, 6]) {
    const probs = (fx.compression_problems as (Problem & { n_numbers: number })[])
      .filter((p) => p.n_numbers === n);
    const maxDepth = Math.max(...probs.map((p) => (p as any).analyze.length - 1));
    for (let d = 0; d <= maxDepth; d++) {
      const paths = probs.reduce((a, p) => a + (p as any).analyze[d].paths, 0) / 20;
      const states = probs.reduce((a, p) => a + (p as any).analyze[d].states, 0) / 20;
      const [pubPaths, pubStates] = PUBLISHED[n][d];
      assert.ok(Math.abs(paths - pubPaths) <= 0.5, `n=${n} d=${d} paths ${paths} vs ${pubPaths}`);
      assert.ok(Math.abs(states - pubStates) <= 0.5, `n=${n} d=${d} states ${states} vs ${pubStates}`);
    }
  }
});

test('countdown: exact solver agrees with upstream on all reachable states', () => {
  let checked = 0;
  for (const p of fx.hard_problems) {
    const solvable = solver(p.target);
    for (const { s, alive } of p.states as { s: number[]; alive: boolean }[]) {
      assert.equal(solvable(s), alive, `state ${s} of ${p.numbers}->${p.target}`);
      checked++;
    }
    // also covers the start state itself and reachable-state enumeration
    assert.equal(solvable([...p.numbers].sort((a, b) => a - b)), true);
    assert.equal(reachableStates(p.numbers).length, p.n_states);
  }
  assert.equal(checked, 3517);
});

test('search: merge and no-merge runs reproduce upstream results exactly', async () => {
  for (const p of fx.hard_problems as Problem[]) {
    const solvable = solver(p.target);
    const dom = countdown((states) => states.map((s) => (solvable(s) ? 1.0 : 0.0)));
    const prob = { numbers: p.numbers, target: p.target, solution: '' };
    for (const merge of [true, false]) {
      const r = await search(dom, prob, 200, { width: 6, merge });
      const want = (p as any).search[`200_w6_m${merge ? 1 : 0}`];
      assert.equal(r.solved, want.solved, `merge=${merge} ${p.numbers}`);
      assert.equal(r.cost, want.cost);
      assert.equal(r.rounds, want.rounds);
      assert.equal(r.expanded, want.expanded);
      assert.equal(r.merged_away, want.merged_away);
      assert.deepEqual(r.trace, want.trace);
      if (r.solved) assert.deepEqual(r.solution, [p.target]);
    }
  }
});

test('linearSearch: seeded chains reproduce upstream results exactly', async () => {
  for (const p of fx.hard_problems as Problem[]) {
    const solvable = solver(p.target);
    const dom = countdown((states) => states.map((s) => (solvable(s) ? 1.0 : 0.0)));
    const r = await linearSearch(dom, { numbers: p.numbers, target: p.target, solution: '' },
      200, { rng: new PyRandom(0) });
    const want = (p as any).linear;
    assert.equal(r.solved, want.solved, `linear ${p.numbers}`);
    assert.equal(r.cost, want.cost);
    assert.equal(r.rounds, want.rounds);
    assert.equal(r.expanded, want.expanded);
  }
});

test('countdown: move rules and domain glue', () => {
  // (1/1) intermediate states, unordered-pair dedup inside moves
  const m = moves([2, 3]);
  assert.ok(m.some((s) => s.join(',') === '5'));
  assert.ok(m.some((s) => s.join(',') === '6'));
  assert.ok(m.some((s) => s.join(',') === '1'));
  assert.ok(!m.some((s) => s.join(',') === '1.5'));
  const ch = children([5, 5, 5]);
  const keys = new Set(ch.map((s) => s.join(',')));
  assert.equal(keys.size, ch.length); // children are distinct
  assert.ok(ch.every((s) => s.length === 2));
});

test('judge: numFeats equals upstream num_feats (exact modulo libm log1p ulps)', () => {
  // Positions 2..28 are integer/boolean features and must be bit-exact.
  // Positions 0..1 derive from log1p: V8's fdlibm Math.log1p and the macOS
  // libm behind CPython differ by at most 1 f64 ulp (measured: 2.22e-16 on
  // 284/3756 vectors), so those two get a 1e-15 bound. Everything else: 0.
  let maxLog = 0;
  for (const { x, t, f } of judgeFx.feats as { x: number; t: number; f: number[] }[]) {
    const g = numFeats(x, t);
    for (let i = 2; i < 29; i++) assert.equal(g[i], f[i], `numFeats(${x}, ${t})[${i}]`);
    for (const i of [0, 1]) {
      const d = Math.abs(g[i] - f[i]);
      if (d > maxLog) maxLog = d;
      assert.ok(d <= 1e-15, `numFeats(${x}, ${t})[${i}] off by ${d}`);
    }
  }
  assert.equal(judgeFx.feats.length, 3756);
  assert.ok(maxLog <= 2.3e-16, `log1p drift grew beyond 1 ulp: ${maxLog}`);
});

test('judge scaffold: feature encoding matches upstream judge.py', () => {
  assert.equal(N_FEAT, 29); // 7 scalar + 11 residue-equal + 11 residue-norm
  const f = numFeats(10, 4);
  assert.equal(f[2], 0); // x == t
  assert.equal(f[3], 1); // x > t
  assert.equal(f[4], 0); // t % x = 4 % 10 = 4
  assert.equal(f[5], 0); // x % t = 10 % 4 = 2
  const { feats, mask } = encode([[3, 5], [1, 4, 2, 9]], 7);
  assert.equal(feats.length, 2);
  assert.equal(feats[0].length, 7);
  assert.equal(feats[0][0].length, N_FEAT);
  assert.deepEqual(mask[0], [true, true, false, false, false, false, false]);
  assert.deepEqual(mask[1], [true, true, true, true, false, false, false]);
  assert.ok(feats[0][6].every((v) => v === 0)); // pad row
});
