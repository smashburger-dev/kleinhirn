// Port of interference_search/countdown.py (Apache-2.0, Bad Theory Labs):
// the Countdown move rules, problem generation, exact solver and solution
// counter. States are sorted number arrays; the reference code uses sorted
// tuples, so every helper keeps elements ascending.
import type { PyRandom } from './pyrng.ts';

export const OPS = ['+', '-', '*', '/'] as const;
export type Op = (typeof OPS)[number];
export type State = number[];

interface Frac {
  n: number;
  d: number;
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

function frac(n: number, d = 1): Frac {
  const g = gcd(Math.abs(n), Math.abs(d)) || 1;
  n /= g; d /= g;
  if (d < 0) { n = -n; d = -d; }
  return { n, d };
}

const fadd = (a: Frac, b: Frac): Frac => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const fsub = (a: Frac, b: Frac): Frac => frac(a.n * b.d - b.n * a.d, a.d * b.d);
const fmul = (a: Frac, b: Frac): Frac => frac(a.n * b.n, a.d * b.d);
const fdiv = (a: Frac, b: Frac): Frac => frac(a.n * b.d, a.d * b.n);

function combine(a: Frac, b: Frac, op: Op): Frac | null {
  if (op === '+') return fadd(a, b);
  if (op === '-') return fsub(a, b);
  if (op === '*') return fmul(a, b);
  return b.n === 0 ? null : fdiv(a, b);
}

const stateKey = (s: readonly number[]): string => s.join(',');
const cmpState = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
};

// ---------- move rules and exact solving ----------

export function moves(state: State): State[] {
  const out: State[] = [];
  for (let i = 0; i < state.length; i++) {
    for (let j = 0; j < state.length; j++) {
      if (i === j) continue;
      const a = state[i], b = state[j];
      const rest = state.filter((_, k) => k !== i && k !== j);
      const results: number[] = [];
      if (i < j) results.push(a + b, a * b); // commutative once per unordered pair
      if (a > b) results.push(a - b);
      if (b > 1 && a % b === 0) results.push(a / b);
      for (const r of results) out.push([...rest, r].sort((x, y) => x - y));
    }
  }
  return out;
}

export function children(state: State): State[] {
  const seen = new Set<string>();
  const out: State[] = [];
  for (const m of moves(state)) {
    const k = stateKey(m);
    if (!seen.has(k)) { seen.add(k); out.push(m); }
  }
  return out.sort(cmpState);
}

export function applyMove(state: State, a: number, op: string, b: number): State | null {
  const s = state.slice();
  const ia = s.indexOf(a);
  if (ia < 0) return null;
  s.splice(ia, 1);
  const ib = s.indexOf(b);
  if (ib < 0) return null;
  s.splice(ib, 1);
  const norm = ({ x: '*', '×': '*', '÷': '/' } as Record<string, string>)[op] ?? op;
  let r: number;
  if (norm === '+') r = a + b;
  else if (norm === '*') r = a * b;
  else if (norm === '-') r = a - b;
  else {
    if (b === 0 || a % b) return null;
    r = a / b;
  }
  if (r <= 0) return null;
  return [...s, r].sort((x, y) => x - y);
}

export function solver(target: number): (state: State) => boolean {
  const memo = new Map<string, boolean>();
  const solvable = (state: State): boolean => {
    if (state.length === 1) return state[0] === target;
    const k = stateKey(state);
    const hit = memo.get(k);
    if (hit !== undefined) return hit;
    const v = children(state).some(solvable);
    memo.set(k, v);
    return v;
  };
  return solvable;
}

export function reachableStates(numbers: readonly number[]): State[] {
  const seen = new Set<string>();
  const out: State[] = [];
  const stack: State[] = [[...numbers].sort((a, b) => a - b)];
  while (stack.length) {
    const s = stack.pop()!;
    const k = stateKey(s);
    if (seen.has(k) || s.length === 1) continue;
    seen.add(k);
    out.push(s);
    stack.push(...moves(s));
  }
  return out.sort(cmpState);
}

// ---------- expression trees and canonical forms ----------

type ENode = { v: Frac } & ({ leaf: number } | { op: Op; l: ENode; r: ENode });

function flatExpr(e: ENode, kind: 'add' | 'mul', out: [number, ENode][]): void {
  const inKind = 'op' in e &&
    (kind === 'add' ? e.op === '+' || e.op === '-' : e.op === '*' || e.op === '/');
  if (!inKind || !('op' in e)) { out.push([1, e]); return; }
  flatExpr(e.l, kind, out);
  const right: [number, ENode][] = [];
  flatExpr(e.r, kind, right);
  const neg = kind === 'add' ? e.op === '-' : e.op === '/';
  for (const [s, n] of right) out.push([neg ? -s : s, n]);
}

// Canonical string identical to countdown.py's _Canon output: the kind,
// then signed operand strings sorted inside parentheses.
function canon(e: ENode): string {
  if ('leaf' in e) return String(e.leaf);
  const kind = e.op === '+' || e.op === '-' ? 'add' : 'mul';
  const parts: [number, ENode][] = [];
  flatExpr(e, kind, parts);
  const strs = parts.map(([sgn, sub]) => {
    if (kind === 'mul' && sgn < 0 && sub.v.n === 0) throw new Error('zero division');
    return (kind === 'add' ? (sgn > 0 ? '+' : '-') : (sgn > 0 ? '*' : '/')) + canon(sub);
  });
  strs.sort();
  return `${kind}(${strs.join(',')})`;
}

export function countSolutions(numbers: readonly number[], target: number): number {
  const sols = new Set<string>();
  const rec = (items: ENode[]): void => {
    if (items.length === 1) {
      const it = items[0];
      if (it.v.d === 1 && it.v.n === target) {
        try { sols.add(canon(it)); } catch { /* unparseable, e.g. divide by zero */ }
      }
      return;
    }
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        for (const op of OPS) {
          if ((op === '+' || op === '*') && i > j) continue;
          const v = combine(items[i].v, items[j].v, op);
          if (v === null) continue;
          const rest = items.filter((_, k) => k !== i && k !== j);
          rest.push({ v, op, l: items[i], r: items[j] });
          rec(rest);
        }
      }
    }
  };
  rec(numbers.map((n) => ({ v: frac(n), leaf: n })));
  return sols.size;
}

// ---------- problem generation ----------

export interface CountdownProblem {
  numbers: number[];
  target: number;
  solution: string;
  n_solutions?: number;
}

export function genProblem(rng: PyRandom, nNumbers = 4): CountdownProblem {
  for (;;) {
    const nums: number[] = [];
    for (let i = 0; i < nNumbers - 1; i++) nums.push(rng.randint(1, 25));
    nums.push(rng.randint(25, 100));
    rng.shuffle(nums);
    let items = nums.map((n) => ({ v: frac(n), e: String(n) }));
    let ok = true;
    while (items.length > 1) {
      const [i, j] = rng.sample(items.map((_, k) => k), 2);
      const { v: va, e: ea } = items[i];
      const { v: vb, e: eb } = items[j];
      const op = rng.choice(OPS) as Op;
      const v = combine(va, vb, op);
      if (v === null || v.d !== 1 || v.n <= 0) { ok = false; break; }
      items = items.filter((_, k) => k !== i && k !== j);
      items.push({ v, e: `(${ea} ${op} ${eb})` });
    }
    if (!ok) continue;
    const target = items[0].v.n;
    if (target >= 10 && target <= 999 && !nums.includes(target)) {
      return { numbers: nums, target, solution: items[0].e.slice(1, -1) };
    }
  }
}

export function genHardProblem(rng: PyRandom, maxSolutions = 2): CountdownProblem {
  for (;;) {
    const p = genProblem(rng);
    const { numbers: nums, target: t } = p;
    let anchored = false;
    for (let i = 0; i < nums.length && !anchored; i++) {
      for (let j = 0; j < nums.length && !anchored; j++) {
        if (i === j) continue;
        for (const op of OPS) {
          const v = combine(frac(nums[i]), frac(nums[j]), op);
          // abs(v - t) <= 25, all in exact rationals
          if (v && Math.abs(v.n - t * v.d) <= 25 * v.d) { anchored = true; break; }
        }
      }
    }
    if (anchored) continue;
    const n = countSolutions(nums, t);
    if (n >= 1 && n <= maxSolutions) {
      p.n_solutions = n;
      return p;
    }
  }
}
