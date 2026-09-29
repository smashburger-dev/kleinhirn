// Port of interference_search/core.py (Apache-2.0, Bad Theory Labs): the
// Interference Search loop and the linear baseline. Behaviour, cost
// accounting and trace format match upstream exactly; propose/judge may be
// async so a kleinhirn engine can serve judge calls in batches.
import { PyRandom } from './pyrng.ts';

export type MaybePromise<T> = T | Promise<T>;

export interface Domain<S, P> {
  start(problem: P): S;
  propose(states: S[], problem: P): MaybePromise<[S[][], number]>;
  key(state: S): string;
  judge(states: S[], problem: P): MaybePromise<[number[] | null, number]>;
  isGoal(state: S, problem: P): boolean;
  isDead(state: S, problem: P): boolean;
}

export interface TraceRow {
  round: number;
  pool: number;
  raw: number;
  kept: number;
}

export interface Result<S> {
  solved: boolean;
  solution: S | null;
  cost: number;
  rounds: number;
  expanded: number;
  merged_away: number;
  trace: TraceRow[];
}

export interface SearchOptions {
  width?: number;
  merge?: boolean;
  restarts?: boolean;
}

export async function search<S, P>(
  domain: Domain<S, P>,
  problem: P,
  budget: number,
  { width = 6, merge = true, restarts = true }: SearchOptions = {},
): Promise<Result<S>> {
  let live = [domain.start(problem)];
  const expandedKeys = new Set<string>();
  const res: Result<S> = {
    solved: false, solution: null, cost: 0,
    rounds: 0, expanded: 0, merged_away: 0, trace: [],
  };
  while (res.cost < budget) {
    if (live.length === 0) {
      if (!restarts) break;
      live = [domain.start(problem)];
    }
    res.rounds += 1;
    const [proposals, pcost] = await domain.propose(live, problem);
    res.cost += pcost;
    res.expanded += live.length;
    for (const s of live) expandedKeys.add(domain.key(s));
    const pool = new Map<string, { c: S; votes: number }>();
    let raw = 0;
    for (const kids of proposals) {
      for (let rank = 0; rank < kids.length; rank++) {
        const c = kids[rank];
        raw += 1;
        if (domain.isGoal(c, problem)) {
          res.solved = true;
          res.solution = c;
          return res;
        }
        if (domain.isDead(c, problem)) continue;
        const k = merge ? domain.key(c) : `${domain.key(c)}${raw}`;
        if (merge && expandedKeys.has(k)) continue;
        const e = pool.get(k);
        if (e) e.votes += 1.0 / (rank + 1);
        else pool.set(k, { c, votes: 1.0 / (rank + 1) });
      }
    }
    res.merged_away += raw - pool.size;
    if (pool.size === 0) { live = []; continue; }
    const entries = [...pool.values()];
    const cands = entries.map((e) => e.c);
    const [judged, jcost] = await domain.judge(cands, problem);
    res.cost += jcost;
    const scores = judged ?? entries.map((e) => e.votes);
    const order = cands.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    live = order.slice(0, width).map((i) => cands[i]);
    res.trace.push({ round: res.rounds, pool: pool.size, raw, kept: live.length });
  }
  return res;
}

export interface LinearOptions {
  temp?: number;
  rng?: PyRandom;
}

// One chain at a time: judge the children, sample one, continue; when the
// chain dies, restart from the beginning until the budget runs out.
export async function linearSearch<S, P>(
  domain: Domain<S, P>,
  problem: P,
  budget: number,
  { temp = 0.5, rng = new PyRandom(0) }: LinearOptions = {},
): Promise<Result<S>> {
  const res: Result<S> = {
    solved: false, solution: null, cost: 0,
    rounds: 0, expanded: 0, merged_away: 0, trace: [],
  };
  while (res.cost < budget) {
    let s = domain.start(problem);
    while (res.cost < budget) {
      res.rounds += 1;
      res.expanded += 1;
      const [proposals, pcost] = await domain.propose([s], problem);
      let kids = proposals[0];
      res.cost += pcost;
      for (const c of kids) {
        if (domain.isGoal(c, problem)) {
          res.solved = true;
          res.solution = c;
          return res;
        }
      }
      kids = kids.filter((c) => !domain.isDead(c, problem));
      if (kids.length === 0) break;
      const [judged, jcost] = await domain.judge(kids, problem);
      res.cost += jcost;
      const scores = judged ?? kids.map((_, i) => 1.0 / (i + 1));
      const w = scores.map((x) => Math.exp(Math.log(Math.max(x, 1e-6)) / temp));
      s = rng.choices(kids, w)[0];
    }
  }
  return res;
}
