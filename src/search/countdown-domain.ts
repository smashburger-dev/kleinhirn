// Port of interference_search/countdown_domain.py (Apache-2.0, Bad Theory
// Labs): Countdown as an Interference Search domain. The environment lists
// every legal move for free; cost is counted in judged states.
import { children } from './countdown.ts';
import type { CountdownProblem, State } from './countdown.ts';
import type { Domain, MaybePromise } from './core.ts';

// Scores one batch of states for the target; higher = more promising.
// The kleinhirn implementation will batch these calls per search round.
export type JudgeFn = (states: State[], target: number) => MaybePromise<number[]>;

export function countdown(judgeFn: JudgeFn): Domain<State, CountdownProblem> {
  return {
    start: (p) => [...p.numbers].sort((a, b) => a - b),
    // sorted(set(moves(s))) is exactly children(s)
    propose: (states) => [states.map((s) => children(s)), 0],
    key: (s) => s.join(','),
    judge: async (states, p) => [await judgeFn(states, p.target), states.length],
    isGoal: (s, p) => s.length === 1 && s[0] === p.target,
    isDead: (s, p) => s.length === 1 && s[0] !== p.target,
  };
}
