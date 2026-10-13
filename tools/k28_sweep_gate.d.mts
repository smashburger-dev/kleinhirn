// Types for tools/k28_sweep_gate.mjs (plain JS), used by tests/k28_sweep_gate.test.ts.
export interface GateInput {
  task: string;
  forward?: unknown;
  f64?: unknown;
  p32?: unknown;
  p16?: unknown;
  text?: unknown;
  close?: unknown;
  sim?: unknown;
}
export interface GateResult {
  pass: boolean;
  reasons: string[];
  recommendedPrecision?: 'f32';
}
export function evaluateGate(input: GateInput): GateResult;
export function missedOnlyLogitTolerance(forward: unknown): boolean;
export const SPAN_SCORE_TOL: number;
export function spanScoreTolerance(forward: unknown, f64: unknown): number;
