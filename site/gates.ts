// Parity gates per stage, identical to docs/PLAN.md K3 and bench/parity.ts:
// f32 and wasm are exact (argmax 100 %, max logit diff <= 1e-3), f16 needs
// argmax >= 99.5 % and max probability diff <= 1e-2. Shared by the page and
// tools/check_device_result.mjs so both apply the same numbers.

import type { ParitySummary } from '../bench/metrics.ts';

export type StageName = 'f16' | 'f32' | 'wasm';

export const STAGE_ORDER: StageName[] = ['f16', 'f32', 'wasm'];

export const GATES = {
  exact: { argmaxAgreementMin: 1, maxAbsLogitDiffMax: 1e-3 },
  f16: { argmaxAgreementMin: 0.995, maxAbsProbDiffMax: 1e-2 },
} as const;

export function gateRule(stage: StageName): 'exact' | 'f16' {
  return stage === 'f16' ? 'f16' : 'exact';
}

export function parityPasses(stage: StageName, s: ParitySummary): boolean {
  if (gateRule(stage) === 'f16') {
    return s.argmaxAgreement >= GATES.f16.argmaxAgreementMin
      && s.maxAbsProbDiff <= GATES.f16.maxAbsProbDiffMax;
  }
  return s.argmaxAgreement >= GATES.exact.argmaxAgreementMin
    && s.maxAbsLogitDiff <= GATES.exact.maxAbsLogitDiffMax;
}
