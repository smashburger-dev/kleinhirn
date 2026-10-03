// Plan operations against kernel text: the number of `bind` entries of an
// operation must equal the number of @binding declarations of its kernel.
// The mock GPU checks that every bound index exists but not that all of them
// are bound, and WebGPU reports a missing entry as a validation error, not an
// exception.

import { KERNELS } from '../../src/kernels/index.ts';
import type { Plan } from '../../src/plan/ir.ts';

export function declaredBindings(kernel: keyof typeof KERNELS): number {
  return (KERNELS[kernel].match(/@group\(0\)\s*@binding\(\d+\)/g) ?? []).length;
}

// One message per operation whose bind list has the wrong length.
export function bindingMismatches(plan: Plan): string[] {
  const bad: string[] = [];
  for (const seg of plan.segments) {
    for (const op of [...(seg.captureOps ?? []), ...seg.ops]) {
      const want = declaredBindings(op.kernel);
      if (op.bind.length !== want) {
        bad.push(`${seg.prefix}${op.name} (${op.kernel}): ${op.bind.length} bound, ${want} declared`);
      }
    }
  }
  return bad;
}
