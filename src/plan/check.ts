// Plan checker (K28.R group A): a finished Plan against the limits of the
// device it will run on. Pure: no GPU calls, so an engine runs it before a
// single buffer is allocated and a plan that does not fit is never built.
// Rejects what the kernels cannot do (bucket length not a multiple of 4,
// relative attention with a head width outside groups of 4) and what the
// device cannot hold (bindings, buffers, dispatch grids, workgroup memory).

import { BATCH_SIZES } from '../cache.ts';
import { MINIMUM_LIMITS } from '../device.ts';
import type { Dim, KernelName, Plan } from './ir.ts';

// The limits the checker reads. GPUSupportedLimits fits; a missing value
// counts as the spec minimum, like an unrequested limit on a real device.
export interface PlanLimits {
  maxStorageBufferBindingSize?: number;
  maxBufferSize?: number;
  maxComputeWorkgroupsPerDimension?: number;
  maxComputeWorkgroupStorageSize?: number;
  maxStorageBuffersPerShaderStage?: number;
}

// Workgroup bytes per kernel for a bucket length, from the var<workgroup>
// declarations in src/kernels (tests/k28r_a.test.mjs reads them back).
export const WORKGROUP_BYTES: Record<KernelName, (length: number) => number> = {
  add: () => 0,
  attention: (L) => 4 * L + 256,   // scores[L] + red[64]
  attpv: () => 8200,               // sa[256] vec4<f32> + sb[256] vec4 (f32 plans) + keyFirst, keyLast
  attrel: () => 12292,             // sa[256] + sw[512] vec4 (f32 plans) + tileLive
  attscore: () => 12292,           // sa[256] + sw[512] vec4 (f32 plans) + tileLive
  attsoftmax: () => 256,           // red[64]
  attsoftrel: () => 256,           // red[64]
  embln: () => 256,                // red[64]
  gather: () => 0,
  geglu: () => 0,
  im2col: () => 0,
  layernorm: () => 256,            // red[64]
  masklogits: () => 0,
  matmul: () => 2048,              // ta[256] + tw[256]
  mbattention: (L) => 4 * L + 256, // scores[L] + red[64]
  mbflash: () => 16384,            // sk[512] + sv[512] vec4<f32> (f32 plans; f16 half)
  mmtile: () => 12288,
  mmtile16: () => 6144,            // sa[128] + sw[256] vec4<f32> (f32 plans; f16 half)
  mmtile8: () => 5120,             // sa[64] + sw[256] vec4<f32> (f32 plans; f16 half)             // sa[256] + sw[512] vec4<f32> (f32 plans; f16 half)
  pool: () => 0,
  rope: () => 0,
};

const dimOf = (d: Dim, rows: number): number =>
  d === 'rows' ? rows : d === 'rows8' ? Math.ceil(rows / 8) : d === 'rows16' ? Math.ceil(rows / 16) : d === 'rows32' ? Math.ceil(rows / 32) : d;

export interface CheckOptions {
  // Batch plans only: what depends on the stride alone (length rule,
  // workgroup memory, bindings per stage, head width) and on the weights was
  // settled when the bucket plan the stride comes from was loaded.
  batchOnly?: boolean;
}

// Every violation as text; empty when the plan fits. weightBytes gives the
// size of a bound weight tensor (undefined: not checked).
export function checkPlan(
  plan: Plan, limits: PlanLimits, weightBytes?: (name: string) => number | undefined,
  options: CheckOptions = {},
): string[] {
  const binding = limits.maxStorageBufferBindingSize ?? MINIMUM_LIMITS.maxStorageBufferBindingSize;
  const maxBuffer = limits.maxBufferSize ?? MINIMUM_LIMITS.maxBufferSize;
  const groups = limits.maxComputeWorkgroupsPerDimension ?? MINIMUM_LIMITS.maxComputeWorkgroupsPerDimension;
  const shared = limits.maxComputeWorkgroupStorageSize ?? MINIMUM_LIMITS.maxComputeWorkgroupStorageSize;
  const perStage = limits.maxStorageBuffersPerShaderStage ?? MINIMUM_LIMITS.maxStorageBuffersPerShaderStage;
  const where = `bucket L${plan.length} B${plan.batch}`;
  const bad: string[] = [];
  const over = (what: string, need: number, name: string, limit: number): void => {
    if (need > limit) bad.push(`${where}: ${what} needs ${need}, ${name} is ${limit}`);
  };
  const ops = plan.segments.flatMap((s) => [...(s.captureOps ?? []), ...s.ops]
    .map((op) => ({ op, at: `${s.prefix}${op.name}` })));

  if (!options.batchOnly) {
    if (!Number.isInteger(plan.length) || plan.length <= 0 || plan.length % 4 !== 0) {
      bad.push(`${where}: the length must be a positive multiple of 4 (the attention kernels read keys in groups of 4)`);
    }
  }
  if (!Number.isInteger(plan.batch) || plan.batch <= 0) bad.push(`${where}: the batch must be a positive integer`);

  const eltBytes = plan.f16 ? 2 : 4;
  const upload = plan.length * plan.batch * plan.embeddingSize * eltBytes;
  if (upload % 4 !== 0) {
    bad.push(`${where}: the word row upload of ${upload} B is not a multiple of 4 (writeBuffer needs it)`);
  }

  for (const b of plan.buffers) {
    over(`buffer ${b.id}`, b.bytes, 'maxStorageBufferBindingSize', binding);
    over(`buffer ${b.id}`, b.bytes, 'maxBufferSize', maxBuffer);
  }
  over('the output staging buffer', plan.output.bytes, 'maxBufferSize', maxBuffer);

  const rows = plan.length * plan.batch;
  for (const { op, at } of ops) {
    for (const d of op.dispatch) over(`dispatch ${at}`, dimOf(d, rows), 'maxComputeWorkgroupsPerDimension', groups);
    for (const a of op.alts ?? []) for (const d of a.dispatch) over(`dispatch ${at} (${a.kernel})`, dimOf(d, rows), 'maxComputeWorkgroupsPerDimension', groups);
    if (options.batchOnly) continue;
    over(`op ${at}`, op.bind.length, 'maxStorageBuffersPerShaderStage', perStage);
    over(`workgroup memory of ${at} (${op.kernel})`, WORKGROUP_BYTES[op.kernel](plan.length),
      'maxComputeWorkgroupStorageSize', shared);
    if (op.kernel === 'attention' && op.constants.D % 4 !== 0) {
      bad.push(`${where}: relative attention with head width ${op.constants.D} is not supported (needs a multiple of 4)`);
    }
  }

  if (weightBytes && !options.batchOnly) {
    const seen = new Set<string>();
    for (const { op } of ops) {
      for (const id of op.bind) {
        if (!id.startsWith('w:') || seen.has(id)) continue;
        seen.add(id);
        const bytes = weightBytes(id.slice(2));
        if (bytes !== undefined) over(`weight ${id.slice(2)}`, bytes, 'maxStorageBufferBindingSize', binding);
      }
    }
  }
  return [...new Set(bad)];
}

// Throws one error naming every violation; used for bucket plans at load.
export function assertPlan(
  plan: Plan, limits: PlanLimits, weightBytes?: (name: string) => number | undefined,
): void {
  const bad = checkPlan(plan, limits, weightBytes);
  if (bad.length) throw new Error(bad.join('; '));
}

// The plan for batch size `batch`, or for the largest smaller size of
// BATCH_SIZES (16, 8, 4) that fits; undefined when none fits and the caller
// runs its rows on the bucket plan (B1). Only plans that fit are built.
export function fitBatchPlan(
  build: (batch: number) => Plan, limits: PlanLimits, batch: number,
): Plan | undefined {
  for (const size of [...BATCH_SIZES].reverse()) {
    if (size === 1 || size > batch) continue;
    const plan = build(size);
    if (checkPlan(plan, limits, undefined, { batchOnly: true }).length === 0) return plan;
  }
  return undefined;
}
