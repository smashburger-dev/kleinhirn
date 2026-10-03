// WebGPU device acquisition: adapter selection, feature detection
// (shader-f16, subgroups), required limits. No runtime dependencies.

export interface KhDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasF16: boolean;
  hasTimestamps: boolean;
  limitsMode: 'minimum' | 'default';
  adapterInfo: Record<string, string>;
  // The first error the device reported outside a scoped call (uncaptured
  // error, device lost). Once set, every later scoped call throws it.
  failure?: string | null;
}

// The limits every conformant WebGPU adapter must support (the spec's
// required values). Requesting them explicitly turns "stays within the
// minimum limits" into a hard contract: the device never grants more, so a
// kernel cannot silently start depending on a larger limit on a desktop GPU.
// These cover everything the engine uses; unlisted limits also default to
// the spec minimum, so omitting them here keeps the same guarantee.
export const MINIMUM_LIMITS: Record<string, number> = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
  maxComputeWorkgroupStorageSize: 16384,
  maxStorageBuffersPerShaderStage: 8,
  maxStorageBufferBindingSize: 134217728,
  maxBindGroups: 4,
  maxBufferSize: 268435456,
};

export async function getDevice(
  preferF16: boolean, minLimits: boolean,
): Promise<KhDevice> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable: navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU unavailable: no adapter');
  const hasF16 = adapter.features.has('shader-f16');
  // timestamp-query is optional: present only with developer/unsafe WebGPU
  // flags. Requested when the adapter exposes it; profiling degrades to null
  // otherwise (docs/PLAN.md K5, "where available").
  const hasTs = adapter.features.has('timestamp-query');
  const requiredFeatures: GPUFeatureName[] = [];
  if (preferF16 && hasF16) requiredFeatures.push('shader-f16');
  if (hasTs) requiredFeatures.push('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures,
    ...(minLimits ? { requiredLimits: MINIMUM_LIMITS } : {}),
  });
  const info = adapter.info ?? {};
  const kh: KhDevice = {
    adapter,
    device,
    hasF16: requiredFeatures.includes('shader-f16'),
    hasTimestamps: requiredFeatures.includes('timestamp-query'),
    limitsMode: minLimits ? 'minimum' : 'default',
    adapterInfo: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
    },
    failure: null,
  };
  // Errors the device raises outside a scope and a lost device are kept, not
  // only logged: the next call throws them instead of returning stale output.
  device.addEventListener('uncapturederror', (e) => {
    kh.failure ??= `WebGPU error: ${(e as GPUUncapturedErrorEvent).error.message}`;
  });
  void device.lost.then((lost) => {
    kh.failure ??= `WebGPU device lost (${lost.reason}): ${lost.message}`;
  });
  return kh;
}

function failIfBroken(kh: KhDevice): void {
  if (kh.failure) throw new Error(kh.failure);
}

// One serialized GPU call under a validation scope. `start` is the whole
// synchronous part (plan build, upload, encode, submit); the scope is pushed
// before it and popped right after it, in a finally so scopes stay paired when
// it throws. `read` (readback) and the scope result are awaited together and
// the output counts only when both are clean. WebGPU reports validation errors
// asynchronously: without this, an invalid command buffer is dropped and the
// readback returns zeros or an older result.
export async function scopedCall<S, R>(
  kh: KhDevice, start: () => S, read: (started: S) => Promise<R>,
): Promise<R> {
  failIfBroken(kh);
  const { device } = kh;
  device.pushErrorScope('validation');
  let started: S;
  let popped: Promise<GPUError | null> | undefined;
  try {
    started = start();
  } finally {
    popped = device.popErrorScope();
    popped.catch(() => {}); // a throw in `start` leaves this one unobserved
  }
  const [result, scope] = await Promise.allSettled([read(started), popped]);
  if (scope.status === 'rejected') throw scope.reason;
  if (scope.value) throw new Error(`WebGPU validation error: ${scope.value.message}`);
  failIfBroken(kh);
  if (result.status === 'rejected') throw result.reason;
  return result.value;
}

// Synchronous work under a validation scope (plan build at load).
export function scopedSync<S>(kh: KhDevice, work: () => S): Promise<S> {
  return scopedCall(kh, work, async (s) => s);
}
