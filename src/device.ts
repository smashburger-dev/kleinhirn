// WebGPU device acquisition: adapter selection, feature detection
// (shader-f16, subgroups), required limits. No runtime dependencies.

export interface KhDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasF16: boolean;
  hasTimestamps: boolean;
  limitsMode: 'minimum' | 'default';
  adapterInfo: Record<string, string>;
}

// The limits every conformant WebGPU adapter must support (the spec's
// required values). Requesting them explicitly turns "stays within the
// minimum limits" into a hard contract: the device never grants more, so a
// kernel cannot silently start depending on a larger limit on a desktop GPU.
// These cover everything the engine uses; unlisted limits also default to
// the spec minimum, so omitting them here keeps the same guarantee.
const MINIMUM_LIMITS: Record<string, number> = {
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
  return {
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
  };
}
