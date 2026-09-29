// Memory calibration page: allocates a known 512 MB GPUBuffer so the runner
// can identify which OS-level metric reflects GPU memory (see
// docs/ARCHITECTURE.md, Messprotokoll). State lives on window.khCalibrate.

export {};

declare global {
  interface Window {
    khCalibrate?: {
      stage: string;
      adapterInfo?: unknown;
      alloc?: () => Promise<void>;
      release?: () => void;
      error?: string;
    };
  }
}

const state: NonNullable<Window['khCalibrate']> = { stage: 'boot' };
window.khCalibrate = state;

async function main(): Promise<void> {
  if (!navigator.gpu) {
    state.stage = 'error';
    state.error = 'navigator.gpu missing';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    state.stage = 'error';
    state.error = 'no adapter';
    return;
  }
  state.adapterInfo = adapter.info;
  const device = await adapter.requestDevice();
  const size = 512 * 1024 * 1024;
  let buffer: GPUBuffer | null = null;
  state.alloc = async () => {
    buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Touch the allocation so it is actually resident, not just reserved.
    const data = new Uint8Array(64 * 1024 * 1024).fill(1);
    for (let off = 0; off < size; off += data.byteLength) {
      device.queue.writeBuffer(buffer!, 0, data, 0, data.byteLength);
    }
    await device.queue.onSubmittedWorkDone();
    state.stage = 'allocated';
  };
  state.release = () => {
    buffer?.destroy();
    buffer = null;
    state.stage = 'released';
  };
  state.stage = 'ready';
}

void main();
