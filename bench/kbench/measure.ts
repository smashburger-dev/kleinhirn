// R1 measurement window (Festlegung 3 to 5). Pure loop, the timer is injected so the
// block rule can be tested without a browser.
//
// block 1: one sample per call. block B > 1 (timer step above 0.1 ms, Festlegung 5):
// the window covers B consecutive calls and the sample is the mean per call; with
// `calls` calls that gives calls / B samples, and p95 is a p95 over block means.

export interface WindowResult {
  samplesMs: number[];
  block: number;
  calls: number;
  outputs: Float32Array[];
}

export async function measureWindow(
  run: (index: number) => Promise<Float32Array>,
  first: number,
  calls: number,
  block: number,
  now: () => number = () => performance.now(),
  onProgress?: (done: number) => void,
): Promise<WindowResult> {
  if (block < 1 || calls % block !== 0) throw new Error(`calls ${calls} is not a multiple of block ${block}`);
  const samplesMs: number[] = [];
  const outputs: Float32Array[] = [];
  for (let k = 0; k < calls; k += block) {
    // The engine may reuse its output array, so a row is copied before the next call. With
    // block 1 the copy happens after the timer stops; inside a block it costs a few
    // microseconds per call and is part of the block mean.
    const t1 = now();
    let t2 = t1;
    for (let j = 0; j < block; j += 1) {
      const o = await run(first + k + j);
      if (block === 1) t2 = now();
      outputs.push(o.slice());
    }
    if (block > 1) t2 = now();
    samplesMs.push((t2 - t1) / block);
    onProgress?.(k + block);
  }
  return { samplesMs, block, calls, outputs };
}

// Block size from the measured timer step: above 0.1 ms blocks of 10 (Festlegung 5).
export function blockForTimerStep(stepMs: number): number {
  return stepMs > 0.1 ? 10 : 1;
}
