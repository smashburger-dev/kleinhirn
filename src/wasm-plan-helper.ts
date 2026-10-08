// Helper worker of the threaded WASM executor (R8 Festlegung 6): instantiates the coordinator's
// module on its shared memory and runs helper(ctl, tid), which takes work items of each
// dispatched op until the coordinator sets stop.

import type { WasmExports } from './plan/wasm.ts';

const scope = self as unknown as Worker;

scope.onmessage = async (e: MessageEvent<{ module: WebAssembly.Module; memory: WebAssembly.Memory; ctl: number; tid: number }>) => {
  const { module, memory, ctl, tid } = e.data;
  const instance = await WebAssembly.instantiate(module, { env: { memory, abort: () => { throw new Error('wasm abort'); } } });
  scope.postMessage('ready');
  (instance.exports as unknown as WasmExports).helper(ctl, tid);
  scope.postMessage('stopped');
};
