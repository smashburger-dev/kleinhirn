// Worker of the WASM executor (R2): one module instance and its memory, requests in order. With
// threads (R8) it is the coordinator and starts the helper workers (src/wasm-plan-helper.ts).

import type { SpawnHelper, WasmBuild } from './plan/wasm.ts';
import { createHost, type HostReply, type HostRequest } from './wasm-plan-host.ts';

// Each build sits in its own chunk (R8 hc6, hc7): a browser fetches only the one it runs.
const builds: Record<string, () => Promise<{ default: string }>> = {
  plain: () => import('./wasm/build-plain.ts'),
  relaxed: () => import('./wasm/build-relaxed.ts'),
  'plain-mt': () => import('./wasm/build-plain-mt.ts'),
  'relaxed-mt': () => import('./wasm/build-relaxed-mt.ts'),
};

const spawn: SpawnHelper = (module, memory, ctl, tid) => {
  const w = new Worker(new URL('./wasm-plan-helper.ts', import.meta.url), { type: 'module' });
  const ready = new Promise<void>((resolve, reject) => {
    w.onmessage = (e) => { if (e.data === 'ready') resolve(); };
    w.onerror = (e) => reject(new Error(`helper ${tid}: ${e.message ?? 'script error'}`));
  });
  w.postMessage({ module, memory, ctl, tid });
  return { ready, terminate: () => w.terminate() };
};

const host = createHost(
  async (build: WasmBuild, threaded: boolean) => (await fetch((await builds[`${build}${threaded ? '-mt' : ''}`]()).default)).arrayBuffer(),
  spawn, () => navigator.hardwareConcurrency || 1);
const scope = self as unknown as Worker;

scope.onmessage = async (e: MessageEvent<{ id: number; msg: HostRequest }>) => {
  let reply: HostReply;
  try {
    reply = await host(e.data.msg);
  } catch (err) {
    reply = { error: err instanceof Error ? err.message : String(err) };
  }
  scope.postMessage({ id: e.data.id, ...reply }, reply.out ? [reply.out.buffer] : []);
};
