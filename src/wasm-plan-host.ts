// Message handler of the WASM executor (R2): what the worker (src/wasm-plan-worker.ts) does with
// each request, as a plain function so the Node parity run can call it without a worker.
// Requests run strictly in arrival order. The weights wait in their shards until the first plan
// arrives: its matmuls decide which tensors are stored in panels.

import type { RunInput } from './plan/executor.ts';
import type { Plan } from './plan/ir.ts';
import {
  DEFAULT_SPLIT, panelWeights, resolveBuild, WasmPlanRuntime, type OpProfile, type SpawnHelper, type ThreadSplit,
  type WasmBuild, type WasmPlan,
} from './plan/wasm.ts';
import type { TensorDesc } from './weights.ts';

export type HostRequest =
  | { type: 'init'; shards: ArrayBuffer[]; tensors: TensorDesc[]; build?: WasmBuild | 'auto';
    threads?: number | 'auto'; split?: Partial<ThreadSplit> }
  | { type: 'close' }
  | { type: 'plan'; pid: number; plan: Plan }
  | { type: 'sync' }
  | { type: 'run'; pid: number; input: RunInput; rows: number; profile?: number }
  | { type: 'destroy'; pid: number };

export interface HostReply {
  out?: Float32Array;
  memoryBytes?: number;
  weightBytes?: number;
  error?: string;
  profile?: OpProfile;
  build?: WasmBuild;
  threads?: number;
  // why the executor runs fewer threads than asked (no isolation, no SharedArrayBuffer, ...)
  threadNote?: string;
}

// 'auto' (R8 hc8): the core count, at most 8. On the M1 Pro (8 P-cores, 2 E-cores; Chromium and
// Firefox report 10) 8 threads beat 10 in every searched cell (FINDINGS §57); other devices are
// not measured.
const AUTO_MAX = 8;

// Thread count from the option; more than one needs SharedArrayBuffer,
// cross-origin isolation (where the environment reports it) and a way to start helpers.
function threadsFor(asked: number | 'auto' | undefined, cores: number, canSpawn: boolean): { n: number; note?: string } {
  const want = asked === 'auto' ? Math.min(cores, AUTO_MAX) : Math.max(1, Math.floor(asked ?? 1));
  if (want <= 1) return { n: 1 };
  if (typeof SharedArrayBuffer === 'undefined') return { n: 1, note: 'no SharedArrayBuffer' };
  if ((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === false) return { n: 1, note: 'not cross-origin isolated' };
  if (!canSpawn) return { n: 1, note: 'no helper workers' };
  return { n: want };
}

// wasm(build, threaded): the bytes or module of plan.wasm, plan-relaxed.wasm, or their -mt builds.
// spawn starts a helper worker (absent: one thread); cores is the thread count of 'auto'.
export function createHost(
  wasm: (build: WasmBuild, threaded: boolean) => Promise<BufferSource | WebAssembly.Module>,
  spawn?: SpawnHelper, cores: () => number = () => 1,
): (m: HostRequest) => Promise<HostReply> {
  let rt: WasmPlanRuntime | undefined;
  let pending: { shards: ArrayBuffer[]; tensors: TensorDesc[] } | undefined;
  const plans = new Map<number, WasmPlan>();
  // A failed plan message has no reply of its own: the next sync reports it.
  let failure: string | undefined;
  let chain: Promise<unknown> = Promise.resolve();

  const sizes = (): HostReply => ({ memoryBytes: rt?.memoryBytes, weightBytes: rt?.weightBytes });

  const handle = async (m: HostRequest): Promise<HostReply> => {
    switch (m.type) {
      case 'init': {
        const build = resolveBuild(m.build);
        const t = threadsFor(m.threads, cores(), spawn !== undefined);
        let note = t.note;
        if (t.n > 1) {
          try {
            rt = await WasmPlanRuntime.createThreaded(await wasm(build, true), t.n, spawn!, { ...DEFAULT_SPLIT, ...m.split });
          } catch (e) {
            note = `threaded build failed: ${e instanceof Error ? e.message : String(e)}`.slice(0, 200);
          }
        }
        rt ??= await WasmPlanRuntime.create(await wasm(build, false));
        pending = { shards: m.shards, tensors: m.tensors };
        return { ...sizes(), build, threads: rt.threads, ...(note ? { threadNote: note } : {}) };
      }
      case 'close':
        rt?.stopThreads();
        return {};
      case 'plan': {
        try {
          if (!rt) throw new Error('wasm executor: plan before init');
          if (pending) {
            const panel = panelWeights([m.plan]);
            for (const t of pending.tensors) {
              const bytes = pending.shards[t.shard].slice(t.offset, t.offset + t.byteLength);
              rt.addWeight({ name: t.name, shape: t.shape, data: new Float32Array(bytes) }, panel.has(t.name));
            }
            pending = undefined;
          }
          plans.set(m.pid, rt.plan(m.plan));
        } catch (e) {
          failure ??= `plan L${m.plan.length} B${m.plan.batch}: ${e instanceof Error ? e.message : String(e)}`;
        }
        return {};
      }
      case 'sync':
        return failure ? { error: failure } : sizes();
      case 'run': {
        const p = plans.get(m.pid);
        if (!p) throw new Error(failure ?? `wasm executor: plan ${m.pid} missing`);
        p.upload(m.input);
        if (m.profile) {
          const profile = p.profile(m.rows, m.profile, () => performance.now());
          return { out: p.readOutput(), profile, ...sizes() };
        }
        p.run(m.rows);
        return { out: p.readOutput(), ...sizes() };
      }
      case 'destroy':
        plans.delete(m.pid);
        return {};
      default:
        throw new Error(`wasm executor: request ${(m as { type: string }).type}`);
    }
  };

  return (m) => {
    const next = chain.then(() => handle(m));
    chain = next.catch(() => {});
    return next;
  };
}
