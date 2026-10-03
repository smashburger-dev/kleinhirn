// Recording WebGPU mock for plan tests (K28.1). It implements exactly the
// calls the plan classes make, records them in submission order, and checks
// two rules a real device enforces at dispatch time: a bind group must come
// from the layout of the pipeline that runs it, and a writable binding may
// not share its buffer with another binding of the same dispatch.

import { createHash } from 'node:crypto';

export const USAGE = {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
  UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
} as const;

export interface MockBuffer {
  kind: 'buffer';
  id: number;
  size: number;
  usage: number;
  label: string;
  // sha256 pieces of build-time writes, in write order
  initParts: string[];
}

interface MockModule { kind: 'module'; code: string }

export interface MockPipeline {
  kind: 'pipeline';
  code: string;
  constants: Record<string, number>;
  bindings: Set<number>;
  writable: Set<number>;
}

interface MockLayout { kind: 'layout'; pipeline: MockPipeline }

interface MockBindGroup {
  kind: 'bindgroup';
  pipeline: MockPipeline;
  entries: { binding: number; buffer: MockBuffer }[];
}

interface MockQuerySet { kind: 'queryset'; count: number }

export type RawEvent =
  | { op: 'write'; buf: MockBuffer; off: number; bytes: number; sha: string; phase: 'build' | 'run' }
  | { op: 'begin'; ts: [number, number] | null }
  | { op: 'pipeline'; pipeline: MockPipeline }
  | { op: 'bind'; group: number; bg: MockBindGroup }
  | { op: 'dispatch'; x: number; y: number }
  | { op: 'end' }
  | { op: 'copy'; src: MockBuffer; so: number; dst: MockBuffer; do: number; size: number }
  | { op: 'resolve'; first: number; count: number; dst: MockBuffer; off: number }
  | { op: 'map' };

export interface MockGpu {
  device: GPUDevice;
  log: RawEvent[];
  buffers: MockBuffer[];
  weights(absent?: string[]): Map<string, GPUBuffer>;
  markBuilt(): void;
}

export function installGpuGlobals(): void {
  const g = globalThis as Record<string, unknown>;
  g.GPUBufferUsage = USAGE;
  g.GPUMapMode = { READ: 1, WRITE: 2 };
}

const sha = (data: Uint8Array | string): string =>
  createHash('sha256').update(data).digest('hex');

function parseShader(code: string): { bindings: Set<number>; writable: Set<number> } {
  const bindings = new Set<number>();
  const writable = new Set<number>();
  const re = /@binding\((\d+)\)\s*var<storage,\s*(read_write|read)>/g;
  for (let m = re.exec(code); m; m = re.exec(code)) {
    bindings.add(Number(m[1]));
    if (m[2] === 'read_write') writable.add(Number(m[1]));
  }
  return { bindings, writable };
}

function bytesOf(data: ArrayBufferView | ArrayBuffer): Uint8Array {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function createMockGpu(features: string[] = []): MockGpu {
  const log: RawEvent[] = [];
  const buffers: MockBuffer[] = [];
  let phase: 'build' | 'run' = 'build';
  let nextId = 0;

  const mkBuffer = (size: number, usage: number, label = ''): MockBuffer => {
    const b: MockBuffer = {
      kind: 'buffer', id: nextId++, size, usage, label, initParts: [] };
    return b;
  };

  const wrapBuffer = (b: MockBuffer): GPUBuffer => Object.assign(b, {
    size: b.size,
    async mapAsync(): Promise<void> { log.push({ op: 'map' }); },
    getMappedRange(): ArrayBuffer { return new ArrayBuffer(b.size); },
    unmap(): void {},
    destroy(): void {},
  }) as unknown as GPUBuffer;

  const encoderFor = () => {
    const cmds: RawEvent[] = [];
    let current: MockPipeline | undefined;
    let bound: MockBindGroup | undefined;
    let inPass = false;
    const pass = {
      setPipeline(p: MockPipeline): void {
        current = p;
        cmds.push({ op: 'pipeline', pipeline: p });
      },
      setBindGroup(group: number, bg: MockBindGroup): void {
        if (group !== 0) throw new Error(`mock: bind group index ${group}`);
        bound = bg;
        cmds.push({ op: 'bind', group, bg });
      },
      dispatchWorkgroups(x: number, y = 1, z = 1): void {
        if (z !== 1) throw new Error('mock: z dispatch');
        if (!current || !bound) throw new Error('mock: dispatch without pipeline or bind group');
        if (bound.pipeline !== current) {
          throw new Error('mock: bind group was built from another pipeline layout');
        }
        const seen = new Map<MockBuffer, number[]>();
        for (const e of bound.entries) {
          seen.set(e.buffer, [...(seen.get(e.buffer) ?? []), e.binding]);
        }
        for (const [buf, at] of seen) {
          if (at.length > 1 && at.some((b) => current!.writable.has(b))) {
            throw new Error(
              `mock: writable buffer ${buf.label || buf.id} bound at ${at.join(',')} in one dispatch`);
          }
        }
        cmds.push({ op: 'dispatch', x, y });
      },
      end(): void { inPass = false; cmds.push({ op: 'end' }); },
    };
    return {
      beginComputePass(desc?: GPUComputePassDescriptor): GPUComputePassEncoder {
        if (inPass) throw new Error('mock: nested compute pass');
        inPass = true;
        const w = desc?.timestampWrites;
        cmds.push({
          op: 'begin',
          ts: w ? [w.beginningOfPassWriteIndex as number, w.endOfPassWriteIndex as number] : null,
        });
        current = undefined;
        bound = undefined;
        return pass as unknown as GPUComputePassEncoder;
      },
      copyBufferToBuffer(
        src: MockBuffer, so: number, dst: MockBuffer, dof: number, size: number,
      ): void {
        if (inPass) throw new Error('mock: copy inside a compute pass');
        cmds.push({ op: 'copy', src, so, dst, do: dof, size });
      },
      resolveQuerySet(
        _qs: MockQuerySet, first: number, count: number, dst: MockBuffer, off: number,
      ): void {
        if (inPass) throw new Error('mock: resolve inside a compute pass');
        cmds.push({ op: 'resolve', first, count, dst, off });
      },
      finish(): { cmds: RawEvent[] } {
        if (inPass) throw new Error('mock: unfinished compute pass');
        return { cmds };
      },
    };
  };

  const device = {
    features: new Set(features),
    limits: { maxStorageBufferBindingSize: 1 << 30 },
    queue: {
      writeBuffer(
        buf: MockBuffer, off: number, data: ArrayBufferView | ArrayBuffer,
        dataOffset?: number, size?: number,
      ): void {
        if (dataOffset !== undefined || size !== undefined) {
          throw new Error('mock: writeBuffer with data offset or size');
        }
        const bytes = bytesOf(data);
        if (off + bytes.byteLength > buf.size) {
          throw new Error(`mock: write past end of buffer ${buf.label || buf.id}`);
        }
        const h = sha(bytes);
        if (phase === 'build') buf.initParts.push(`${off}:${h}`);
        log.push({ op: 'write', buf, off, bytes: bytes.byteLength, sha: h, phase });
      },
      submit(cbs: { cmds: RawEvent[] }[]): void {
        for (const cb of cbs) log.push(...cb.cmds);
      },
    },
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      const b = mkBuffer(desc.size, desc.usage);
      buffers.push(b);
      return wrapBuffer(b);
    },
    createShaderModule(desc: { code: string }): MockModule {
      return { kind: 'module', code: desc.code };
    },
    createComputePipeline(desc: {
      compute: { module: MockModule; constants?: Record<string, number> };
    }): MockPipeline {
      const { bindings, writable } = parseShader(desc.compute.module.code);
      const p = {
        kind: 'pipeline' as const, code: desc.compute.module.code,
        constants: { ...(desc.compute.constants ?? {}) }, bindings, writable,
      };
      return Object.assign(p, {
        getBindGroupLayout(i: number): MockLayout {
          if (i !== 0) throw new Error('mock: layout index');
          return { kind: 'layout', pipeline: p };
        },
      });
    },
    createBindGroup(desc: {
      layout: MockLayout;
      entries: { binding: number; resource: { buffer: MockBuffer } }[];
    }): MockBindGroup {
      const pipeline = desc.layout.pipeline;
      for (const e of desc.entries) {
        if (!pipeline.bindings.has(e.binding)) {
          throw new Error(`mock: binding ${e.binding} is not in the pipeline layout`);
        }
      }
      return {
        kind: 'bindgroup', pipeline,
        entries: desc.entries.map((e) => ({ binding: e.binding, buffer: e.resource.buffer })),
      };
    },
    // Error scopes and events: no-ops that record nothing in the trace log.
    pushErrorScope(): void {},
    popErrorScope(): Promise<GPUError | null> { return Promise.resolve(null); },
    addEventListener(): void {},
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    createCommandEncoder: encoderFor,
    createQuerySet(desc: { count: number }): MockQuerySet {
      return Object.assign({ kind: 'queryset' as const, count: desc.count }, {
        destroy(): void {},
      });
    },
  } as unknown as GPUDevice;

  return {
    device,
    log,
    buffers,
    weights(absent: string[] = []): Map<string, GPUBuffer> {
      const skip = new Set(absent);
      // Missing names are created on first get, like a loaded manifest.
      class AutoMap extends Map<string, GPUBuffer> {
        override get(name: string): GPUBuffer | undefined {
          if (!this.has(name) && !skip.has(name)) {
            this.set(name, wrapBuffer(mkBuffer(0, USAGE.STORAGE, `w:${name}`)));
          }
          return super.get(name);
        }
      }
      return new AutoMap();
    },
    markBuilt(): void { phase = 'run'; },
  };
}
