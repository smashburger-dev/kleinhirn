/// <reference lib="esnext.float16" />
// safetensors reader for the converter core (K28 design section 7). Runs
// under Node and in the browser: no node: imports, erasable syntax only.
// Float tensors come out as f32 (F16, BF16 and F64 convert exactly or by
// round-to-nearest-even); integer tensors (position_ids) are listed but not
// readable.

export interface StTensor {
  name: string;
  dtype: string;
  shape: number[];
  start: number; // absolute byte offset in the file
  end: number;
}

export interface StFile {
  bytes: Uint8Array;
  tensors: Map<string, StTensor>;
}

const FLOAT_DTYPES = new Set(['F32', 'F16', 'BF16', 'F64']);

// Byte width of every dtype of the safetensors format; a header with another dtype is rejected.
const DTYPE_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1,
  I64: 8, I32: 4, I16: 2, I8: 1, U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

export function isFloatDtype(dtype: string): boolean {
  return FLOAT_DTYPES.has(dtype);
}

export function parseSafetensors(bytes: Uint8Array): StFile {
  if (bytes.byteLength < 8) throw new Error('safetensors: file shorter than 8 bytes');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLen) || 8 + headerLen > bytes.byteLength) throw new Error('safetensors: header exceeds file');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen))) as
    Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const base = 8 + headerLen;
  const dataLength = bytes.byteLength - base;
  const tensors = new Map<string, StTensor>();
  for (const [name, v] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const width = DTYPE_BYTES[v?.dtype];
    if (width === undefined) throw new Error(`safetensors: ${name} has unknown dtype ${String(v?.dtype)}`);
    if (!Array.isArray(v.shape) || v.shape.some((d) => !Number.isInteger(d) || d < 0)) {
      throw new Error(`safetensors: ${name} has shape ${JSON.stringify(v.shape)}, expected non-negative integers`);
    }
    const [from, to] = Array.isArray(v.data_offsets) ? v.data_offsets : [NaN, NaN];
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || from > to || to > dataLength) {
      throw new Error(`safetensors: ${name} has data_offsets [${String(from)}, ${String(to)}], `
        + `outside the ${dataLength} bytes of data (truncated file?)`);
    }
    const expected = elementCount(v.shape) * width;
    if (to - from !== expected) {
      throw new Error(`safetensors: ${name} has ${to - from} bytes, ${v.dtype} ${v.shape.join('x')} needs ${expected}`);
    }
    tensors.set(name, { name, dtype: v.dtype, shape: v.shape, start: base + from, end: base + to });
  }
  const byStart = [...tensors.values()].sort((x, y) => x.start - y.start || x.end - y.end);
  for (let i = 1; i < byStart.length; i += 1) {
    if (byStart[i].start < byStart[i - 1].end) {
      throw new Error(`safetensors: ${byStart[i].name} overlaps ${byStart[i - 1].name}`);
    }
  }
  return { bytes, tensors };
}

export function elementCount(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

// Aligned copy of the tensor bytes, so typed-array views are always legal.
function slice(file: StFile, t: StTensor): ArrayBuffer {
  const out = new Uint8Array(t.end - t.start);
  out.set(file.bytes.subarray(t.start, t.end));
  return out.buffer;
}

export function readF32(file: StFile, name: string): Float32Array {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`safetensors: missing tensor ${name}`);
  const n = elementCount(t.shape);
  const buf = slice(file, t);
  switch (t.dtype) {
    case 'F32':
      return new Float32Array(buf);
    case 'F16':
      return Float32Array.from(new Float16Array(buf));
    case 'BF16': {
      // bf16 is the top half of an f32: shifting is exact.
      const bits = new Uint16Array(buf);
      const out = new Uint32Array(n);
      for (let i = 0; i < n; i += 1) out[i] = bits[i] << 16;
      return new Float32Array(out.buffer);
    }
    case 'F64':
      return Float32Array.from(new Float64Array(buf));
    default:
      throw new Error(`safetensors: ${name} has dtype ${t.dtype}, no float view`);
  }
}
