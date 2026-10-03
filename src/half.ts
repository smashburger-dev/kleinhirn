// IEEE binary16 bits to f32. Float16Array is used where the runtime has it;
// halfToFloat is the same conversion by hand (sign, exponent, mantissa,
// subnormals, Inf, NaN) and tests/half.test.ts checks both on one value set.

export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man === 0 ? sign * Infinity : NaN;
  return sign * (1 + man / 1024) * 2 ** (exp - 15);
}

export function halfBitsToFloat32(
  bits: Uint16Array, useNative = true,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(bits.length);
  const Native = (globalThis as { Float16Array?: new (b: ArrayBuffer, o: number, n: number) => ArrayLike<number> }).Float16Array;
  if (useNative && Native) {
    const view = new Native(bits.buffer as ArrayBuffer, bits.byteOffset, bits.length);
    for (let i = 0; i < bits.length; i += 1) out[i] = view[i];
    return out;
  }
  for (let i = 0; i < bits.length; i += 1) out[i] = halfToFloat(bits[i]);
  return out;
}
