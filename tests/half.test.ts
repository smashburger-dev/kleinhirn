// K28.4 step 4: binary16 to f32, native Float16Array and the hand-written
// conversion on the same values (all 65536 bit patterns plus named cases).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { halfBitsToFloat32, halfToFloat } from '../src/half.ts';

test('named values', () => {
  const cases: [number, number][] = [
    [0x0000, 0], [0x8000, -0], [0x3c00, 1], [0xbc00, -1], [0x4000, 2],
    [0x3555, 0.333251953125], [0x7bff, 65504], [0x0400, 2 ** -14],
    [0x0001, 2 ** -24], [0x03ff, 1023 * 2 ** -24], [0x7c00, Infinity], [0xfc00, -Infinity],
  ];
  for (const [bits, want] of cases) {
    assert.ok(Object.is(halfToFloat(bits), want), `0x${bits.toString(16)}`);
  }
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
  assert.ok(Number.isNaN(halfToFloat(0x7c01)));
});

test('own conversion equals Float16Array on every bit pattern', () => {
  const bits = new Uint16Array(65536);
  for (let i = 0; i < bits.length; i += 1) bits[i] = i;
  const own = halfBitsToFloat32(bits, false);
  const native = halfBitsToFloat32(bits, true);
  assert.equal(
    typeof (globalThis as { Float16Array?: unknown }).Float16Array, 'function',
    'this runtime has no Float16Array, the native path was not exercised');
  for (let i = 0; i < bits.length; i += 1) {
    assert.ok(Object.is(own[i], native[i]) || (Number.isNaN(own[i]) && Number.isNaN(native[i])),
      `0x${i.toString(16)}: ${own[i]} vs ${native[i]}`);
  }
});
