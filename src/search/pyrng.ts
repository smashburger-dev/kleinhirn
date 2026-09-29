// Port of CPython's random.Random (MT19937 + init_by_array seeding,
// randbelow/randint/choice/sample/shuffle/choices). Bitwise compatible with
// Python 3.12 for the methods used by interference-search, so a seeded search
// run produces identical draws.

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

function mul(a: number, b: number): number {
  return Math.imul(a, b) >>> 0;
}

export class PyRandom {
  private mt = new Uint32Array(N);
  private mti = N + 1;

  constructor(seed: number) {
    // CPython splits a positive int seed into little-endian 32-bit words.
    const key: number[] = [];
    let s = Math.floor(Math.abs(seed));
    while (s > 0) {
      key.push(s % 0x100000000);
      s = Math.floor(s / 0x100000000);
    }
    this.initByArray(key.length ? key : [0]);
  }

  private initGenrand(s: number): void {
    this.mt[0] = s >>> 0;
    for (this.mti = 1; this.mti < N; this.mti++) {
      const prev = this.mt[this.mti - 1];
      this.mt[this.mti] = (mul(1812433253, prev ^ (prev >>> 30)) + this.mti) >>> 0;
    }
  }

  private initByArray(key: number[]): void {
    this.initGenrand(19650218);
    let i = 1, j = 0;
    let k = N > key.length ? N : key.length;
    for (; k > 0; k--) {
      const prev = this.mt[i - 1];
      const mix = mul(1664525, prev ^ (prev >>> 30));
      this.mt[i] = ((this.mt[i] ^ mix) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { this.mt[0] = this.mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (k = N - 1; k > 0; k--) {
      const prev = this.mt[i - 1];
      const mix = mul(1566083941, prev ^ (prev >>> 30));
      this.mt[i] = ((this.mt[i] ^ mix) - i) >>> 0;
      i++;
      if (i >= N) { this.mt[0] = this.mt[N - 1]; i = 1; }
    }
    this.mt[0] = UPPER_MASK;
  }

  private genrand(): number {
    const mt = this.mt;
    if (this.mti >= N) {
      let y: number;
      for (let kk = 0; kk < N - M; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (let kk = N - M; kk < N - 1; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M - N] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random(): number {
    const a = this.genrand() >>> 5;
    const b = this.genrand() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  private getrandbits(k: number): number {
    if (k <= 0) return 0;
    const words = Math.ceil(k / 32);
    let r = 0;
    for (let i = 0; i < words; i++) {
      let w = this.genrand();
      if (i === words - 1) w >>>= words * 32 - k;
      r += w * Math.pow(2, 32 * i);
    }
    return r;
  }

  randbelow(n: number): number {
    let k = 0; // exact bit_length of n, no float log
    for (let x = n; x > 0; x = Math.floor(x / 2)) k++;
    if (k === 0) return 0;
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  randint(a: number, b: number): number {
    return a + this.randbelow(b - a + 1);
  }

  choice<T>(arr: readonly T[]): T {
    return arr[this.randbelow(arr.length)];
  }

  shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randbelow(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  sample<T>(population: readonly T[], k: number): T[] {
    const n = population.length;
    if (k > n) throw new Error('sample larger than population');
    let setsize = 21;
    if (k > 5) setsize += Math.pow(4, Math.ceil(Math.log(k * 3) / Math.log(4)));
    const result: T[] = [];
    if (n <= setsize) {
      const pool = population.slice();
      for (let i = 0; i < k; i++) {
        const j = this.randbelow(n - i);
        result.push(pool[j]);
        pool[j] = pool[n - i - 1]; // move non-selected item into vacancy
      }
    } else {
      const selected = new Set<number>();
      for (let i = 0; i < k; i++) {
        let j = this.randbelow(n);
        while (selected.has(j)) j = this.randbelow(n);
        selected.add(j);
        result.push(population[j]);
      }
    }
    return result;
  }

  choices<T>(population: readonly T[], weights: number[], k = 1): T[] {
    const cum: number[] = [];
    let total = 0;
    for (const w of weights) { total += w; cum.push(total); }
    const hi = cum.length - 1;
    const out: T[] = [];
    for (let i = 0; i < k; i++) {
      const x = this.random() * total;
      let lo = 0, h = hi;
      while (lo < h) {
        const mid = (lo + h) >> 1;
        if (x < cum[mid]) h = mid; else lo = mid + 1;
      }
      out.push(population[lo]);
    }
    return out;
  }
}
