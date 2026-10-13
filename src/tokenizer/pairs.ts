// Open-addressing table from a pair of vocab ids (a, b) to an int32 value. The keys are
// stored exactly (no string or double keys), so a BPE merge table with hundreds of thousands
// of pairs costs no string per pair.

export class PairTable {
  private mask!: number;
  private keys!: Int32Array; // a and b of slot i at 2i and 2i + 1
  private vals!: Int32Array; // -1 for an empty slot
  private count = 0;

  // room for `entries` pairs at a load of at most 2/3 (the table grows beyond that)
  constructor(entries: number) {
    let size = 16;
    while (size * 2 < entries * 3) size *= 2;
    this.alloc(size);
  }

  private alloc(size: number): void {
    this.mask = size - 1;
    this.keys = new Int32Array(size * 2);
    this.vals = new Int32Array(size).fill(-1);
  }

  private grow(): void {
    const oldKeys = this.keys;
    const oldVals = this.vals;
    this.alloc(this.vals.length * 2);
    for (let i = 0; i < oldVals.length; i += 1) {
      if (oldVals[i] !== -1) this.place(oldKeys[2 * i], oldKeys[2 * i + 1], oldVals[i]);
    }
  }

  // inserts a pair that is not in the table
  private place(a: number, b: number, value: number): void {
    let i = this.slot(a, b);
    while (this.vals[i] !== -1) i = (i + 1) & this.mask;
    this.keys[2 * i] = a;
    this.keys[2 * i + 1] = b;
    this.vals[i] = value;
  }

  private slot(a: number, b: number): number {
    let h = Math.imul(a ^ Math.imul(b, 0x9e3779b1), 0x85ebca77);
    h ^= h >>> 15;
    return h & this.mask;
  }

  // a later set of the same pair replaces the value
  set(a: number, b: number, value: number): void {
    let i = this.slot(a, b);
    for (;;) {
      if (this.vals[i] === -1) {
        this.count += 1;
        this.place(a, b, value);
        if (this.count * 3 > this.vals.length * 2) this.grow();
        return;
      }
      if (this.keys[2 * i] === a && this.keys[2 * i + 1] === b) {
        this.vals[i] = value;
        return;
      }
      i = (i + 1) & this.mask;
    }
  }

  // the value of the pair, or -1
  get(a: number, b: number): number {
    let i = this.slot(a, b);
    for (;;) {
      const v = this.vals[i];
      if (v === -1) return -1;
      if (this.keys[2 * i] === a && this.keys[2 * i + 1] === b) return v;
      i = (i + 1) & this.mask;
    }
  }
}
