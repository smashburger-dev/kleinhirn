// Token strings in one blob with a typed hash index (R4 Stufe T2). The tokens of a tokenizer.bin
// are one decoded string plus a Uint32Array of UTF-16 offsets, so loading creates no string per
// token. The index maps a string to the position of its token (the last position wins for a
// repeated string, as Map.set in order does). Lookups compare code units against the blob and
// build no string: a merged BPE token is looked up as two blob ranges, a Unigram piece as the
// code units of its chars.

export const FNV_BASIS = 0x811c9dc5 | 0;

// FNV-1a over the UTF-16 code units s[from..to) continuing the hash h.
export function fnv(h: number, s: string, from = 0, to = s.length): number {
  let x = h;
  for (let k = from; k < to; k += 1) x = Math.imul(x ^ s.charCodeAt(k), 0x01000193);
  return x;
}

export class Tokens {
  // offs[i]..offs[i + 1] is token i in blob (UTF-16 units); offs has length + 1 entries
  readonly offs: Uint32Array;

  constructor(readonly blob: string, offs: Uint32Array) {
    this.offs = offs;
  }

  static fromStrings(list: ArrayLike<string>): Tokens {
    const offs = new Uint32Array(list.length + 1);
    let parts = '';
    for (let i = 0; i < list.length; i += 1) {
      offs[i + 1] = offs[i] + list[i].length;
      parts += list[i];
    }
    return new Tokens(parts, offs);
  }

  get length(): number {
    return this.offs.length - 1;
  }

  tokenAt(i: number): string {
    return this.blob.slice(this.offs[i], this.offs[i + 1]);
  }
}

export class StringIndex {
  private readonly slots: Int32Array; // position + 1 of a token, 0 for empty
  private readonly hashes: Int32Array;
  private readonly mask: number;
  readonly tokens: Tokens;

  constructor(tokens: Tokens) {
    this.tokens = tokens;
    const n = tokens.length;
    let size = 16;
    while (size < 2 * n) size *= 2;
    this.mask = size - 1;
    this.slots = new Int32Array(size);
    this.hashes = new Int32Array(n);
    const { blob, offs } = tokens;
    for (let i = 0; i < n; i += 1) {
      const h = fnv(FNV_BASIS, blob, offs[i], offs[i + 1]);
      this.hashes[i] = h;
      let p = this.slotOf(h);
      for (;;) {
        const v = this.slots[p] - 1;
        if (v < 0) { this.slots[p] = i + 1; break; }
        if (this.hashes[v] === h && this.sameRange(v, i)) { this.slots[p] = i + 1; break; }
        p = (p + 1) & this.mask;
      }
    }
  }

  private slotOf(h: number): number {
    return (h ^ (h >>> 13)) & this.mask;
  }

  private sameRange(a: number, b: number): boolean {
    const { blob, offs } = this.tokens;
    const la = offs[a + 1] - offs[a];
    if (la !== offs[b + 1] - offs[b]) return false;
    for (let k = 0; k < la; k += 1) {
      if (blob.charCodeAt(offs[a] + k) !== blob.charCodeAt(offs[b] + k)) return false;
    }
    return true;
  }

  // position of the token equal to s, or -1
  indexOf(s: string): number {
    const h = fnv(FNV_BASIS, s);
    const { blob, offs } = this.tokens;
    let p = this.slotOf(h);
    for (;;) {
      const v = this.slots[p] - 1;
      if (v < 0) return -1;
      if (this.hashes[v] === h && offs[v + 1] - offs[v] === s.length) {
        let k = 0;
        const base = offs[v];
        while (k < s.length && blob.charCodeAt(base + k) === s.charCodeAt(k)) k += 1;
        if (k === s.length) return v;
      }
      p = (p + 1) & this.mask;
    }
  }

  // position of the token equal to token a followed by token b without its first `from` units
  // (the BPE merged token a + b.slice(from)); h is not needed by the caller
  indexOfConcat(a: number, b: number, from: number): number {
    const { blob, offs } = this.tokens;
    const aS = offs[a];
    const aLen = offs[a + 1] - aS;
    const bS = offs[b] + from;
    const bLen = offs[b + 1] - bS;
    const len = aLen + bLen;
    let h = fnv(FNV_BASIS, blob, aS, aS + aLen);
    h = fnv(h, blob, bS, bS + bLen);
    let p = this.slotOf(h);
    for (;;) {
      const v = this.slots[p] - 1;
      if (v < 0) return -1;
      if (this.hashes[v] === h && offs[v + 1] - offs[v] === len) {
        const base = offs[v];
        let k = 0;
        while (k < aLen && blob.charCodeAt(base + k) === blob.charCodeAt(aS + k)) k += 1;
        if (k === aLen) {
          while (k < len && blob.charCodeAt(base + k) === blob.charCodeAt(bS + k - aLen)) k += 1;
          if (k === len) return v;
        }
      }
      p = (p + 1) & this.mask;
    }
  }

  // position of the token equal to chars[start] + ... + chars[end - 1], h being FNV_BASIS fed with
  // the code units of those chars (fnv(h, chars[k]) for each k in order)
  indexOfChars(chars: string[], start: number, end: number, h: number): number {
    const { blob, offs } = this.tokens;
    let len = 0;
    for (let k = start; k < end; k += 1) len += chars[k].length;
    let p = this.slotOf(h);
    for (;;) {
      const v = this.slots[p] - 1;
      if (v < 0) return -1;
      if (this.hashes[v] === h && offs[v + 1] - offs[v] === len) {
        let at = offs[v];
        let ok = true;
        for (let k = start; k < end && ok; k += 1) {
          const c = chars[k];
          for (let j = 0; j < c.length; j += 1) {
            if (blob.charCodeAt(at) !== c.charCodeAt(j)) { ok = false; break; }
            at += 1;
          }
        }
        if (ok) return v;
      }
      p = (p + 1) & this.mask;
    }
  }
}
