// NormalizedString of HF tokenizers 0.22.2 (tokenizers/src/tokenizer/normalizer.rs
// and pattern.rs, Apache-2.0) as arrays: one entry per normalized code point with
// the span it came from in the original text. The Rust type keeps the alignment
// per byte, here it is per code point; offsets count UTF-16 units of the original
// string, so text.slice(start, end) in JS gives the token.

export interface Normalized {
  chars: string[];
  starts: number[];
  ends: number[];
  // UTF-16 start of this piece in the original text (Rust offsets_original().0)
  origin?: number;
}

export function fromText(text: string, base = 0): Normalized {
  const chars: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  let pos = base;
  for (const c of text) {
    chars.push(c);
    starts.push(pos);
    pos += c.length;
    ends.push(pos);
  }
  return { chars, starts, ends, origin: base };
}

export function utf8Length(c: string): number {
  const cp = c.codePointAt(0) as number;
  return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
}

// Rust NormalizedString::transform: dest is (char, changes). changes 0 replaces
// one source char, a positive value adds a char that shares the alignment of the
// previous source char, a negative value replaces one char and swallows -changes
// following source chars. initialOffset source chars are dropped up front.
export function transform(n: Normalized, dest: Array<[string, number]>, initialOffset = 0): Normalized {
  const chars: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  const last = n.chars.length - 1;
  const origin = n.origin ?? 0;
  let p = initialOffset;
  for (const [c, changes] of dest) {
    let s: number;
    let e: number;
    if (changes > 0) {
      if (p < 1) {
        s = origin;
        e = origin;
      } else {
        s = n.starts[p - 1];
        e = n.ends[p - 1];
      }
    } else {
      const q = Math.min(p, last);
      s = n.starts[q];
      e = n.ends[q];
      p += 1;
      if (changes < 0) p += -changes;
    }
    chars.push(c);
    starts.push(s);
    ends.push(e);
  }
  return { chars, starts, ends, origin: n.origin };
}

export function slice(n: Normalized, a: number, b: number): Normalized {
  return {
    chars: n.chars.slice(a, b), starts: n.starts.slice(a, b), ends: n.ends.slice(a, b),
    origin: a < n.starts.length ? n.starts[a] : n.origin,
  };
}

// Match spans over code point indices: [start, end, isMatch]; together they cover the string.
export type Span = [number, number, boolean];

export function matchPredicate(chars: string[], pred: (c: string) => boolean): Span[] {
  if (chars.length === 0) return [[0, 0, false]];
  const out: Span[] = [];
  let lastOffset = 0;
  for (let i = 0; i < chars.length; i += 1) {
    if (pred(chars[i])) {
      if (lastOffset < i) out.push([lastOffset, i, false]);
      out.push([i, i + 1, true]);
      lastOffset = i + 1;
    }
  }
  if (chars.length > lastOffset) out.push([lastOffset, chars.length, false]);
  return out;
}

// UTF-16 index of the joined string to code point index, one extra entry for the end.
export function cpIndex(chars: string[]): number[] {
  const map: number[] = [];
  let pos = 0;
  for (let i = 0; i < chars.length; i += 1) {
    while (map.length < pos) map.push(i - 1);
    map.push(i);
    pos += chars[i].length;
  }
  while (map.length <= pos) map.push(chars.length);
  map[pos] = chars.length;
  return map;
}

// re needs the g flag. Code points are the unit, so the u flag keeps astral chars whole.
export function matchRegex(chars: string[], re: RegExp): Span[] {
  if (chars.length === 0) return [[0, 0, false]];
  const s = chars.join('');
  const map = cpIndex(chars);
  const out: Span[] = [];
  let prev = 0;
  re.lastIndex = 0;
  for (let m = re.exec(s); m !== null; m = re.exec(s)) {
    const a = map[m.index];
    const b = map[m.index + m[0].length];
    if (m[0].length === 0) re.lastIndex += 1;
    if (prev !== a) out.push([prev, a, false]);
    out.push([a, b, true]);
    prev = b;
  }
  if (prev !== chars.length) out.push([prev, chars.length, false]);
  return out;
}

export type SplitBehavior = 'removed' | 'isolated' | 'merged_with_next';

// Rust NormalizedString::split.
export function splitMatches(n: Normalized, spans: Span[], behavior: SplitBehavior): Normalized[] {
  let ranges: Array<[number, number, boolean]>;
  if (behavior === 'removed') {
    ranges = spans;
  } else if (behavior === 'isolated') {
    ranges = spans.map(([a, b]) => [a, b, false]);
  } else {
    // reversed fold: a match merges with the piece that follows it, unless that is a match too
    let prev = false;
    const acc: Array<[number, number, boolean]> = [];
    for (let i = spans.length - 1; i >= 0; i -= 1) {
      const [a, b, m] = spans[i];
      if (m && !prev) {
        if (acc.length) acc[acc.length - 1][0] = a;
        else acc.push([a, b, false]);
      } else {
        acc.push([a, b, false]);
      }
      prev = m;
    }
    acc.reverse();
    ranges = acc;
  }
  const out: Normalized[] = [];
  for (const [a, b, remove] of ranges) if (!remove) out.push(slice(n, a, b));
  return out;
}

// Rust NormalizedString::replace: every match becomes content, all content chars
// take the alignment of the last matched char.
export function replaceMatches(n: Normalized, spans: Span[], content: string): Normalized {
  const chars: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  const add = (from: number, to: number) => {
    for (let i = from; i < to; i += 1) {
      chars.push(n.chars[i]);
      starts.push(n.starts[i]);
      ends.push(n.ends[i]);
    }
  };
  const cs = [...content];
  for (const [a, b, m] of spans) {
    if (!m) {
      add(a, b);
      continue;
    }
    const s = b >= 1 ? n.starts[b - 1] : (n.origin ?? 0);
    const e = b >= 1 ? n.ends[b - 1] : (n.origin ?? 0);
    for (const c of cs) {
      chars.push(c);
      starts.push(s);
      ends.push(e);
    }
  }
  return { chars, starts, ends, origin: n.origin };
}

// Rust NormalizedString::prepend: the new chars take the alignment of the first char.
export function prepend(n: Normalized, s: string): Normalized {
  if (n.chars.length === 0) return n;
  const cs = [...s];
  const chars = [...cs, ...n.chars];
  const starts = [...cs.map(() => n.starts[0]), ...n.starts];
  const ends = [...cs.map(() => n.ends[0]), ...n.ends];
  return { chars, starts, ends, origin: n.origin };
}

export function join(n: Normalized): string {
  return n.chars.join('');
}
