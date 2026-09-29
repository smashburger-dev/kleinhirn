// Port of the feature side of interference_search/judge.py (Apache-2.0, Bad
// Theory Labs). The network itself is not ported yet; this file carries the
// input encoding a kleinhirn judge will consume (see docs/ARCHITECTURE.md,
// K17 section).

export const MODS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const;

export function numFeats(x: number, t: number): number[] {
  const lx = Math.log1p(x), lt = Math.log1p(t);
  const f = [
    lx / 10,
    (lx - lt) / 5,
    x === t ? 1 : 0,
    x > t ? 1 : 0,
    t % x === 0 ? 1 : 0,
    x % t === 0 ? 1 : 0,
    Math.min(Math.abs(x - t), 1000) / 1000,
  ];
  for (const m of MODS) f.push(x % m === t % m ? 1 : 0);
  for (const m of MODS) f.push((x % m) / m);
  return f;
}

export const N_FEAT = numFeats(3, 7).length; // 29
const PAD = new Array<number>(N_FEAT).fill(0);

// Batch of states -> per-number feature rows plus a mask, padded to maxN
// numbers per state. `target` is one int or one per state.
export function encode(
  states: readonly (readonly number[])[],
  target: number | readonly number[],
  maxN = 7,
): { feats: number[][][]; mask: boolean[][] } {
  const ts = typeof target === 'number' ? states.map(() => target) : target;
  const feats: number[][][] = [];
  const mask: boolean[][] = [];
  states.forEach((s, i) => {
    const t = ts[i];
    const rows = s.map((x) => numFeats(x, t));
    while (rows.length < maxN) rows.push([...PAD]);
    feats.push(rows);
    mask.push([...s.map(() => true), ...new Array(maxN - s.length).fill(false)]);
  });
  return { feats, mask };
}
