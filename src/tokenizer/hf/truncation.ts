// Truncation of a sequence pair before the post-processor, as in tokenizers
// 0.22.2 (direction right, stride 0). max is already reduced by the number
// of special tokens the template adds.

export type TruncationStrategy = 'longest_first' | 'only_first' | 'only_second';

export const SEQUENCE_TOO_SHORT = 'Truncation error: Sequence to truncate too short to respect the provided max_length';

// Returns the kept lengths of the first and second sequence.
export function truncateLengths(
  n1: number, n2: number | null, max: number, strategy: TruncationStrategy,
): [number, number] {
  const second = n2 ?? 0;
  const total = n1 + second;
  if (max <= 0) return [0, 0];
  if (total <= max) return [n1, second];
  const remove = total - max;
  if (strategy === 'longest_first') {
    // The longer sequence shrinks to the length of the shorter one, then both
    // lose the same amount; the shorter one (the first on a tie) loses the odd
    // token. Measured against the goldens, not token-by-token alternation.
    const diff = Math.abs(n1 - second);
    const step = Math.min(diff, remove);
    const left = remove - step;
    const firstIsShorter = n1 <= second;
    const shortLoss = Math.ceil(left / 2);
    const longLoss = step + Math.floor(left / 2);
    const a = n1 - (firstIsShorter ? shortLoss : longLoss);
    const b = second - (firstIsShorter ? longLoss : shortLoss);
    return [a, b];
  }
  if (strategy === 'only_first') {
    if (n1 <= remove) throw new Error(SEQUENCE_TOO_SHORT);
    return [n1 - remove, second];
  }
  if (n2 === null) throw new Error('Truncation error: Second sequence not provided');
  if (n2 <= remove) throw new Error(SEQUENCE_TOO_SHORT);
  return [n1, n2 - remove];
}
