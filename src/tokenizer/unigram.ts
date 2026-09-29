// SentencePiece Unigram decoding: Viterbi over the vocab scores exactly as
// HF tokenizers' Unigram model does (min-score - 10 for unknown chars,
// fused consecutive UNK tokens). Pieces are indexed in a code-point trie so
// each lattice position only walks actual prefixes.

const UNK_PENALTY = 10;

interface TrieNode {
  children: Map<number, TrieNode>;
  id: number; // vocab id or -1
}

export interface UnigramVocab {
  root: TrieNode;
  scores: number[];
  ids: Map<string, number>;
  unkScore: number;
  unkId: number;
}

export function buildUnigram(vocab: [string, number][], unkId: number): UnigramVocab {
  const root: TrieNode = { children: new Map(), id: -1 };
  const scores: number[] = [];
  const ids = new Map<string, number>();
  let min = Infinity;
  for (const [piece, score] of vocab) {
    const id = scores.length;
    scores.push(score);
    ids.set(piece, id);
    let node = root;
    for (const ch of piece) {
      const cp = ch.codePointAt(0) as number;
      let next = node.children.get(cp);
      if (!next) {
        next = { children: new Map(), id: -1 };
        node.children.set(cp, next);
      }
      node = next;
    }
    node.id = id;
    if (score < min) min = score;
  }
  return { root, scores, ids, unkScore: min - UNK_PENALTY, unkId };
}

// Best segmentation of one pretoken. Returns ids; unknown code points map to
// unkId, fused into a single UNK token each run.
export function unigramEncode(v: UnigramVocab, text: string): number[] {
  const chars = [...text];
  const n = chars.length;
  if (n === 0) return [];
  const best = new Array<number>(n + 1).fill(-Infinity);
  const back = new Array<number>(n + 1).fill(-1); // vocab id or -2 for unk
  const backStart = new Array<number>(n + 1).fill(0);
  best[0] = 0;
  for (let i = 0; i < n; i += 1) {
    if (best[i] === -Infinity) continue;
    let node = v.root;
    let hasSingleCharNode = false;
    for (let j = i; j < n; j += 1) {
      const next = node.children.get(chars[j].codePointAt(0) as number);
      if (!next) break;
      node = next;
      if (node.id >= 0) {
        if (j === i) hasSingleCharNode = true;
        const s = best[i] + v.scores[node.id];
        if (s > best[j + 1]) {
          best[j + 1] = s;
          back[j + 1] = node.id;
          backStart[j + 1] = i;
        }
      }
    }
    if (!hasSingleCharNode) {
      const s = best[i] + v.unkScore;
      if (s > best[i + 1]) {
        best[i + 1] = s;
        back[i + 1] = -2;
        backStart[i + 1] = i;
      }
    }
  }
  const rev: number[] = [];
  for (let j = n; j > 0; j = backStart[j]) {
    rev.push(back[j] === -2 ? v.unkId : back[j]);
  }
  const out: number[] = [];
  let pendingUnk = false;
  for (const id of rev.reverse()) {
    if (id === v.unkId) {
      if (pendingUnk) continue; // fuse consecutive unknowns
      pendingUnk = true;
    } else {
      pendingUnk = false;
    }
    out.push(id);
  }
  return out;
}
