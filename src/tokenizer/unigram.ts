// SentencePiece Unigram decoding: Viterbi over the vocab scores exactly as
// HF tokenizers' Unigram model does (min-score - 10 for unknown chars,
// fused consecutive UNK tokens). Pieces are indexed in a code-point trie so
// each lattice position only walks actual prefixes.

import type { CompactUnigram } from './bin.ts';
import { PairTable } from './pairs.ts';

const UNK_PENALTY = 10;

// The trie as a table of edges (node, code point) to child node, node 0 is the root. valueOf[node]
// is the vocab id of the piece that ends there, or -1.
export interface UnigramVocab {
  edges: PairTable;
  valueOf: Int32Array;
  scores: Float64Array;
  unkScore: number;
  unkId: number;
}

// The pieces of the trie are read from the blob of the vocab (no string per piece).
export function buildUnigram(vocab: CompactUnigram, unkId: number): UnigramVocab {
  const scores = Float64Array.from(vocab.scores);
  const { blob, offs } = vocab.pieces;
  const n = vocab.pieces.length;
  // a node per code point at most: the UTF-16 length of the pieces bounds the count
  let bound = 1;
  for (let id = 0; id < n; id += 1) bound += offs[id + 1] - offs[id];
  const valueOf = new Int32Array(bound).fill(-1);
  const edges = new PairTable(0);
  let nodes = 1;
  let min = Infinity;
  for (let id = 0; id < n; id += 1) {
    let node = 0;
    for (let k = offs[id]; k < offs[id + 1];) {
      const cp = blob.codePointAt(k) as number;
      k += cp > 0xffff ? 2 : 1;
      let next = edges.get(node, cp);
      if (next === -1) {
        next = nodes;
        nodes += 1;
        edges.set(node, cp, next);
      }
      node = next;
    }
    valueOf[node] = id;
    if (scores[id] < min) min = scores[id];
  }
  return { edges, valueOf, scores, unkScore: min - UNK_PENALTY, unkId };
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
    let node = 0;
    let hasSingleCharNode = false;
    for (let j = i; j < n; j += 1) {
      const next = v.edges.get(node, chars[j].codePointAt(0) as number);
      if (next === -1) break;
      node = next;
      const id = v.valueOf[node];
      if (id >= 0) {
        if (j === i) hasSingleCharNode = true;
        const s = best[i] + v.scores[id];
        if (s > best[j + 1]) {
          best[j + 1] = s;
          back[j + 1] = id;
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
