// GLiNER2 classification schema input, replicating
// gliner2.processor.SchemaTransformer (inference path) plus the
// cls_marker_indices/cls_marker_mask/cls_group_index routing of
// ExtractorCollator, for classification tasks only:
// schema tokens "( [P] <task> ( [L] <label> ... ) )" joined by [SEP_STRUCT],
// then [SEP_TEXT] and the whitespace-split lowercased text words.
// Each combined token is tokenized separately; classification markers are
// the [L] positions (the [P] position is dropped via positions[1:]).

import type { HfTokenizer } from './tokenizer.ts';

export class BucketOverflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BucketOverflowError';
  }
}

const SEP_STRUCT = '[SEP_STRUCT]';
const SEP_TEXT = '[SEP_TEXT]';
const P_TOKEN = '[P]';
const L_TOKEN = '[L]';

// WhitespaceTokenSplitter from gliner2.processing.word_splitter: URLs,
// emails, @handles, \w words with -/_ joins, else single non-space char.
const WORD_RE = /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]/giu;

export function splitWords(text: string): string[] {
  const words: string[] = [];
  WORD_RE.lastIndex = 0;
  for (const m of text.matchAll(WORD_RE)) words.push(m[0].toLowerCase());
  return words;
}

export interface SchemaInput {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  markerIndices: Int32Array;
  markerMask: Float32Array;
  markerGroups: Int32Array;
  seqLen: number;
}

export function prepareTasks(
  tokenizer: HfTokenizer,
  text: string,
  tasks: [string, string[]][],
  length: number,
  maxOptions: number,
): SchemaInput {
  const totalLabels = tasks.reduce((n, [, labels]) => n + labels.length, 0);
  if (totalLabels < 1 || totalLabels > maxOptions) {
    throw new BucketOverflowError(
      `expected 1..${maxOptions} labels in total, got ${totalLabels}`);
  }
  let t = text;
  if (t && !/[.!?]$/.test(t)) t += '.';
  if (!t) t = '.';
  const words = splitWords(t);

  const structs = tasks.map(([task, labels]) => [
    '(', P_TOKEN, task, '(',
    ...labels.flatMap((l) => [L_TOKEN, l]),
    ')', ')',
  ]);
  const combined: string[] = [];
  for (const s of structs) {
    combined.push(...s, SEP_STRUCT);
  }
  combined.pop();
  combined.push(SEP_TEXT, ...words);

  const markerOrig = new Set<number>();
  const pOrig = new Set<number>();
  let offset = 0;
  for (const s of structs) {
    if (s.length > 1) {
      markerOrig.add(offset + 1);
      pOrig.add(offset + 1); // [P]
    }
    for (let i = 4; i < s.length - 2; i += 2) markerOrig.add(offset + i); // [L]
    offset += s.length + 1;
  }

  const ids: number[] = [];
  // positions[0] of each struct is the [P] marker; cls routes keep [L] only.
  const markerPositions: { pos: number; group: number }[] = [];
  let structIdx = 0;
  let seenSepText = false;
  for (let orig = 0; orig < combined.length; orig += 1) {
    const token = combined[orig];
    const isSchema = !seenSepText;
    if (token === SEP_TEXT) seenSepText = true;
    else if (token === SEP_STRUCT) structIdx += 1;
    const subwordPos = ids.length;
    ids.push(...tokenizer.encodeIds(token));
    if (isSchema && !pOrig.has(orig) && markerOrig.has(orig)) {
      markerPositions.push({ pos: subwordPos, group: structIdx });
    }
  }
  const seqLen = ids.length;
  if (seqLen > length || markerPositions.length > maxOptions) {
    throw new BucketOverflowError(
      `input exceeds bucket: seqLen ${seqLen} > ${length} or markers > ${maxOptions}`);
  }
  return finishRoutes(
    markerPositions, ids, seqLen, length, maxOptions, tokenizer.padTokenId);
}

function finishRoutes(
  markerPositions: { pos: number; group: number }[],
  ids: number[],
  seqLen: number,
  length: number,
  maxOptions: number,
  padId: number,
): SchemaInput {
  const inputIds = new Int32Array(length).fill(padId);
  inputIds.set(ids.slice(0, seqLen));
  const attentionMask = new Int32Array(length);
  attentionMask.fill(1, 0, seqLen);
  const markerIndices = new Int32Array(maxOptions);
  const markerMask = new Float32Array(maxOptions);
  const markerGroups = new Int32Array(maxOptions);
  for (const [i, m] of markerPositions.entries()) {
    if (i >= maxOptions) break;
    markerIndices[i] = m.pos;
    markerMask[i] = 1;
    markerGroups[i] = m.group;
  }
  return { inputIds, attentionMask, markerIndices, markerMask, markerGroups, seqLen };
}
