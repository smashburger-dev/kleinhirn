// Julia decision input assembly, port of julia/data.py:sequence().
// Layout: [CLS] "<type> question: <q>" [SEP] ([MASK] <option_i>) x k [SEP]
// <state> [SEP]. Markers point at each [MASK]; qtype selects type_emb.
import type { BpeTokenizer } from './bpe.ts';

export interface JuliaRequest {
  state: string | unknown;
  question: string;
  options: string[];
  type?: 'choice' | 'score' | 'noul';
}

export interface JuliaInput {
  inputIds: Int32Array;
  markers: number[];
  qtype: number;
  seqLen: number;
  truncated: boolean;
}

const QTYPES: Record<string, number> = { choice: 0, score: 1, noul: 2 };

export function prepareDecision(
  tok: BpeTokenizer, row: JuliaRequest, maxLength = 1024, headLength = 256,
  strict = false,
): JuliaInput {
  const state = typeof row.state === 'string'
    ? row.state : JSON.stringify(row.state);
  if (strict && [state, row.question, ...row.options]
    .some((s) => s.includes(tok.maskToken))) {
    throw new Error('reserved model marker in request');
  }
  const clean = (s: string) => s.replaceAll(tok.maskToken, ' ');
  const type = row.type ?? 'choice';
  const head = tok.encodeIds(`${type} question: ${clean(row.question)}`);
  const optionIds = row.options.map((o) => tok.encodeIds(` ${clean(o)}`));
  if (strict && optionIds.some((x) => x.length > 48)) {
    throw new Error('option exceeds 48-token model contract');
  }
  let options = optionIds.map((x) => [tok.maskTokenId, ...x.slice(0, 48)]);
  let budget = headLength - options.reduce((n, o) => n + o.length, 0);
  if (budget < 16) {
    const perOption = Math.max(4, Math.floor((headLength - 16) / options.length));
    options = options.map((o) => o.slice(0, perOption));
    budget = headLength - options.reduce((n, o) => n + o.length, 0);
  }
  if (strict && (head.length > budget
    || options.some((o, i) => o.length !== optionIds[i].length + 1))) {
    throw new Error('question/options exceed lossless head budget');
  }
  const ids = [tok.bosTokenId,
    ...head.slice(0, Math.max(8, budget)), tok.eosTokenId];
  const markers: number[] = [];
  for (const option of options) {
    markers.push(ids.length);
    ids.push(...option);
  }
  ids.push(tok.eosTokenId);
  const stateIds = tok.encodeIds(clean(state));
  const room = maxLength - ids.length - 1;
  if (room < 1) {
    throw new Error('question/options exceed sequence budget');
  }
  const truncated = stateIds.length > room;
  if (strict && truncated) {
    throw new Error('state exceeds lossless context budget');
  }
  ids.push(...stateIds.slice(0, room), tok.eosTokenId);
  return {
    inputIds: Int32Array.from(ids), markers,
    qtype: QTYPES[type] ?? 0, seqLen: ids.length, truncated,
  };
}
