// K9 typed-decisions eval for kleinhirn-Julia: same 2,000 rows as the
// PyTorch gate run, strict encoding (head_length 512, max_length 1024),
// one request per forward, abstentions counted as errors.
import { JuliaEngine } from '../src/julia.ts';
import type { JuliaRequest } from '../src/tokenizer/julia-input.ts';

interface TaskEntry { row: JuliaRequest; meta: {
  id: string; type: string; keys: string[]; gold: string } }
interface Pred {
  id: string; type: string; gold: string; predicted: string | null;
  correct: boolean; status: string; latencyMs: number }
interface Result {
  stage: string; done?: boolean; error?: string; info?: unknown;
  precision?: string; total?: number; correct?: number; accuracy?: number;
  abstained?: number; byType?: Record<string, unknown>;
  predictions?: Pred[] }

declare global {
  interface Window { khK9Result: Result }
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const precision = params.get('precision') ?? 'f16';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const task = params.get('task') ?? 'typed';
  const limit = Number(params.get('limit') ?? 0);
  const result: Result = { stage: 'boot', precision };
  window.khK9Result = result;
  try {
    const t0 = performance.now();
    const kh = await JuliaEngine.load({
      manifestUrl: `/models/julia-1/${precision}/manifest.json`,
      buckets: [512, 1024],
      precision: 'auto',
      limits,
    });
    result.info = { ...kh.info(), loadMs: performance.now() - t0 };

    const entries = (await (await fetch(
      `/models/k9-data/tasks/${task}.jsonl`)).text())
      .split('\n').filter((l) => l.trim())
      .map((l) => JSON.parse(l) as TaskEntry);
    const items = limit > 0 ? entries.slice(0, limit) : entries;

    const predictions: Pred[] = [];
    let correct = 0;
    let abstained = 0;
    for (const [i, e] of items.entries()) {
      const started = performance.now();
      let predicted: string | null = null;
      let status = 'ok';
      try {
        const input = kh.prepare(e.row, 1024, 512, true);
        const res = await kh.runPrepared({
          inputIds: input.inputIds, markers: input.markers,
          qtype: input.qtype, seqLen: input.seqLen,
        });
        let best = 0;
        for (let k = 1; k < input.markers.length; k += 1) {
          if (res.logits[k] > res.logits[best]) best = k;
        }
        predicted = e.meta.keys[best];
      } catch {
        status = 'abstained';
        abstained += 1;
      }
      const latencyMs = performance.now() - started;
      const ok = predicted === e.meta.gold;
      if (ok) correct += 1;
      predictions.push({
        id: e.meta.id, type: e.meta.type, gold: e.meta.gold,
        predicted, correct: ok, status,
        latencyMs: Math.round(latencyMs * 100) / 100 });
      if (i % 200 === 0) result.stage = `eval ${i}/${items.length}`;
    }
    const byType: Record<string, { count: number; correct: number }> = {};
    for (const p of predictions) {
      const b = byType[p.type] ??= { count: 0, correct: 0 };
      b.count += 1;
      b.correct += p.correct ? 1 : 0;
    }
    result.byType = byType;
    result.predictions = predictions;
    result.total = items.length;
    result.correct = correct;
    result.accuracy = correct / items.length;
    result.abstained = abstained;
    result.stage = 'done';
    result.done = true;
    kh.dispose();
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

await main();
