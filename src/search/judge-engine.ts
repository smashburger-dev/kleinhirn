// JudgeEngine: serves the Countdown domain's judge(states, target) calls on
// WebGPU. Candidate states are encoded with the upstream feature function,
// chunked through the K16 batch sizes (1/4/8/16), and scored by
// SetJudgePlan; the sigmoid on the returned logits mirrors upstream's
// score(states, target) -> P(alive).
import { getDevice } from '../device.ts';
import { fetchManifest, loadWeights, type LoadedWeights } from '../weights.ts';
import { MAX_BATCH, nextBatchSize } from '../cache.ts';
import { SetJudgePlan, type SetJudgeSpec } from '../graph/setjudge.ts';
import { encode as judgeEncode, N_FEAT } from './judge.ts';
import type { JudgeFn } from './countdown-domain.ts';

function f32ToF16Bits(v: number): number {
  const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
  const a = Math.abs(v);
  if (a === 0 || a < 2 ** -25) return sign;
  if (a > 65504) return sign | 0x7bff;
  const e = Math.floor(Math.log2(a));
  if (e >= -14) {
    let m = Math.round((a / 2 ** e - 1) * 1024);
    let ee = e;
    if (m === 1024) { m = 0; ee += 1; }
    if (ee > 15) return sign | 0x7bff;
    return sign | ((ee + 15) << 10) | m;
  }
  return sign | Math.round(a / 2 ** -24); // subnormal
}

function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 31) return m === 0 ? sign * Infinity : NaN;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

export interface JudgeLoadOptions {
  manifestUrl: string;
  precision?: 'auto' | 'f16' | 'f32';
  limits?: 'minimum' | 'default';
}

export class JudgeEngine {
  private plans = new Map<number, SetJudgePlan>();
  private queue: Promise<unknown> = Promise.resolve();
  gpuBytes = 0;
  downloadBytes = 0;

  private constructor(
    private device: GPUDevice,
    private weights: LoadedWeights,
    private spec: SetJudgeSpec,
    public readonly precision: 'f16' | 'f32',
  ) {}

  static async load(options: JudgeLoadOptions): Promise<JudgeEngine> {
    const kh = await getDevice(options.precision !== 'f32',
      options.limits !== 'default');
    const weights = await loadWeights(kh.device, options.manifestUrl);
    const dtype = weights.manifest.tensors[0]?.dtype ?? 'f32';
    if (dtype === 'f16' && !kh.hasF16) {
      throw new Error('f16 manifest but adapter lacks shader-f16; use the f32 manifest');
    }
    if (options.precision && options.precision !== 'auto'
      && options.precision !== dtype) {
      throw new Error(`precision ${options.precision} requested but manifest is ${dtype}`);
    }
    const spec = weights.manifest.encoder as unknown as SetJudgeSpec;
    const engine = new JudgeEngine(
      kh.device, weights, spec, dtype as 'f16' | 'f32');
    engine.downloadBytes = weights.downloadBytes;
    engine.gpuBytes = weights.gpuBytes;
    return engine;
  }

  private planFor(batch: number): SetJudgePlan {
    let p = this.plans.get(batch);
    if (!p) {
      p = new SetJudgePlan(
        this.device, this.spec, this.weights.tensors,
        this.precision === 'f16', batch);
      this.plans.set(batch, p);
      this.gpuBytes += p.gpuBytes;
    }
    return p;
  }

  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const next = this.queue.then(job);
    this.queue = next.catch(() => {});
    return next;
  }

  // Scores one chunk of up to MAX_BATCH states: feature rows at
  // b*maxN+1..+len, class slot row b*maxN left zero (the plan writes the
  // learned class token over it); mask [1, 1*len, 0*pad] per sequence.
  private async scoreChunk(
    states: readonly (readonly number[])[], target: number,
  ): Promise<number[]> {
    const plan = this.planFor(nextBatchSize(states.length));
    const S = plan.batch * this.spec.maxN;
    const mask = new Float32Array(S);
    const f16 = this.precision === 'f16';
    const feats = f16
      ? new Uint16Array(S * this.spec.nFeat)
      : new Float32Array(S * this.spec.nFeat);
    const { feats: rows, mask: ms } = judgeEncode(states, target,
      this.spec.maxN - 1);
    states.forEach((_, b) => {
      mask[b * this.spec.maxN] = 1; // class token always valid
      for (let j = 1; j < this.spec.maxN; j += 1) {
        mask[b * this.spec.maxN + j] = ms[b][j - 1] ? 1 : 0;
        // encoded row j-1 holds the features of number j-1; row 0 is the
        // class slot and stays zero (the plan writes the token over it)
        const src = rows[b][j - 1];
        for (let f = 0; f < this.spec.nFeat; f += 1) {
          const v = src[f];
          feats[b * this.spec.maxN * this.spec.nFeat + j * this.spec.nFeat + f] =
            f16 ? f32ToF16Bits(v) : v;
        }
      }
    });
    return this.enqueue(async () => {
      plan.upload(feats, mask);
      plan.submit();
      const raw = await plan.readLogits();
      const out: number[] = [];
      for (let b = 0; b < states.length; b += 1) {
        const logit = raw instanceof Float32Array
          ? raw[b] : f16BitsToF32((raw as Uint16Array)[b]);
        out.push(1 / (1 + Math.exp(-logit)));
      }
      return out;
    });
  }

  // JudgeFn for the Countdown domain: P(alive) per state, chunked through
  // the K16 batch sizes.
  score: JudgeFn = async (states, target) => {
    if (states.some((s) => s.length > this.spec.maxN - 1)) {
      throw new Error(`judge state longer than ${this.spec.maxN - 1} numbers`);
    }
    const out: number[] = [];
    for (let i = 0; i < states.length; i += MAX_BATCH) {
      out.push(...await this.scoreChunk(
        states.slice(i, i + MAX_BATCH), target));
    }
    return out;
  };
}

export { N_FEAT };
