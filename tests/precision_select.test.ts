// K28.S3 step 4: an f16 manifest with recommendedPrecision "f32" makes EncoderModel.load (precision 'auto' or
// none) read the f32 manifest beside it. Explicit precisions, other manifests and a missing f32 manifest keep
// the f16 manifest; the missing case leaves a note that info() reports. Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveManifest, type Manifest } from '../src/weights.ts';

const f16 = { format: 'kleinhirn-weights-2', recommendedPrecision: 'f32', tensors: [], shards: [] } as unknown as Manifest;
const f32 = { format: 'kleinhirn-weights-2', tensors: [], shards: [] } as unknown as Manifest;
const plain = { ...f16, recommendedPrecision: undefined } as Manifest;
const url = '/models/k28/m/f16/manifest.json';
const fetcher = async (u: string): Promise<Manifest> => {
  if (u === '/models/k28/m/f32/manifest.json') return f32;
  throw new Error(`fetch ${u}: 404`);
};

test('auto and no precision follow the recommendation to the f32 manifest', async () => {
  for (const precision of ['auto', undefined] as const) {
    const r = await resolveManifest(url, precision, f16, fetcher);
    assert.equal(r.url, '/models/k28/m/f32/manifest.json');
    assert.equal(r.manifest, f32);
    assert.equal(r.note, undefined);
  }
});

test('an explicit precision, or a manifest without the recommendation, stays on the given manifest', async () => {
  for (const precision of ['f16', 'f32'] as const) {
    assert.equal((await resolveManifest(url, precision, f16, fetcher)).manifest, f16);
  }
  assert.equal((await resolveManifest(url, 'auto', plain, fetcher)).manifest, plain);
});

test('without an f32 manifest beside it the f16 manifest is used and the note says why', async () => {
  const r = await resolveManifest('/models/k28/other/f16/manifest.json', 'auto', f16, fetcher);
  assert.equal(r.manifest, f16);
  assert.match(r.note ?? '', /recommends f32, f32 manifest not available/);
  const odd = await resolveManifest('/models/k28/m/manifest.json', 'auto', f16, fetcher);
  assert.match(odd.note ?? '', /not .*f16\/manifest\.json/);
});
