// R4 stage 3a: RemoteRows (src/words-rows.ts) against a fake server with Range support switches,
// redirects, 403s, wrong lengths, corrupt rows and an in-memory Cache Storage.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { encodeWords } from '../src/words-encode.ts';
import type { WordsDesc } from '../src/words-format.ts';
import { wordsView, type RowArray, type WordTable } from '../src/words.ts';
import { RemoteRows } from '../src/words-rows.ts';

const N = 3000;
const W = 8;
const ORIGIN = 'https://words.test/words.bin';
const CACHE = 'kleinhirn-words';

const hex = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

async function words(dtype: 'f16' | 'f32') {
  const bpe = dtype === 'f16' ? 2 : 4;
  const table = new Uint8Array(N * W * bpe);
  const dv = new DataView(table.buffer);
  for (let i = 0; i < N * W; i += 1) {
    if (dtype === 'f16') dv.setUint16(i * 2, (i * 37) & 0xffff, true);
    else dv.setFloat32(i * 4, i * 0.5 - 3, true);
  }
  const { bytes, salt } = await encodeWords(table, N, W, dtype);
  const rb = W * bpe + 4;
  const desc: WordsDesc = {
    file: 'words.bin', bytes: bytes.length, sha256: hex(bytes), dtype, rows: N, width: W, rowBytes: rb, salt,
  };
  return { bytes, desc, rb, dense: wordsView(bytes.buffer as ArrayBuffer, desc) };
}
type Words = Awaited<ReturnType<typeof words>>;

const W16 = await words('f16');
const W32 = await words('f32');
const each = (name: string, fn: (w: Words) => Promise<void>): void => {
  for (const w of [W16, W32]) test(`${name} (${w.desc.dtype})`, () => fn(w));
};

interface Ctl {
  rangeOff?: boolean; // ignore Range, answer 200 with the whole file
  fullFrom?: number; // request number (1 = probe) from which Range is ignored
  redirect?: boolean; // origin answers with a redirect to a signed URL (?sig=n)
  forbidOnce?: boolean; // the first signed URL answers 403
  badLength?: boolean; // 206 bodies lose their last byte
  corrupt?: number; // one data byte of this row is flipped in every range that holds it
}

// A fake server: Range, redirects, 403, bad length, corruption. Every request lands in log.
function server(w: Words, ctl: Ctl = {}) {
  const log: { url: string; range: string | null; status: number }[] = [];
  let epoch = 0;
  let forbidden = false;
  const fetchFn = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const asked = String(input);
    const range = (init?.headers as Record<string, string> | undefined)?.Range ?? null;
    const n = log.length + 1;
    let url = asked;
    let redirected = false;
    if (ctl.redirect && !asked.includes('?sig=')) {
      epoch += 1;
      url = `${ORIGIN}?sig=${epoch}`;
      redirected = true;
    }
    let status = 200;
    let body = w.bytes;
    if (ctl.forbidOnce && !forbidden && asked.includes('?sig=')) {
      forbidden = true;
      status = 403;
      body = new Uint8Array(0);
    } else if (range && !ctl.rangeOff && !(ctl.fullFrom && n >= ctl.fullFrom)) {
      const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) as RegExpExecArray;
      const start = Number(a);
      status = 206;
      body = w.bytes.slice(start, Number(b) + 1);
      if (ctl.corrupt !== undefined) {
        const off = ctl.corrupt * w.rb - start;
        if (off >= 0 && off < body.length) {
          body = body.slice();
          body[off + 1] ^= 0x10;
        }
      }
      if (ctl.badLength) body = body.slice(0, -1);
    }
    log.push({ url: asked, range, status });
    return { status, ok: status < 300, url, redirected, arrayBuffer: async () => body.slice().buffer } as unknown as Response;
  };
  return { fetch: fetchFn as unknown as typeof fetch, log, ctl };
}

// Cache Storage stand-in: one Map of key to bytes per cache name.
function cacheStorage() {
  const stores = new Map<string, Map<string, Uint8Array>>();
  return {
    stores,
    open: async (name: string) => {
      const m = stores.get(name) ?? new Map<string, Uint8Array>();
      stores.set(name, m);
      return {
        keys: async () => [...m.keys()].map((k) => new Request(k)),
        match: async (k: string) => {
          const b = m.get(String(k));
          return b ? new Response(b.slice()) : undefined;
        },
        put: async (k: string, r: Response) => {
          m.set(String(k), new Uint8Array(await r.arrayBuffer()));
        },
        delete: async (k: string) => m.delete(String(k)),
      };
    },
  };
}

// Installs a caches mock for the duration of fn and removes it afterwards.
async function withCaches<T>(api: unknown, fn: () => Promise<T>): Promise<T> {
  Object.defineProperty(globalThis, 'caches', { value: api, configurable: true, writable: true });
  try {
    return await fn();
  } finally {
    delete (globalThis as { caches?: unknown }).caches;
  }
}

function rowOf(t: WordTable, id: number): RowArray {
  const out = t.alloc(W);
  t.copyRow(id, out, 0);
  return out;
}

function sameRow(t: WordTable, dense: WordTable, id: number): void {
  assert.deepEqual(Array.from(rowOf(t, id)), Array.from(rowOf(dense, id)), `row ${id}`);
}

const rangeOf = (w: Words, first: number, last: number): string => `bytes=${first * w.rb}-${(last + 1) * w.rb - 1}`;

each('probe 206: rows mode with the probe rows present, equal to the dense table', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  const info = t.info();
  assert.equal(info.mode, 'rows');
  assert.equal(info.range, true);
  assert.equal(info.rows, 8);
  assert.equal(info.requests, 1);
  assert.equal(s.log[0].range, rangeOf(w, 0, 7));
  for (let id = 0; id < 8; id += 1) sameRow(t, w.dense, id);
});

each('probe 200: full table with range false and fallback no-range', async (w) => {
  const s = server(w, { rangeOff: true });
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  const info = t.info();
  assert.equal(info.mode, 'full');
  assert.equal(info.range, false);
  assert.equal(info.fallback, 'no-range');
  for (const id of [0, 5, 1234, N - 1]) sameRow(t, w.dense, id);
  assert.equal(t.ensure([1, 2], 2), null);
});

each('ensure returns null when the rows are present', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  assert.equal(t.ensure([0, 3, 7], 3), null);
  assert.equal(s.log.length, 1);
});

each('ranges merge gaps of up to 4 rows and not 5', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch, probeRows: 1 });
  await t.ensure([100, 105], 2);
  assert.deepEqual(s.log.slice(1).map((e) => e.range), [rangeOf(w, 100, 105)]);
  await t.ensure([300, 306], 2);
  assert.deepEqual(s.log.slice(2).map((e) => e.range), [rangeOf(w, 300, 300), rangeOf(w, 306, 306)]);
  for (const id of [100, 105, 300, 306]) sameRow(t, w.dense, id);
});

each('concurrent ensure calls share the rows in flight', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  const before = s.log.length;
  const p1 = t.ensure([500, 501], 2);
  const p2 = t.ensure([500, 501], 2);
  assert.ok(p1 && p2);
  await Promise.all([p1, p2]);
  assert.equal(s.log.length - before, 1);
  sameRow(t, w.dense, 501);
});

each('copyRow equals the dense table for every requested id', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  const ids = [42, 43, 44, 77, 1500, 2999];
  await t.ensure(ids, ids.length);
  for (const id of ids) sameRow(t, w.dense, id);
  const out = t.alloc(2 * W);
  t.copyRow(77, out, W);
  assert.deepEqual(Array.from(out.subarray(W)), Array.from(rowOf(w.dense, 77)));
});

each('a checksum failure rejects the call and the row stays absent', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  s.ctl.corrupt = 200;
  await assert.rejects(t.ensure([199, 200, 201], 3) as Promise<void>, /row 200 failed its checksum/);
  assert.throws(() => t.copyRow(200, t.alloc(W), 0), /row 200 is not loaded/);
  sameRow(t, w.dense, 199);
});

each('a wrong length rejects the call with status, range and url', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  s.ctl.badLength = true;
  await assert.rejects(t.ensure([400], 1) as Promise<void>, /range bytes=\d+-\d+ on https:\/\/words\.test\/words\.bin answered 206 with \d+ bytes, expected \d+/);
});

each('200 on a later range switches to the full table and the call resolves', async (w) => {
  const s = server(w, { fullFrom: 2 });
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  await t.ensure([900], 1);
  const info = t.info();
  assert.equal(info.mode, 'full');
  assert.equal(info.range, false);
  assert.equal(info.fallback, 'no-range');
  sameRow(t, w.dense, 900);
  assert.equal(t.ensure([901], 1), null);
});

each('redirect, then 403 on the signed url: one retry against the original url', async (w) => {
  const s = server(w, { redirect: true, forbidOnce: true });
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  await t.ensure([900], 1);
  assert.equal(s.log[1].url, `${ORIGIN}?sig=1`);
  assert.equal(s.log[1].status, 403);
  assert.equal(s.log[2].url, ORIGIN);
  assert.equal(s.log[2].status, 206);
  assert.equal(s.log.length, 3);
  await t.ensure([901], 1);
  assert.equal(s.log[3].url, `${ORIGIN}?sig=2`);
  sameRow(t, w.dense, 901);
});

each('cache: a second instance finds the rows without network requests', async (w) => {
  const storage = cacheStorage();
  await withCaches(storage, async () => {
    const s1 = server(w);
    const t1 = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s1.fetch });
    await t1.ensure(Array.from({ length: 11 }, (_, k) => 10 + k), 11);
    await sleep(5);
    const s2 = server(w);
    const t2 = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s2.fetch });
    assert.equal(t2.info().cacheRows, 11);
    assert.equal(t2.ensure(Array.from({ length: 11 }, (_, k) => 10 + k), 11), null);
    assert.equal(s2.log.length, 1);
    for (let id = 10; id <= 20; id += 1) sameRow(t2, w.dense, id);
  });
});

each('cache: a corrupted part is skipped', async (w) => {
  const storage = cacheStorage();
  await withCaches(storage, async () => {
    const s1 = server(w);
    const t1 = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s1.fetch });
    await t1.ensure(Array.from({ length: 11 }, (_, k) => 10 + k), 11);
    await sleep(5);
    const part = storage.stores.get(CACHE)?.values().next().value as Uint8Array;
    part[4 + 4 * 11 + 5] ^= 0x01;
    const s2 = server(w);
    const t2 = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s2.fetch });
    assert.equal(t2.info().cacheRows, 0);
    await t2.ensure([10], 1);
    assert.equal(s2.log.length, 2);
    sameRow(t2, w.dense, 10);
  });
});

each('cache: more than 64 parts merge into one', async (w) => {
  const storage = cacheStorage();
  const prefix = `${ORIGIN}?kh=${w.desc.sha256.slice(0, 16)}&part=`;
  const m = new Map<string, Uint8Array>();
  for (let k = 0; k < 65; k += 1) {
    const id = 100 + k;
    const body = new Uint8Array(8 + w.rb);
    const dv = new DataView(body.buffer);
    dv.setUint32(0, 1, true);
    dv.setUint32(4, id, true);
    body.set(w.bytes.subarray(id * w.rb, (id + 1) * w.rb), 8);
    m.set(`${prefix}${k}`, body);
  }
  storage.stores.set(CACHE, m);
  await withCaches(storage, async () => {
    const s = server(w);
    const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s.fetch });
    assert.equal(t.info().cacheRows, 65);
    assert.deepEqual([...m.keys()], [`${prefix}65`]);
    assert.equal(t.ensure(Array.from({ length: 65 }, (_, k) => 100 + k), 65), null);
    sameRow(t, w.dense, 164);
  });
});

each('caches that throw do not fail calls', async (w) => {
  const broken = { open: async () => { throw new Error('storage denied'); } };
  await withCaches(broken, async () => {
    const s = server(w);
    const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s.fetch });
    await t.ensure([50, 51], 2);
    assert.match(String(t.info().cacheError), /storage denied/);
    sameRow(t, w.dense, 51);
  });
});

each('progressive switches to the full table after the background download', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch, progressive: true });
  await t.ensure([50], 1);
  for (let i = 0; i < 200 && t.info().mode !== 'full'; i += 1) await sleep(5);
  assert.equal(t.info().mode, 'full');
  assert.ok(s.log.some((e) => e.range === null && e.status === 200));
  sameRow(t, w.dense, 50);
});

each('progressive arms on a first call whose rows are all cached, and still switches', async (w) => {
  const storage = cacheStorage();
  await withCaches(storage, async () => {
    const first = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: server(w).fetch });
    await first.ensure([50, 51], 2);
    await sleep(20); // the cache write of the first call settles
    const s = server(w);
    const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, fetch: s.fetch, progressive: true });
    assert.equal(t.ensure([50, 51], 2), null);
    for (let i = 0; i < 200 && t.info().mode !== 'full'; i += 1) await sleep(5);
    assert.equal(t.info().mode, 'full');
    assert.ok(s.log.some((e) => e.range === null && e.status === 200));
    sameRow(t, w.dense, 51);
  });
});

each('an id outside the vocabulary throws', async (w) => {
  const s = server(w);
  const t = await RemoteRows.open(ORIGIN, w.desc, { view: wordsView, cache: false, fetch: s.fetch });
  assert.throws(() => t.copyRow(N, t.alloc(W), 0), /token id 3000 is outside the vocabulary of 3000 rows/);
  assert.throws(() => t.ensure([-1], 1), /token id -1 is outside the vocabulary/);
});
