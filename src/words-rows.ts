// RemoteRows (R4 Festlegung 4 to 6): word rows fetched over HTTP Range, checked per row, kept in a
// growing pool, persisted in Cache Storage. Lazy chunk `words-rows`; the core knows only WordTable.

import { checkRow, rowChecksum, type WordsDesc } from './words-format.ts';
import type { FullTable, RowArray, WordTable } from './words.ts';

// The core passes wordsView in (open option view): this chunk imports nothing at runtime from words.ts.
export type WordsView = (buf: ArrayBuffer, desc: WordsDesc) => FullTable;

export interface RemoteRowsOptions {
  view: WordsView; // wordsView of src/words.ts: the whole-file body becomes a FullTable
  progressive?: boolean; // after the first fetch batch, download words.bin whole and switch to it
  cache?: boolean; // Cache Storage persistence, default true when globalThis.caches exists
  fetch?: typeof fetch; // injectable for tests
  probeRows?: number; // rows fetched by the probe at open, default 8
}

const PAGE = 1024; // rows per pool page
const GAP = 4; // ranges merge across gaps of up to this many rows
const MAX_PARTS = 64; // more cache parts than this merge into one
const CACHE = 'kleinhirn-words';

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

// Cache part: u32 LE count, count u32 LE row ids, then the rows with their checksums.
function pack(ids: number[], rows: Uint8Array[], rowBytes: number): Uint8Array<ArrayBuffer> {
  const n = ids.length;
  const body = new Uint8Array(4 + 4 * n + n * rowBytes);
  const dv = new DataView(body.buffer);
  dv.setUint32(0, n, true);
  ids.forEach((id, i) => {
    dv.setUint32(4 + 4 * i, id, true);
    body.set(rows[i], 4 + 4 * n + i * rowBytes);
  });
  return body;
}

export class RemoteRows implements WordTable {
  readonly dtype: 'f16' | 'f32';
  readonly rows: number;
  readonly width: number;
  private readonly rowBytes: number;
  private readonly dataBytes: number;
  private readonly elemBytes: number;
  private readonly salt: number;
  private readonly slot: Int32Array; // row id -> pool slot, -1 when absent
  private pages: RowArray[] = [];
  private count = 0;
  private cacheRows = 0;
  private bytes = 0;
  private requests = 0;
  private url: string;
  private range = true;
  private full: FullTable | null = null;
  private readonly inflight = new Map<number, Promise<void>>();
  private readonly fetchFn: typeof fetch;
  private readonly view: WordsView;
  private useCache: boolean;
  private readonly progressive: boolean;
  private armed = false; // the progressive download is started (or was, once the first call ran)
  private opened = false; // set when open() has settled
  private nextPart = 0; // next cache part number
  private readonly meta: { fallback?: string; cacheError?: string; fullError?: string } = {};

  private constructor(private readonly origin: string, private readonly desc: WordsDesc, opts: RemoteRowsOptions) {
    this.dtype = desc.dtype;
    this.rows = desc.rows;
    this.width = desc.width;
    this.rowBytes = desc.rowBytes;
    this.elemBytes = desc.dtype === 'f16' ? 2 : 4;
    this.dataBytes = desc.width * this.elemBytes;
    this.salt = desc.salt;
    this.slot = new Int32Array(desc.rows).fill(-1);
    this.url = origin;
    this.fetchFn = opts.fetch ?? fetch.bind(globalThis);
    this.view = opts.view;
    this.useCache = opts.cache ?? typeof globalThis.caches !== 'undefined';
    this.progressive = opts.progressive ?? false;
  }

  // Reads the cached parts, then probes the first rows with a Range. A 200 probe means the server
  // ignored Range: the body is the whole file and the instance serves it as a FullTable.
  static async open(url: string, desc: WordsDesc, opts: RemoteRowsOptions): Promise<WordTable> {
    const table = new RemoteRows(url, desc, opts);
    await table.readCache();
    await table.probe(Math.min(opts.probeRows ?? 8, desc.rows));
    table.opened = true;
    return table;
  }

  ensure(ids: ArrayLike<number>, count: number): Promise<void> | null {
    if (this.full) return null;
    // progressive arms on the first call after open, whether it fetches or not
    const arm = this.progressive && this.opened && !this.armed;
    if (arm) this.armed = true;
    const missing: number[] = [];
    const wait: Promise<void>[] = [];
    const seen = new Set<number>();
    for (let k = 0; k < count; k += 1) {
      const id = ids[k];
      this.check(id);
      if (seen.has(id) || this.slot[id] >= 0) continue;
      seen.add(id);
      const flying = this.inflight.get(id);
      if (flying) wait.push(flying);
      else missing.push(id);
    }
    if (missing.length > 0) wait.push(this.fetchRows(missing.sort((a, b) => a - b), arm));
    else if (arm) void this.download();
    return wait.length > 0 ? Promise.all(wait).then(() => undefined) : null;
  }

  copyRow(id: number, out: RowArray, at: number): void {
    this.check(id);
    if (this.full) {
      this.full.copyRow(id, out, at);
      return;
    }
    const s = this.slot[id];
    if (s < 0) throw new Error(`words: row ${id} is not loaded (call ensure first)`);
    const page = this.pages[s >>> 10];
    const off = (s & (PAGE - 1)) * this.width;
    out.set(page.subarray(off, off + this.width), at);
  }

  alloc(elements: number): RowArray {
    return this.dtype === 'f16' ? new Uint16Array(elements) : new Float32Array(elements);
  }

  info(): Record<string, unknown> {
    return {
      mode: this.full ? 'full' : this.progressive ? 'progressive' : 'rows',
      range: this.range,
      rows: this.full ? this.rows : this.count,
      cacheRows: this.cacheRows,
      bytes: this.bytes,
      requests: this.requests,
      ...this.meta,
    };
  }

  private check(id: number): void {
    if (!Number.isInteger(id) || id < 0 || id >= this.rows) {
      throw new Error(`token id ${id} is outside the vocabulary of ${this.rows} rows (ids are integers from 0)`);
    }
  }

  // Copies the data bytes of a checked row into the next free slot; false when the row is present
  // or the table is already full.
  private add(id: number, row: Uint8Array): boolean {
    if (this.full || this.slot[id] >= 0) return false;
    const s = this.count;
    this.count += 1;
    if (s >>> 10 === this.pages.length) this.pages.push(this.alloc(PAGE * this.width));
    const page = this.pages[s >>> 10];
    const at = page.byteOffset + (s & (PAGE - 1)) * this.dataBytes;
    new Uint8Array(page.buffer, at, this.dataBytes).set(row.subarray(0, this.dataBytes));
    this.slot[id] = s;
    return true;
  }

  // One row of the pool with its checksum, for the merged cache part.
  private rowOf(id: number): Uint8Array {
    const s = this.slot[id];
    const page = this.pages[s >>> 10];
    const out = new Uint8Array(this.rowBytes);
    out.set(new Uint8Array(page.buffer, page.byteOffset + (s & (PAGE - 1)) * this.dataBytes, this.dataBytes));
    new DataView(out.buffer).setUint32(this.dataBytes, rowChecksum(out.subarray(0, this.dataBytes), id, this.salt), true);
    return out;
  }

  private noRange(): void {
    this.range = false;
    this.meta.fallback = 'no-range';
  }

  // One request. A redirect moves later requests to res.url; a 403 or 410 on a redirected URL is
  // retried once against the original URL.
  private async request(range?: string): Promise<Response> {
    const sent = this.url;
    const init = range ? { headers: { Range: range } } : undefined;
    let res = await this.net(sent, init);
    if ((res.status === 403 || res.status === 410) && sent !== this.origin) {
      res = await this.net(this.origin, init);
      this.url = res.url || this.origin;
      return res;
    }
    if (res.redirected && res.url && res.url !== sent) this.url = res.url;
    return res;
  }

  private net(url: string, init?: RequestInit): Promise<Response> {
    this.requests += 1;
    return this.fetchFn(url, init);
  }

  private async probe(n: number): Promise<void> {
    const range = `bytes=0-${n * this.rowBytes - 1}`;
    const res = await this.request(range);
    if (res.status === 200) {
      this.noRange();
      await this.adopt(res);
      return;
    }
    if (res.status !== 206) throw new Error(`words: range probe on ${this.url} answered ${res.status}`);
    await this.rowsIn(res, range, 0, n - 1);
  }

  // The body of a 206: exact length, then every row through its checksum into the pool.
  private async rowsIn(res: Response, range: string, first: number, last: number): Promise<[number[], Uint8Array[]]> {
    const buf = new Uint8Array(await res.arrayBuffer());
    this.bytes += buf.byteLength;
    const want = (last - first + 1) * this.rowBytes;
    if (buf.byteLength !== want) {
      throw new Error(`words: range ${range} on ${this.url} answered 206 with ${buf.byteLength} bytes, expected ${want}`);
    }
    const ids: number[] = [];
    const rows: Uint8Array[] = [];
    let bad = -1;
    for (let id = first; id <= last; id += 1) {
      const row = buf.subarray((id - first) * this.rowBytes, (id - first + 1) * this.rowBytes);
      if (!checkRow(row, id, this.rowBytes, this.salt)) {
        if (bad < 0) bad = id;
        continue;
      }
      this.add(id, row);
      ids.push(id);
      rows.push(row);
    }
    if (bad >= 0) throw new Error(`words: row ${bad} failed its checksum (${this.url})`);
    return [ids, rows];
  }

  // Whole-file body: SHA-256 and length against the manifest, then the table is served from it.
  private async adopt(res: Response): Promise<void> {
    if (this.full) return;
    const buf = await res.arrayBuffer();
    if (buf.byteLength !== this.desc.bytes) {
      throw new Error(`words.bin is ${buf.byteLength} bytes, manifest says ${this.desc.bytes}`);
    }
    const hex = [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buf))]
      .map((b) => b.toString(16).padStart(2, '0')).join('');
    if (hex !== this.desc.sha256) throw new Error('sha256 mismatch on words.bin');
    this.full = this.view(buf, this.desc);
    this.pages = [];
  }

  private async download(): Promise<void> {
    if (this.full) return;
    try {
      const res = await this.request();
      if (res.status !== 200) throw new Error(`words: full download of ${this.url} answered ${res.status}`);
      await this.adopt(res);
    } catch (e) {
      this.meta.fullError = message(e);
    }
  }

  // Fetches the missing rows as one call: merged ranges, all sent at once. Rows in flight are
  // shared through this.inflight.
  private fetchRows(missing: number[], arm: boolean): Promise<void> {
    const runs: [number, number][] = [];
    for (const id of missing) {
      const run = runs[runs.length - 1];
      if (run && id - run[1] <= GAP + 1) run[1] = id;
      else runs.push([id, id]);
    }
    let done!: () => void;
    let failed!: (e: unknown) => void;
    const call = new Promise<void>((resolve, reject) => {
      done = resolve;
      failed = reject;
    });
    for (const id of missing) this.inflight.set(id, call);
    Promise.all(runs.map(([a, b]) => this.getRange(a, b))).then(
      (parts) => {
        this.store(parts.flatMap((p) => p[0]), parts.flatMap((p) => p[1]));
        done();
      },
      failed,
    ).finally(() => {
      for (const id of missing) this.inflight.delete(id);
      if (arm) void this.download();
    });
    return call;
  }

  private async getRange(first: number, last: number): Promise<[number[], Uint8Array[]]> {
    const range = `bytes=${first * this.rowBytes}-${(last + 1) * this.rowBytes - 1}`;
    const res = await this.request(range);
    if (res.status === 200) {
      this.noRange();
      await this.adopt(res);
      return [[], []];
    }
    if (res.status !== 206) throw new Error(`words: range ${range} on ${this.url} answered ${res.status}`);
    return this.rowsIn(res, range, first, last);
  }

  private keyPrefix(): string {
    return `${this.origin}?kh=${this.desc.sha256.slice(0, 16)}&part=`;
  }

  // Any exception from caches turns the cache off for this instance and is reported in info().
  private async cacheOp<T>(fn: (c: Cache) => Promise<T>): Promise<T | undefined> {
    if (!this.useCache) return undefined;
    try {
      return await fn(await globalThis.caches.open(CACHE));
    } catch (e) {
      this.useCache = false;
      this.meta.cacheError = message(e);
      return undefined;
    }
  }

  private store(ids: number[], rows: Uint8Array[]): void {
    if (ids.length === 0) return;
    const n = this.nextPart;
    this.nextPart += 1;
    void this.cacheOp((c) => c.put(`${this.keyPrefix()}${n}`, new Response(pack(ids, rows, this.rowBytes))));
  }

  // A part with a bad length or a bad row is skipped whole.
  private parsePart(body: Uint8Array): [number[], Uint8Array[]] | null {
    if (body.byteLength < 4) return null;
    const dv = new DataView(body.buffer, body.byteOffset, body.byteLength);
    const n = dv.getUint32(0, true);
    if (body.byteLength !== 4 + 4 * n + n * this.rowBytes) return null;
    const ids: number[] = [];
    const rows: Uint8Array[] = [];
    for (let i = 0; i < n; i += 1) {
      const id = dv.getUint32(4 + 4 * i, true);
      const at = 4 + 4 * n + i * this.rowBytes;
      const row = body.subarray(at, at + this.rowBytes);
      if (id >= this.rows || !checkRow(row, id, this.rowBytes, this.salt)) return null;
      ids.push(id);
      rows.push(row);
    }
    return [ids, rows];
  }

  private async readCache(): Promise<void> {
    const prefix = this.keyPrefix();
    const keys = await this.cacheOp(async (c) => (await c.keys())
      .map((r) => r.url)
      .filter((u) => u.startsWith(prefix) && /^\d+$/.test(u.slice(prefix.length))));
    if (!keys) return;
    for (const k of keys) this.nextPart = Math.max(this.nextPart, Number(k.slice(prefix.length)) + 1);
    for (const k of keys) {
      const body = await this.cacheOp(async (c) => {
        const r = await c.match(k);
        return r ? new Uint8Array(await r.arrayBuffer()) : null;
      });
      const got = body ? this.parsePart(body) : null;
      if (!got) continue;
      got[0].forEach((id, i) => {
        if (this.add(id, got[1][i])) this.cacheRows += 1;
      });
    }
    if (keys.length > MAX_PARTS) {
      const ids: number[] = [];
      const rows: Uint8Array[] = [];
      for (let id = 0; id < this.rows; id += 1) {
        if (this.slot[id] >= 0) {
          ids.push(id);
          rows.push(this.rowOf(id));
        }
      }
      const n = this.nextPart;
      this.nextPart += 1;
      await this.cacheOp(async (c) => {
        await c.put(`${prefix}${n}`, new Response(pack(ids, rows, this.rowBytes)));
        for (const k of keys) await c.delete(k);
      });
    }
  }
}
