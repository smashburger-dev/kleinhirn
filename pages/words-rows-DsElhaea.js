//#region src/words-format.ts
var e = (e, t) => e << t | e >>> 32 - t;
function t(t, n, r) {
	if (t.byteLength % 4 != 0) throw Error(`words: row of ${t.byteLength} bytes is not whole u32 words`);
	let i = new DataView(t.buffer, t.byteOffset, t.byteLength), a = (r ^ Math.imul(n + 1, 2654435761)) >>> 0;
	for (let n = 0; n < t.byteLength; n += 4) {
		let t = i.getUint32(n, !0);
		t = Math.imul(t, 3432918353), t = e(t, 15), t = Math.imul(t, 461845907), a ^= t, a = e(a, 13), a = Math.imul(a, 5) + 3864292196 | 0;
	}
	return a ^= t.byteLength, a ^= a >>> 16, a = Math.imul(a, 2246822507), a ^= a >>> 13, a = Math.imul(a, 3266489909), a ^= a >>> 16, a >>> 0;
}
function n(e, n, r, i) {
	if (e.byteLength !== r) return !1;
	let a = new DataView(e.buffer, e.byteOffset, e.byteLength).getUint32(r - 4, !0);
	return t(e.subarray(0, r - 4), n, i) === a;
}
//#endregion
//#region src/words-rows.ts
var r = 1024, i = 64, a = "kleinhirn-words", o = (e) => e instanceof Error ? e.message : String(e);
function s(e, t, n) {
	let r = e.length, i = new Uint8Array(4 + 4 * r + r * n), a = new DataView(i.buffer);
	return a.setUint32(0, r, !0), e.forEach((e, o) => {
		a.setUint32(4 + 4 * o, e, !0), i.set(t[o], 4 + 4 * r + o * n);
	}), i;
}
var c = class e {
	origin;
	desc;
	dtype;
	rows;
	width;
	rowBytes;
	dataBytes;
	elemBytes;
	salt;
	slot;
	pages = [];
	count = 0;
	cacheRows = 0;
	bytes = 0;
	requests = 0;
	url;
	range = !0;
	full = null;
	inflight = /* @__PURE__ */ new Map();
	fetchFn;
	view;
	useCache;
	progressive;
	armed = !1;
	opened = !1;
	nextPart = 0;
	meta = {};
	constructor(e, t, n) {
		this.origin = e, this.desc = t, this.dtype = t.dtype, this.rows = t.rows, this.width = t.width, this.rowBytes = t.rowBytes, this.elemBytes = t.dtype === "f16" ? 2 : 4, this.dataBytes = t.width * this.elemBytes, this.salt = t.salt, this.slot = new Int32Array(t.rows).fill(-1), this.url = e, this.fetchFn = n.fetch ?? fetch.bind(globalThis), this.view = n.view, this.useCache = n.cache ?? globalThis.caches !== void 0, this.progressive = n.progressive ?? !1;
	}
	static async open(t, n, r) {
		let i = new e(t, n, r);
		return await i.readCache(), await i.probe(Math.min(r.probeRows ?? 8, n.rows)), i.opened = !0, i;
	}
	ensure(e, t) {
		if (this.full) return null;
		let n = this.progressive && this.opened && !this.armed;
		n && (this.armed = !0);
		let r = [], i = [], a = /* @__PURE__ */ new Set();
		for (let n = 0; n < t; n += 1) {
			let t = e[n];
			if (this.check(t), a.has(t) || this.slot[t] >= 0) continue;
			a.add(t);
			let o = this.inflight.get(t);
			o ? i.push(o) : r.push(t);
		}
		return r.length > 0 ? i.push(this.fetchRows(r.sort((e, t) => e - t), n)) : n && this.download(), i.length > 0 ? Promise.all(i).then(() => void 0) : null;
	}
	copyRow(e, t, n) {
		if (this.check(e), this.full) {
			this.full.copyRow(e, t, n);
			return;
		}
		let r = this.slot[e];
		if (r < 0) throw Error(`words: row ${e} is not loaded (call ensure first)`);
		let i = this.pages[r >>> 10], a = (r & 1023) * this.width;
		t.set(i.subarray(a, a + this.width), n);
	}
	alloc(e) {
		return this.dtype === "f16" ? new Uint16Array(e) : new Float32Array(e);
	}
	info() {
		return {
			mode: this.full ? "full" : this.progressive ? "progressive" : "rows",
			range: this.range,
			rows: this.full ? this.rows : this.count,
			cacheRows: this.cacheRows,
			bytes: this.bytes,
			requests: this.requests,
			...this.meta
		};
	}
	check(e) {
		if (!Number.isInteger(e) || e < 0 || e >= this.rows) throw Error(`token id ${e} is outside the vocabulary of ${this.rows} rows (ids are integers from 0)`);
	}
	add(e, t) {
		if (this.full || this.slot[e] >= 0) return !1;
		let n = this.count;
		this.count += 1, n >>> 10 === this.pages.length && this.pages.push(this.alloc(r * this.width));
		let i = this.pages[n >>> 10], a = i.byteOffset + (n & 1023) * this.dataBytes;
		return new Uint8Array(i.buffer, a, this.dataBytes).set(t.subarray(0, this.dataBytes)), this.slot[e] = n, !0;
	}
	rowOf(e) {
		let n = this.slot[e], r = this.pages[n >>> 10], i = new Uint8Array(this.rowBytes);
		return i.set(new Uint8Array(r.buffer, r.byteOffset + (n & 1023) * this.dataBytes, this.dataBytes)), new DataView(i.buffer).setUint32(this.dataBytes, t(i.subarray(0, this.dataBytes), e, this.salt), !0), i;
	}
	noRange() {
		this.range = !1, this.meta.fallback = "no-range";
	}
	async request(e) {
		let t = this.url, n = e ? { headers: { Range: e } } : void 0, r = await this.net(t, n);
		return (r.status === 403 || r.status === 410) && t !== this.origin ? (r = await this.net(this.origin, n), this.url = r.url || this.origin, r) : (r.redirected && r.url && r.url !== t && (this.url = r.url), r);
	}
	net(e, t) {
		return this.requests += 1, this.fetchFn(e, t);
	}
	async probe(e) {
		let t = `bytes=0-${e * this.rowBytes - 1}`, n = await this.request(t);
		if (n.status === 200) {
			this.noRange(), await this.adopt(n);
			return;
		}
		if (n.status !== 206) throw Error(`words: range probe on ${this.url} answered ${n.status}`);
		await this.rowsIn(n, t, 0, e - 1);
	}
	async rowsIn(e, t, r, i) {
		let a = new Uint8Array(await e.arrayBuffer());
		this.bytes += a.byteLength;
		let o = (i - r + 1) * this.rowBytes;
		if (a.byteLength !== o) throw Error(`words: range ${t} on ${this.url} answered 206 with ${a.byteLength} bytes, expected ${o}`);
		let s = [], c = [], l = -1;
		for (let e = r; e <= i; e += 1) {
			let t = a.subarray((e - r) * this.rowBytes, (e - r + 1) * this.rowBytes);
			if (!n(t, e, this.rowBytes, this.salt)) {
				l < 0 && (l = e);
				continue;
			}
			this.add(e, t), s.push(e), c.push(t);
		}
		if (l >= 0) throw Error(`words: row ${l} failed its checksum (${this.url})`);
		return [s, c];
	}
	async adopt(e) {
		if (this.full) return;
		let t = await e.arrayBuffer();
		if (t.byteLength !== this.desc.bytes) throw Error(`words.bin is ${t.byteLength} bytes, manifest says ${this.desc.bytes}`);
		if ([...new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", t))].map((e) => e.toString(16).padStart(2, "0")).join("") !== this.desc.sha256) throw Error("sha256 mismatch on words.bin");
		this.full = this.view(t, this.desc), this.pages = [];
	}
	async download() {
		if (!this.full) try {
			let e = await this.request();
			if (e.status !== 200) throw Error(`words: full download of ${this.url} answered ${e.status}`);
			await this.adopt(e);
		} catch (e) {
			this.meta.fullError = o(e);
		}
	}
	fetchRows(e, t) {
		let n = [];
		for (let t of e) {
			let e = n[n.length - 1];
			e && t - e[1] <= 5 ? e[1] = t : n.push([t, t]);
		}
		let r, i, a = new Promise((e, t) => {
			r = e, i = t;
		});
		for (let t of e) this.inflight.set(t, a);
		return Promise.all(n.map(([e, t]) => this.getRange(e, t))).then((e) => {
			this.store(e.flatMap((e) => e[0]), e.flatMap((e) => e[1])), r();
		}, i).finally(() => {
			for (let t of e) this.inflight.delete(t);
			t && this.download();
		}), a;
	}
	async getRange(e, t) {
		let n = `bytes=${e * this.rowBytes}-${(t + 1) * this.rowBytes - 1}`, r = await this.request(n);
		if (r.status === 200) return this.noRange(), await this.adopt(r), [[], []];
		if (r.status !== 206) throw Error(`words: range ${n} on ${this.url} answered ${r.status}`);
		return this.rowsIn(r, n, e, t);
	}
	keyPrefix() {
		return `${this.origin}?kh=${this.desc.sha256.slice(0, 16)}&part=`;
	}
	async cacheOp(e) {
		if (this.useCache) try {
			return await e(await globalThis.caches.open(a));
		} catch (e) {
			this.useCache = !1, this.meta.cacheError = o(e);
			return;
		}
	}
	store(e, t) {
		if (e.length === 0) return;
		let n = this.nextPart;
		this.nextPart += 1, this.cacheOp((r) => r.put(`${this.keyPrefix()}${n}`, new Response(s(e, t, this.rowBytes))));
	}
	parsePart(e) {
		if (e.byteLength < 4) return null;
		let t = new DataView(e.buffer, e.byteOffset, e.byteLength), r = t.getUint32(0, !0);
		if (e.byteLength !== 4 + 4 * r + r * this.rowBytes) return null;
		let i = [], a = [];
		for (let o = 0; o < r; o += 1) {
			let s = t.getUint32(4 + 4 * o, !0), c = 4 + 4 * r + o * this.rowBytes, l = e.subarray(c, c + this.rowBytes);
			if (s >= this.rows || !n(l, s, this.rowBytes, this.salt)) return null;
			i.push(s), a.push(l);
		}
		return [i, a];
	}
	async readCache() {
		let e = this.keyPrefix(), t = await this.cacheOp(async (t) => (await t.keys()).map((e) => e.url).filter((t) => t.startsWith(e) && /^\d+$/.test(t.slice(e.length))));
		if (t) {
			for (let n of t) this.nextPart = Math.max(this.nextPart, Number(n.slice(e.length)) + 1);
			for (let e of t) {
				let t = await this.cacheOp(async (t) => {
					let n = await t.match(e);
					return n ? new Uint8Array(await n.arrayBuffer()) : null;
				}), n = t ? this.parsePart(t) : null;
				n && n[0].forEach((e, t) => {
					this.add(e, n[1][t]) && (this.cacheRows += 1);
				});
			}
			if (t.length > i) {
				let n = [], r = [];
				for (let e = 0; e < this.rows; e += 1) this.slot[e] >= 0 && (n.push(e), r.push(this.rowOf(e)));
				let i = this.nextPart;
				this.nextPart += 1, await this.cacheOp(async (a) => {
					await a.put(`${e}${i}`, new Response(s(n, r, this.rowBytes)));
					for (let e of t) await a.delete(e);
				});
			}
		}
	}
};
//#endregion
export { c as RemoteRows };
