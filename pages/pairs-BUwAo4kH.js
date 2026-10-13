//#region src/tokenizer/strtab.ts
var e = -2128831035;
function t(e, t, n = 0, r = t.length) {
	let i = e;
	for (let e = n; e < r; e += 1) i = Math.imul(i ^ t.charCodeAt(e), 16777619);
	return i;
}
var n = class e {
	blob;
	offs;
	constructor(e, t) {
		this.blob = e, this.offs = t;
	}
	static fromStrings(t) {
		let n = new Uint32Array(t.length + 1), r = "";
		for (let e = 0; e < t.length; e += 1) n[e + 1] = n[e] + t[e].length, r += t[e];
		return new e(r, n);
	}
	get length() {
		return this.offs.length - 1;
	}
	tokenAt(e) {
		return this.blob.slice(this.offs[e], this.offs[e + 1]);
	}
}, r = class {
	slots;
	hashes;
	mask;
	tokens;
	constructor(n) {
		this.tokens = n;
		let r = n.length, i = 16;
		for (; i < 2 * r;) i *= 2;
		this.mask = i - 1, this.slots = new Int32Array(i), this.hashes = new Int32Array(r);
		let { blob: a, offs: o } = n;
		for (let n = 0; n < r; n += 1) {
			let r = t(e, a, o[n], o[n + 1]);
			this.hashes[n] = r;
			let i = this.slotOf(r);
			for (;;) {
				let e = this.slots[i] - 1;
				if (e < 0) {
					this.slots[i] = n + 1;
					break;
				}
				if (this.hashes[e] === r && this.sameRange(e, n)) {
					this.slots[i] = n + 1;
					break;
				}
				i = i + 1 & this.mask;
			}
		}
	}
	slotOf(e) {
		return (e ^ e >>> 13) & this.mask;
	}
	sameRange(e, t) {
		let { blob: n, offs: r } = this.tokens, i = r[e + 1] - r[e];
		if (i !== r[t + 1] - r[t]) return !1;
		for (let a = 0; a < i; a += 1) if (n.charCodeAt(r[e] + a) !== n.charCodeAt(r[t] + a)) return !1;
		return !0;
	}
	indexOf(n) {
		let r = t(e, n), { blob: i, offs: a } = this.tokens, o = this.slotOf(r);
		for (;;) {
			let e = this.slots[o] - 1;
			if (e < 0) return -1;
			if (this.hashes[e] === r && a[e + 1] - a[e] === n.length) {
				let t = 0, r = a[e];
				for (; t < n.length && i.charCodeAt(r + t) === n.charCodeAt(t);) t += 1;
				if (t === n.length) return e;
			}
			o = o + 1 & this.mask;
		}
	}
	indexOfConcat(n, r, i) {
		let { blob: a, offs: o } = this.tokens, s = o[n], c = o[n + 1] - s, l = o[r] + i, u = o[r + 1] - l, d = c + u, f = t(e, a, s, s + c);
		f = t(f, a, l, l + u);
		let p = this.slotOf(f);
		for (;;) {
			let e = this.slots[p] - 1;
			if (e < 0) return -1;
			if (this.hashes[e] === f && o[e + 1] - o[e] === d) {
				let t = o[e], n = 0;
				for (; n < c && a.charCodeAt(t + n) === a.charCodeAt(s + n);) n += 1;
				if (n === c) {
					for (; n < d && a.charCodeAt(t + n) === a.charCodeAt(l + n - c);) n += 1;
					if (n === d) return e;
				}
			}
			p = p + 1 & this.mask;
		}
	}
	indexOfChars(e, t, n, r) {
		let { blob: i, offs: a } = this.tokens, o = 0;
		for (let r = t; r < n; r += 1) o += e[r].length;
		let s = this.slotOf(r);
		for (;;) {
			let c = this.slots[s] - 1;
			if (c < 0) return -1;
			if (this.hashes[c] === r && a[c + 1] - a[c] === o) {
				let r = a[c], o = !0;
				for (let a = t; a < n && o; a += 1) {
					let t = e[a];
					for (let e = 0; e < t.length; e += 1) {
						if (i.charCodeAt(r) !== t.charCodeAt(e)) {
							o = !1;
							break;
						}
						r += 1;
					}
				}
				if (o) return c;
			}
			s = s + 1 & this.mask;
		}
	}
}, i = new TextDecoder("utf-8", {
	fatal: !0,
	ignoreBOM: !0
}), a = /* @__PURE__ */ new Float64Array(1), o = new Uint32Array(a.buffer), s = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 0 : 1, c = 1 - s;
function l(e, t) {
	a[0] = e;
	let n = o[s], r = o[c];
	return e > 0 === t ? (n = n + 1 >>> 0, n === 0 && (r = r + 1 >>> 0)) : (n === 0 && (r = r - 1 >>> 0), n = n - 1 >>> 0), o[s] = n, o[c] = r, a[0];
}
function u(e, t, n) {
	t === "__proto__" ? Object.defineProperty(e, t, {
		value: n,
		writable: !0,
		enumerable: !0,
		configurable: !0
	}) : e[t] = n;
}
async function d(e) {
	if (typeof DecompressionStream > "u") throw Error("tokenizer.bin needs DecompressionStream (Chromium 80, Safari 16.4, Firefox 113)");
	let t = new Blob([e]).stream().pipeThrough(new DecompressionStream("gzip"));
	return new Uint8Array(await new Response(t).arrayBuffer());
}
var f = class {
	tokens;
	ids;
	constructor(e, t) {
		this.tokens = e, this.ids = t;
	}
}, p = class {
	pieces;
	scores;
	constructor(e, t) {
		this.pieces = e, this.scores = t;
	}
}, m = class {
	tokens;
	pairs;
	form;
	constructor(e, t, n) {
		this.tokens = e, this.pairs = t, this.form = n;
	}
}, h = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
function g(e, t) {
	let r = new TextEncoder().encode(t), a = 0, o = 0;
	for (let t = 0; t < e.length; a += 1) {
		let n = v(e, t), i = Math.floor(n / 2);
		o += i + (n & 1 ? r.length : 0), t = _ + i;
	}
	let s = new Uint8Array(o), c = new Uint32Array(a + 1), l = 0, u = 0;
	for (let n = 0; n < a; n += 1) {
		let i = v(e, u), a = Math.floor(i / 2), o = _, d = 0;
		if (i & 1) {
			for (let e = 0; e < r.length; e += 1) s[l++] = r[e];
			d = t.length;
		}
		for (let t = o; t < o + a; t += 1) {
			let n = e[t];
			s[l++] = n, (n & 192) != 128 && (d += n >= 240 ? 2 : 1);
		}
		c[n + 1] = c[n] + d, u = o + a;
	}
	return new n(i.decode(s), c);
}
var _ = 0;
function v(e, t) {
	let n = 0, r = 1, i = t, a;
	do
		a = e[i], i += 1, n += (a & 127) * r, r *= 128;
	while (a & 128);
	return _ = i, n;
}
function y(e) {
	let t = 0;
	for (let n = 0; n < e.length; n += 1) e[n] & 128 || (t += 1);
	let n = new Int32Array(t), r = 0;
	for (let i = 0; i < t; i += 1) {
		let t = v(e, r);
		if (r = _, t > 2147483647) throw Error("tokenizer.bin: id beyond int32");
		n[i] = t;
	}
	return n;
}
function b(e) {
	let t = 0;
	for (let n = 0; n < e.length; n += 1) e[n] & 128 || (t += 1);
	let n = new Uint32Array(t), r = 0;
	for (let i = 0; i < t; i += 1) n[i] = v(e, r), r = _;
	return n;
}
function x(e, t, n) {
	let r = t.length, i = new Uint8Array(4 * r);
	for (let t = 0; t < 4; t += 1) {
		let n = t * r;
		for (let a = 0; a < r; a += 1) i[4 * a + t] = e[n + a];
	}
	let a = new Float32Array(i.buffer), o = new DataView(n.buffer, n.byteOffset, n.byteLength), s = new Float64Array(r), c = 0;
	for (let e = 0; e < r; e += 1) {
		let n = t[e];
		if (n === 3) {
			s[e] = o.getFloat64(c, !0), c += 8;
			continue;
		}
		let r = a[e];
		s[e] = n === 0 ? r : l(r, n === 1);
	}
	return s;
}
function S(e) {
	if (!h) throw Error("tokenizer.bin: the fast path needs a little-endian host");
	if (new TextDecoder().decode(e.subarray(0, 4)) !== "KHT1") throw Error("tokenizer.bin: bad magic");
	let t = new DataView(e.buffer, e.byteOffset, e.byteLength).getUint32(4, !0), n = JSON.parse(i.decode(e.subarray(8, 8 + t))), r = 8 + t, a = [];
	for (let [t, o] of n.fields.entries()) {
		let s = o.sections.map((t) => {
			let n = e.subarray(r, r + t);
			return r += t, n;
		}), c;
		if (o.kind === "dict") {
			let e = g(s[0], o.prefix ?? "");
			c = new f(e, o.dense ? null : y(s[1])), a[t] = e;
		} else if (o.kind === "unigram") {
			let e = g(s[0], o.prefix ?? ""), n = x(s[1], s[2], s[3]);
			if (n.length !== e.length) throw Error("tokenizer.bin: unigram sections disagree");
			a[t] = e, c = new p(e, n);
		} else c = o.kind === "merges" ? new m(a[o.of], b(s[0]), o.form === "pair" ? "pair" : "str") : o.kind === "base64" ? s[0] : JSON.parse(i.decode(s[0]));
		let l = n.head;
		for (let e of o.path.slice(0, -1)) l = l[e];
		let d = o.path[o.path.length - 1];
		typeof d == "string" ? u(l, d, c) : l[d] = c;
	}
	if (r !== e.length) throw Error("tokenizer.bin: trailing bytes");
	return n.head;
}
function C(e) {
	if (e instanceof f) return e;
	let t = e, r = Object.keys(t);
	return new f(n.fromStrings(r), r.map((e) => t[e]));
}
function w(e) {
	if (e instanceof p) return e;
	let t = e;
	return new p(n.fromStrings(t.map((e) => e[0])), t.map((e) => e[1]));
}
function T(e, t) {
	if (t instanceof m) {
		if (t.tokens !== e.tokens) throw Error("BPE: merges do not index the vocab of this tokenizer");
		return t.pairs;
	}
	let n = new r(e.tokens), i = t, a = new Uint32Array(i.length * 2);
	return i.forEach((e, t) => {
		let r, i;
		if (typeof e != "string") [r, i] = e;
		else {
			let t = e.split(" ");
			if (t.length !== 2) throw Error(`BPE: invalid merge ${e}`);
			[r, i] = t;
		}
		let o = n.indexOf(r), s = n.indexOf(i);
		if (o < 0) throw Error(`BPE: merge token ${r} is not in the vocab`);
		if (s < 0) throw Error(`BPE: merge token ${i} is not in the vocab`);
		a[2 * t] = o, a[2 * t + 1] = s;
	}), a;
}
function E(e, t) {
	return e.ids ? e.ids[t] : t;
}
async function D(e) {
	return e[0] === 31 ? S(await d(e)) : JSON.parse(new TextDecoder().decode(e));
}
var O = class {
	dict;
	index;
	oneUnit;
	byId;
	byIdMap;
	constructor(e) {
		this.dict = e, this.index = new r(e.tokens), this.oneUnit = (/* @__PURE__ */ new Int32Array(65536)).fill(-1);
		let { blob: t, offs: n } = e.tokens;
		for (let r = 0; r < e.tokens.length; r += 1) n[r + 1] - n[r] === 1 && (this.oneUnit[t.charCodeAt(n[r])] = r);
		if (e.ids === null) {
			this.byId = null, this.byIdMap = null;
			return;
		}
		let i = 0;
		for (let t = 0; t < e.tokens.length; t += 1) {
			let n = E(e, t);
			Number.isInteger(n) && n > i && (i = n);
		}
		if (i < 2 ** 22) {
			this.byId = new Int32Array(i + 1).fill(-1), this.byIdMap = null;
			for (let t = 0; t < e.tokens.length; t += 1) {
				let n = E(e, t);
				Number.isInteger(n) && n >= 0 && (this.byId[n] = t);
			}
		} else {
			this.byId = null, this.byIdMap = /* @__PURE__ */ new Map();
			for (let t = 0; t < e.tokens.length; t += 1) this.byIdMap.set(E(e, t), t);
		}
	}
	id(e) {
		let t = this.index.indexOf(e);
		return t < 0 ? void 0 : E(this.dict, t);
	}
	has(e) {
		return this.index.indexOf(e) >= 0;
	}
	idOfChar(e) {
		if (e.length !== 1) return this.id(e);
		let t = this.oneUnit[e.charCodeAt(0)];
		return t < 0 ? void 0 : E(this.dict, t);
	}
	idOfConcat(e, t, n) {
		let r = this.index.indexOfConcat(e, t, n);
		return r < 0 ? void 0 : E(this.dict, r);
	}
	tokenOfId(e) {
		if (this.byIdMap !== null) {
			let t = this.byIdMap.get(e);
			return t === void 0 ? void 0 : this.dict.tokens.tokenAt(t);
		}
		if (this.byId === null) return e >= 0 && e < this.dict.tokens.length ? this.dict.tokens.tokenAt(e) : void 0;
		let t = e >= 0 && e < this.byId.length ? this.byId[e] : -1;
		return t < 0 ? void 0 : this.dict.tokens.tokenAt(t);
	}
}, k = class {
	mask;
	keys;
	vals;
	count = 0;
	constructor(e) {
		let t = 16;
		for (; t * 2 < e * 3;) t *= 2;
		this.alloc(t);
	}
	alloc(e) {
		this.mask = e - 1, this.keys = new Int32Array(e * 2), this.vals = new Int32Array(e).fill(-1);
	}
	grow() {
		let e = this.keys, t = this.vals;
		this.alloc(this.vals.length * 2);
		for (let n = 0; n < t.length; n += 1) t[n] !== -1 && this.place(e[2 * n], e[2 * n + 1], t[n]);
	}
	place(e, t, n) {
		let r = this.slot(e, t);
		for (; this.vals[r] !== -1;) r = r + 1 & this.mask;
		this.keys[2 * r] = e, this.keys[2 * r + 1] = t, this.vals[r] = n;
	}
	slot(e, t) {
		let n = Math.imul(e ^ Math.imul(t, 2654435761), 2246822519);
		return n ^= n >>> 15, n & this.mask;
	}
	set(e, t, n) {
		let r = this.slot(e, t);
		for (;;) {
			if (this.vals[r] === -1) {
				this.count += 1, this.place(e, t, n), this.count * 3 > this.vals.length * 2 && this.grow();
				return;
			}
			if (this.keys[2 * r] === e && this.keys[2 * r + 1] === t) {
				this.vals[r] = n;
				return;
			}
			r = r + 1 & this.mask;
		}
	}
	get(e, t) {
		let n = this.slot(e, t);
		for (;;) {
			let r = this.vals[n];
			if (r === -1) return -1;
			if (this.keys[2 * n] === e && this.keys[2 * n + 1] === t) return r;
			n = n + 1 & this.mask;
		}
	}
};
//#endregion
export { w as a, e as c, T as i, r as l, O as n, E as o, C as r, D as s, k as t, t as u };
