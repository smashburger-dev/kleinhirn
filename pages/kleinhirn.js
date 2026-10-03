//#region src/device.ts
var e = {
	maxComputeInvocationsPerWorkgroup: 256,
	maxComputeWorkgroupSizeX: 256,
	maxComputeWorkgroupSizeY: 256,
	maxComputeWorkgroupSizeZ: 64,
	maxComputeWorkgroupsPerDimension: 65535,
	maxComputeWorkgroupStorageSize: 16384,
	maxStorageBuffersPerShaderStage: 8,
	maxStorageBufferBindingSize: 134217728,
	maxBindGroups: 4,
	maxBufferSize: 268435456
};
async function t(t, n) {
	if (!navigator.gpu) throw Error("WebGPU unavailable: navigator.gpu missing");
	let r = await navigator.gpu.requestAdapter();
	if (!r) throw Error("WebGPU unavailable: no adapter");
	let i = r.features.has("shader-f16"), a = r.features.has("timestamp-query"), o = [];
	t && i && o.push("shader-f16"), a && o.push("timestamp-query");
	let s = await r.requestDevice({
		requiredFeatures: o,
		...n ? { requiredLimits: e } : {}
	}), c = r.info ?? {}, l = {
		adapter: r,
		device: s,
		hasF16: o.includes("shader-f16"),
		hasTimestamps: o.includes("timestamp-query"),
		limitsMode: n ? "minimum" : "default",
		adapterInfo: {
			vendor: c.vendor ?? "",
			architecture: c.architecture ?? "",
			device: c.device ?? "",
			description: c.description ?? ""
		},
		failure: null
	};
	return s.addEventListener("uncapturederror", (e) => {
		l.failure ??= `WebGPU error: ${e.error.message}`;
	}), s.lost.then((e) => {
		l.failure ??= `WebGPU device lost (${e.reason}): ${e.message}`;
	}), l;
}
function n(e) {
	if (e.failure) throw Error(e.failure);
}
async function r(e, t, r) {
	n(e);
	let { device: i } = e;
	i.pushErrorScope("validation");
	let a, o;
	try {
		a = t();
	} finally {
		o = i.popErrorScope(), o.catch(() => {});
	}
	let [s, c] = await Promise.allSettled([r(a), o]);
	if (c.status === "rejected") throw c.reason;
	if (c.value) throw Error(`WebGPU validation error: ${c.value.message}`);
	if (n(e), s.status === "rejected") throw s.reason;
	return s.value;
}
function i(e, t) {
	return r(e, t, async (e) => e);
}
//#endregion
//#region src/weights.ts
function a(e) {
	return [...new Uint8Array(e)].map((e) => e.toString(16).padStart(2, "0")).join("");
}
async function o(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	let n = performance.now();
	return {
		buf: await t.arrayBuffer(),
		bodyMs: performance.now() - n
	};
}
async function s(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	return t.json();
}
async function c(e, t, n, r = s) {
	if ((t ?? "auto") !== "auto" || n.recommendedPrecision !== "f32") return {
		url: e,
		manifest: n
	};
	let i = e.replace(/\/f16\/manifest\.json(\?.*)?$/, "/f32/manifest.json$1");
	if (i === e) return {
		url: e,
		manifest: n,
		note: "manifest recommends f32, but its URL is not .../f16/manifest.json"
	};
	try {
		return {
			url: i,
			manifest: await r(i)
		};
	} catch (t) {
		return {
			url: e,
			manifest: n,
			note: `manifest recommends f32, f32 manifest not available: ${String(t instanceof Error ? t.message : t)}`
		};
	}
}
async function l(e, t) {
	let n = e.slice(0, e.lastIndexOf("/") + 1), r = t ?? await s(e), i = new TextEncoder().encode(JSON.stringify(r)).byteLength, c = 0, l = 0, u = performance.now(), d = await Promise.all(r.shards.map(async (e) => {
		let { buf: t, bodyMs: r } = await o(n + e.file);
		l += r;
		let s = performance.now(), u = a(await crypto.subtle.digest("SHA-256", t));
		if (c += performance.now() - s, u !== e.sha256) throw Error(`sha256 mismatch on ${e.file}`);
		return i += t.byteLength, t;
	})), f = performance.now() - u;
	return {
		manifest: r,
		shardBytes: d,
		downloadBytes: i,
		fetchMs: f,
		bodyMs: l,
		sha256Ms: c
	};
}
async function u(t, n, r) {
	let { manifest: i, shardBytes: a, downloadBytes: o, fetchMs: s, bodyMs: c, sha256Ms: u } = await l(n, r), d = t.limits?.maxBufferSize ?? e.maxBufferSize;
	for (let e of i.tensors) {
		let t = Math.ceil(e.byteLength / 4) * 4;
		if (!e.keepOnCpu && t > d) throw Error(`tensor ${e.name} needs ${t} B, maxBufferSize is ${d}`);
	}
	let f = /* @__PURE__ */ new Map(), p = /* @__PURE__ */ new Map(), m = 0, h = performance.now();
	for (let e of i.tensors) {
		let n = new Uint8Array(a[e.shard], e.offset, e.byteLength);
		if (e.keepOnCpu) {
			let t = p.get(e.name);
			t || (t = {
				parts: [],
				dtype: e.dtype
			}, p.set(e.name, t)), t.parts.push(n.slice(0));
			continue;
		}
		let r = Math.ceil(e.byteLength / 4) * 4, i = t.createBuffer({
			size: r,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		if (e.byteLength % 4 == 0) t.queue.writeBuffer(i, 0, n);
		else {
			let e = new Uint8Array(r);
			e.set(n), t.queue.writeBuffer(i, 0, e);
		}
		m += r, f.set(e.name, i);
	}
	let g = performance.now() - h, _ = p.get("embeddings.word.weight"), v = performance.now(), y = /* @__PURE__ */ new Float32Array();
	if (_) {
		let e = _.parts.reduce((e, t) => e + t.byteLength, 0), t = new Uint8Array(e), n = 0;
		for (let e of _.parts) t.set(e, n), n += e.byteLength;
		y = _.dtype === "f16" ? new Uint16Array(t.buffer) : new Float32Array(t.buffer);
	}
	let b = performance.now() - v;
	return {
		manifest: i,
		tensors: f,
		embeddings: y,
		downloadBytes: o,
		gpuBytes: m,
		timing: {
			fetchMs: s,
			bodyMs: c,
			sha256Ms: u,
			uploadMs: g,
			embedJoinMs: b
		}
	};
}
//#endregion
//#region src/tokenizer/unigram.ts
var d = 10;
function f(e, t) {
	let n = {
		children: /* @__PURE__ */ new Map(),
		id: -1
	}, r = [], i = /* @__PURE__ */ new Map(), a = Infinity;
	for (let [t, o] of e) {
		let e = r.length;
		r.push(o), i.set(t, e);
		let s = n;
		for (let e of t) {
			let t = e.codePointAt(0), n = s.children.get(t);
			n || (n = {
				children: /* @__PURE__ */ new Map(),
				id: -1
			}, s.children.set(t, n)), s = n;
		}
		s.id = e, o < a && (a = o);
	}
	return {
		root: n,
		scores: r,
		ids: i,
		unkScore: a - d,
		unkId: t
	};
}
function p(e, t) {
	let n = [...t], r = n.length;
	if (r === 0) return [];
	let i = Array(r + 1).fill(-Infinity), a = Array(r + 1).fill(-1), o = Array(r + 1).fill(0);
	i[0] = 0;
	for (let t = 0; t < r; t += 1) {
		if (i[t] === -Infinity) continue;
		let s = e.root, c = !1;
		for (let l = t; l < r; l += 1) {
			let r = s.children.get(n[l].codePointAt(0));
			if (!r) break;
			if (s = r, s.id >= 0) {
				l === t && (c = !0);
				let n = i[t] + e.scores[s.id];
				n > i[l + 1] && (i[l + 1] = n, a[l + 1] = s.id, o[l + 1] = t);
			}
		}
		if (!c) {
			let n = i[t] + e.unkScore;
			n > i[t + 1] && (i[t + 1] = n, a[t + 1] = -2, o[t + 1] = t);
		}
	}
	let s = [];
	for (let t = r; t > 0; t = o[t]) s.push(a[t] === -2 ? e.unkId : a[t]);
	let c = [], l = !1;
	for (let t of s.reverse()) {
		if (t === e.unkId) {
			if (l) continue;
			l = !0;
		} else l = !1;
		c.push(t);
	}
	return c;
}
//#endregion
//#region src/tokenizer/tokenizer.ts
var m = /\s{2,}|[\n\r\t]/g, h = "▁", g = class {
	unigram;
	added;
	addedList;
	padTokenId = 0;
	constructor(e) {
		this.unigram = f(e.model.vocab, e.model.unk_id ?? 3), this.added = /* @__PURE__ */ new Map(), this.addedList = [];
		for (let t of e.added_tokens) this.added.set(t.content, t.id), this.addedList.push(t.content), t.content === "[PAD]" && (this.padTokenId = t.id);
		this.addedList.sort((e, t) => t.length - e.length);
	}
	normalize(e) {
		return e.replace(m, " ").normalize("NFC").replace(/\s+$/, "");
	}
	encodePretoken(e) {
		if (this.added.has(e)) return [this.added.get(e)];
		let t = [], n = 0;
		for (let r = 0; r <= e.length; r += 1) (r < e.length && e[r] === " " || r === e.length) && (r > n && t.push(...p(this.unigram, h + e.slice(n, r))), n = r + 1);
		return t;
	}
	encodeIds(e) {
		let t = this.normalize(e), n = [], r = 0;
		for (; r < t.length;) {
			let e = -1, i = 0;
			for (let n of this.addedList) {
				let a = t.indexOf(n, r);
				a !== -1 && (e === -1 || a < e) && (e = a, i = n.length);
			}
			let a = e === -1 ? t.length : e;
			if (a > r && n.push(...this.encodePretoken(t.slice(r, a))), e === -1) break;
			n.push(this.added.get(t.slice(e, e + i))), r = e + i;
		}
		return n;
	}
}, _ = class extends Error {
	constructor(e) {
		super(e), this.name = "BucketOverflowError";
	}
}, v = "[SEP_STRUCT]", y = "[SEP_TEXT]", b = "[P]", x = "[L]", S = /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]/giu;
function C(e) {
	let t = [];
	S.lastIndex = 0;
	for (let n of e.matchAll(S)) t.push(n[0].toLowerCase());
	return t;
}
function w(e, t, n, r, i) {
	let a = n.reduce((e, [, t]) => e + t.length, 0);
	if (a < 1 || a > i) throw new _(`expected 1..${i} labels in total, got ${a}`);
	let o = t;
	o && !/[.!?]$/.test(o) && (o += "."), o ||= ".";
	let s = C(o), c = n.map(([e, t]) => [
		"(",
		b,
		e,
		"(",
		...t.flatMap((e) => [x, e]),
		")",
		")"
	]), l = [];
	for (let e of c) l.push(...e, v);
	l.pop(), l.push(y, ...s);
	let u = /* @__PURE__ */ new Set(), d = /* @__PURE__ */ new Set(), f = 0;
	for (let e of c) {
		e.length > 1 && (u.add(f + 1), d.add(f + 1));
		for (let t = 4; t < e.length - 2; t += 2) u.add(f + t);
		f += e.length + 1;
	}
	let p = [], m = [], h = 0, g = !1;
	for (let t = 0; t < l.length; t += 1) {
		let n = l[t], r = !g;
		n === y ? g = !0 : n === v && (h += 1);
		let i = p.length;
		p.push(...e.encodeIds(n)), r && !d.has(t) && u.has(t) && m.push({
			pos: i,
			group: h
		});
	}
	let S = p.length;
	if (S > r || m.length > i) throw new _(`input exceeds bucket: seqLen ${S} > ${r} or markers > ${i}`);
	return T(m, p, S, r, i, e.padTokenId);
}
function T(e, t, n, r, i, a) {
	let o = new Int32Array(r).fill(a);
	o.set(t.slice(0, n));
	let s = new Int32Array(r);
	s.fill(1, 0, n);
	let c = new Int32Array(i), l = new Float32Array(i), u = new Int32Array(i);
	for (let [t, n] of e.entries()) {
		if (t >= i) break;
		c[t] = n.pos, l[t] = 1, u[t] = n.group;
	}
	return {
		inputIds: o,
		attentionMask: s,
		markerIndices: c,
		markerMask: l,
		markerGroups: u,
		seqLen: n
	};
}
//#endregion
//#region src/plan/layers.ts
var E = (e, t = "zero") => e === void 0 ? t : `w:${e}`, D = (e, t) => {
	if (e === void 0) throw Error(`layer tensor ${t} missing`);
	return `w:${e}`;
}, ee = (e) => Math.ceil(e / 16), O = 65535, te = (e, t) => Math.min(Math.ceil(e * t / 64), O);
function k(e, t, n, r, i, a) {
	let o = {
		M: t.rows,
		N: n,
		K: r,
		ACT: i
	};
	return r % 32 == 0 ? {
		name: e,
		kernel: "mmtile",
		constants: o,
		bind: a,
		dispatch: [Math.ceil(n / 64), "rows32"],
		alts: [{
			kernel: "mmtile8",
			dispatch: [Math.ceil(n / 32), "rows8"],
			maxRows: 16 * Math.floor(127 / Math.ceil(n / 32))
		}, {
			kernel: "mmtile16",
			dispatch: [Math.ceil(n / 32), "rows16"],
			maxRows: 32 * Math.floor(95 / Math.ceil(n / 64))
		}]
	} : {
		name: e,
		kernel: "matmul",
		constants: o,
		bind: a,
		dispatch: [ee(n), "rows16"]
	};
}
var A = 134217728, j = (e) => e.headDim % 32 == 0 && e.length % 32 == 0 && e.rows * e.heads * e.length * 4 <= A;
function M(e, t, n, r) {
	let i = {
		L: e.length,
		H: e.heads,
		D: e.headDim
	}, a = {
		H: e.heads,
		D: e.headDim,
		ROWS: e.rows,
		NM: n.nm,
		MOFF: n.moff,
		L: e.length
	}, o = e.heads * Math.ceil(n.nm / 64);
	return [
		{
			name: "relC",
			kernel: "attrel",
			constants: {
				...a,
				PART: 0
			},
			bind: [
				"qkv",
				D(r.posKey, "posKey"),
				"kinfo",
				"relidx",
				"c2p"
			],
			dispatch: [o, "rows32"]
		},
		{
			name: "relP",
			kernel: "attrel",
			constants: {
				...a,
				PART: 1
			},
			bind: [
				"qkv",
				D(r.posQuery, "posQuery"),
				"kinfo",
				"relidx",
				"p2c"
			],
			dispatch: [o, "rows32"]
		},
		{
			name: "attS",
			kernel: "attscore",
			constants: {
				...i,
				SCALE: 1 / t,
				ROWS: e.rows
			},
			bind: [
				"qkv",
				"kinfo",
				"scores"
			],
			dispatch: [e.heads * Math.ceil(e.length / 64), "rows32"]
		},
		{
			name: "attP",
			kernel: "attsoftrel",
			constants: {
				L: e.length,
				H: e.heads,
				NM: n.nm,
				MOFF: n.moff,
				INVSCALE: 1 / t
			},
			bind: [
				"mask",
				"relidx",
				"c2p",
				"p2c",
				"scores"
			],
			dispatch: [e.heads, "rows"]
		},
		{
			name: "attV",
			kernel: "attpv",
			constants: {
				...i,
				ROWS: e.rows
			},
			bind: [
				"scores",
				"qkv",
				"mask",
				"kinfo",
				"ctx"
			],
			dispatch: [e.heads * (e.headDim / 32), "rows32"]
		}
	];
}
function ne(e, t, n) {
	let r = {
		L: e.length,
		H: e.heads,
		D: e.headDim
	};
	if (j(e)) return [
		{
			name: "attS",
			kernel: "attscore",
			constants: {
				...r,
				SCALE: t,
				ROWS: e.rows
			},
			bind: [
				"qkv",
				"kinfo",
				"scores"
			],
			dispatch: [e.heads * Math.ceil(e.length / 64), "rows32"]
		},
		{
			name: "attP",
			kernel: "attsoftmax",
			constants: {
				L: e.length,
				H: e.heads,
				WINDOW: n
			},
			bind: ["mask", "scores"],
			dispatch: [e.heads, "rows"]
		},
		{
			name: "attV",
			kernel: "attpv",
			constants: {
				...r,
				ROWS: e.rows
			},
			bind: [
				"scores",
				"qkv",
				"mask",
				"kinfo",
				"ctx"
			],
			dispatch: [e.heads * (e.headDim / 32), "rows32"]
		}
	];
	let i = {
		...r,
		SCALE: t,
		WINDOW: n
	};
	return e.headDim % 4 == 0 && e.headDim <= 64 && e.length % 32 == 0 ? [{
		name: "attn",
		kernel: "mbflash",
		constants: {
			...i,
			ROWS: e.rows
		},
		bind: [
			"qkv",
			"mask",
			"ctx"
		],
		dispatch: [e.heads, "rows32"]
	}] : [{
		name: "attn",
		kernel: "mbattention",
		constants: i,
		bind: [
			"qkv",
			"mask",
			"ctx"
		],
		dispatch: [e.heads, "rows"]
	}];
}
function N(e, t, n, r) {
	return {
		name: e,
		kernel: "layernorm",
		constants: {
			N: t.hidden,
			MODE: n,
			EPS: t.eps
		},
		bind: r,
		dispatch: ["rows", 1]
	};
}
function P(e, t, n) {
	return {
		name: e,
		kernel: "add",
		constants: {
			TOTAL: t.rows * t.hidden,
			N: t.hidden,
			MODE: 0,
			L: t.length
		},
		bind: n,
		dispatch: [te(t.rows, t.hidden), 1]
	};
}
function re(e, t, n) {
	let r = e.hidden, i = e.intermediate;
	return [
		k("qkv", e, 3 * r, r, 0, [
			"x",
			D(t.qkvW, "qkvW"),
			E(t.qkvB),
			"qkv"
		]),
		...n.standard ? ne(e, n.attnScale, 0) : n.rel && j(e) ? M(e, n.attnScale, n.rel, t) : [{
			name: "attn",
			kernel: "attention",
			constants: {
				L: e.length,
				H: e.heads,
				D: e.headDim,
				SCALE: n.attnScale
			},
			bind: [
				"qkv",
				D(t.posKey, "posKey"),
				D(t.posQuery, "posQuery"),
				"relidx",
				"mask",
				"ctx"
			],
			dispatch: [e.heads, "rows"]
		}],
		k("attnOut", e, r, r, 0, [
			"ctx",
			D(t.attnOutW, "attnOutW"),
			E(t.attnOutB),
			"attnOut"
		]),
		N("lnA", e, 2, [
			"x",
			"attnOut",
			D(t.attnNormW, "attnNormW"),
			E(t.attnNormB),
			"mask",
			"tmp"
		]),
		k("ffn1", e, i, r, n.act, [
			"tmp",
			D(t.ffnInW, "ffnInW"),
			E(t.ffnInB),
			"mid"
		]),
		k("ffn2", e, r, i, 0, [
			"mid",
			D(t.ffnOutW, "ffnOutW"),
			E(t.ffnOutB),
			"ffnOut"
		]),
		N("lnF", e, 2, [
			"tmp",
			"ffnOut",
			D(t.ffnNormW, "ffnNormW"),
			E(t.ffnNormB),
			"mask",
			"x"
		])
	];
}
function ie(e, t, n) {
	let r = e.hidden, i = e.intermediate, a = [];
	return n.firstNorm && a.push(N("lnA", e, 0, [
		n.stream,
		"dummy",
		D(t.attnNormW, "attnNormW"),
		E(t.attnNormB),
		"mask",
		"normed"
	])), a.push(k("qkv", e, 3 * r, r, 0, [
		n.firstNorm ? "normed" : n.stream,
		D(t.qkvW, "qkvW"),
		E(t.qkvB),
		"qkv"
	])), n.rope && a.push({
		name: "rope",
		kernel: "rope",
		constants: {
			L: e.length,
			H: e.heads,
			D: e.headDim
		},
		bind: ["qkv", n.ropeTable ?? "cossin"],
		dispatch: ["rows", 2 * e.heads]
	}), a.push(...ne(e, n.attnScale, n.window)), a.push(k("attnOut", e, r, r, 0, [
		"ctx",
		D(t.attnOutW, "attnOutW"),
		E(t.attnOutB),
		"attnOut"
	])), a.push(P("addA", e, [n.stream, "attnOut"])), a.push(N("lnF", e, 0, [
		n.stream,
		"dummy",
		D(t.ffnNormW, "ffnNormW"),
		E(t.ffnNormB),
		"mask",
		"normed"
	])), n.ffn.kind === "geglu" ? (a.push(k("ffnIn", e, 2 * i, r, 0, [
		"normed",
		D(t.ffnInW, "ffnInW"),
		E(t.ffnInB),
		"mid"
	])), a.push({
		name: "geglu",
		kernel: "geglu",
		constants: { I: i },
		bind: ["mid", "gate"],
		dispatch: ["rows", 1]
	}), a.push(k("ffnOut", e, r, i, 0, [
		"gate",
		D(t.ffnOutW, "ffnOutW"),
		E(t.ffnOutB),
		"ffnOut"
	]))) : (a.push(k("lin1", e, n.ffn.width, r, n.ffn.act, [
		"normed",
		D(t.ffnInW, "ffnInW"),
		E(t.ffnInB),
		"mid"
	])), a.push(k("lin2", e, r, n.ffn.width, 0, [
		"mid",
		D(t.ffnOutW, "ffnOutW"),
		E(t.ffnOutB),
		"ffnOut"
	]))), a.push(P("addF", e, [n.stream, "ffnOut"])), a;
}
//#endregion
//#region src/plan/spec.ts
function F(e, t) {
	let n = e.padId;
	if (n === void 0) return -1;
	let r = e.positionOffset === n;
	for (let e = 0; e < t.length; e += 1) if (e === 0 && r) {
		if (t[0] !== n) return 0;
	} else if (t[e] === n) return e;
	return -1;
}
var I = {
	embedNormW: "embeddings.LayerNorm.weight",
	embedNormB: "embeddings.LayerNorm.bias",
	layer: (e) => ({
		qkvW: `layers.${e}.qkv.weight`,
		qkvB: `layers.${e}.qkv.bias`,
		posKey: `layers.${e}.pos_key`,
		posQuery: `layers.${e}.pos_query`,
		attnOutW: `layers.${e}.attn_out.weight`,
		attnOutB: `layers.${e}.attn_out.bias`,
		attnNormW: `layers.${e}.attn_ln.weight`,
		attnNormB: `layers.${e}.attn_ln.bias`,
		ffnInW: `layers.${e}.ffn_in.weight`,
		ffnInB: `layers.${e}.ffn_in.bias`,
		ffnOutW: `layers.${e}.ffn_out.weight`,
		ffnOutB: `layers.${e}.ffn_out.bias`,
		ffnNormW: `layers.${e}.ffn_ln.weight`,
		ffnNormB: `layers.${e}.ffn_ln.bias`
	})
}, ae = {
	w: "conv.weight",
	b: "conv.bias",
	normW: "conv.ln.weight",
	normB: "conv.ln.bias"
}, L = {
	embedNormW: "embeddings.norm.weight",
	finalNormW: "final_norm.weight",
	layer: (e) => ({
		qkvW: `layers.${e}.wqkv.weight`,
		attnOutW: `layers.${e}.attn_out.weight`,
		attnNormW: `layers.${e}.attn_norm.weight`,
		ffnInW: `layers.${e}.mlp_in.weight`,
		ffnOutW: `layers.${e}.mlp_out.weight`,
		ffnNormW: `layers.${e}.mlp_norm.weight`
	})
};
function oe(e) {
	return {
		embedNormW: "embeddings.norm.weight",
		embedNormB: e.norm ? "embeddings.norm.bias" : void 0,
		finalNormW: "final_norm.weight",
		finalNormB: e.norm ? "final_norm.bias" : void 0,
		layer: (t) => ({
			qkvW: `layers.${t}.wqkv.weight`,
			qkvB: e.attention ? `layers.${t}.wqkv.bias` : void 0,
			attnOutW: `layers.${t}.attn_out.weight`,
			attnOutB: e.attention ? `layers.${t}.attn_out.bias` : void 0,
			attnNormW: `layers.${t}.attn_norm.weight`,
			attnNormB: e.norm ? `layers.${t}.attn_norm.bias` : void 0,
			ffnInW: `layers.${t}.mlp_in.weight`,
			ffnInB: e.mlp ? `layers.${t}.mlp_in.bias` : void 0,
			ffnOutW: `layers.${t}.mlp_out.weight`,
			ffnOutB: e.mlp ? `layers.${t}.mlp_out.bias` : void 0,
			ffnNormW: `layers.${t}.mlp_norm.weight`,
			ffnNormB: e.norm ? `layers.${t}.mlp_norm.bias` : void 0
		})
	};
}
var se = {
	...I,
	layer: (e) => {
		let { posKey: t, posQuery: n, ...r } = I.layer(e);
		return r;
	}
};
function ce(e, t) {
	if (e === "bert" || e === "electra" || e === "roberta" || e === "xlm-roberta" || e === "distilbert") return se;
	if (e === "deberta-v2") return t?.conv ? {
		...I,
		conv: ae
	} : I;
	if (e === "modernbert") return t ? oe({
		attention: t.attention.bias,
		mlp: t.ffn.bias,
		norm: t.block.norm.bias
	}) : L;
	throw Error(`no tensor names for family ${e}`);
}
function le(e, t, n = 16) {
	let r = e.hiddenSize / e.heads, i = {
		eps: e.layerNormEps,
		bias: !0
	};
	return {
		spec: {
			family: "deberta-v2",
			hidden: e.hiddenSize,
			layers: e.layers,
			heads: e.heads,
			headDim: r,
			intermediate: e.intermediateSize,
			vocab: e.vocabSize ?? 0,
			embeddingSize: e.hiddenSize,
			embed: {
				positions: "none",
				positionOffset: 0,
				maxPositions: 0,
				typeVocab: 0,
				norm: i,
				maskMultiply: !0,
				project: !1
			},
			attention: {
				kind: "deberta-relative",
				bias: !0,
				scale: 1 / Math.sqrt(3 * r),
				rel: {
					buckets: e.positionBuckets,
					maxPositions: e.maxRelativePositions,
					types: ["c2p", "p2c"]
				}
			},
			block: {
				order: "post",
				firstNormIdentity: !1,
				finalNorm: !1,
				norm: i
			},
			ffn: {
				kind: "mlp",
				act: "gelu",
				bias: !0
			}
		},
		head: {
			type: "gliner2",
			hidden: Number(t.hiddenSize) || 768,
			temperature: t.temperature,
			markers: n
		}
	};
}
function ue(e) {
	let t = e.hiddenSize / e.heads, n = {
		eps: e.normEps,
		bias: !1
	};
	return {
		spec: {
			family: "modernbert",
			hidden: e.hiddenSize,
			layers: e.layers,
			heads: e.heads,
			headDim: t,
			intermediate: e.intermediate,
			vocab: e.vocab ?? 0,
			embeddingSize: e.hiddenSize,
			embed: {
				positions: "none",
				positionOffset: 0,
				maxPositions: e.maxPos ?? 0,
				typeVocab: 0,
				norm: n,
				maskMultiply: !1,
				project: !1
			},
			attention: {
				kind: "standard",
				bias: !1,
				scale: t ** -.5,
				rope: {
					thetaGlobal: e.ropeTheta,
					thetaLocal: e.ropeTheta
				},
				window: {
					half: e.localAttention,
					globalEvery: e.globalEvery
				}
			},
			block: {
				order: "pre",
				firstNormIdentity: !0,
				finalNorm: !0,
				norm: n
			},
			ffn: {
				kind: "geglu",
				act: "gelu",
				bias: !1
			}
		},
		head: {
			type: "julia",
			layers: e.headLayers,
			ffn: e.headFfn,
			options: e.options
		}
	};
}
//#endregion
//#region src/plan/build.ts
function de(e, t, n, r) {
	let i = e - t, a = n >> 1, o = Math.abs(i), s = i;
	return o > a && (s = (Math.ceil(Math.log(o / a) / Math.log((r - 1) / a) * (a - 1)) + a) * Math.sign(i)), Math.min(Math.max(s + n, 0), 2 * n - 1);
}
function fe(e, t, n) {
	let r = new Uint32Array(e * e);
	for (let i = 0; i < e; i += 1) for (let a = 0; a < e; a += 1) r[i * e + a] = de(i, a, t, n);
	return r;
}
function pe(e, t, n) {
	let r = n / 2, i = new Float32Array(e * 2 * n);
	for (let a = 0; a < e; a += 1) for (let e = 0; e < r; e += 1) {
		let r = a * t ** (-(2 * e) / n);
		i[a * 2 * n + e] = Math.cos(r), i[a * 2 * n + n + e] = Math.sin(r);
	}
	return i;
}
function me(e, t) {
	return e === "add" || e === "rope" ? [0] : [t - 1];
}
function he(e, t) {
	for (let n of e) for (let e of [...n.captureOps ?? [], ...n.ops]) {
		let r = `${n.prefix}${e.name}`;
		for (let n of e.bind) if (!n.startsWith("w:") && !t.has(n)) throw Error(`plan op ${r}: unknown buffer ${n}`);
		for (let t of me(e.kernel, e.bind.length)) {
			let n = e.bind[t];
			if (e.bind.some((e, r) => r !== t && e === n)) throw Error(`plan op ${r}: writable buffer ${n} is bound twice`);
		}
	}
}
var ge = {
	relu: 1,
	gelu: 2,
	tanh: 3,
	silu: 4
};
function R(e) {
	let t = ge[e];
	if (t === void 0) throw Error(`activation ${e} has no kernel yet`);
	return t;
}
var _e = (e) => {
	switch (e.kernel) {
		case "matmul":
		case "mmtile":
		case "mmtile16":
		case "mmtile8":
		case "layernorm":
		case "embln":
		case "pool": return e.constants.N;
		case "geglu": return e.constants.I;
		case "attention":
		case "mbattention":
		case "mbflash":
		case "attpv": return e.constants.H * e.constants.D;
		case "gather": return e.constants.D;
		case "im2col": return e.constants.N * e.constants.KS;
		default: return 0;
	}
};
function ve(e, t, n) {
	let { length: r, batch: i, f16: a } = n, o = n.tensorNames ?? ce(e.family, e), s = e.hidden, c = e.embeddingSize, l = (e) => Math.ceil(e * (a ? 2 : 4) / 4) * 4, u = r * i, d = t.type === "classify" || t.type === "token" || t.type === "embed", f = t.type === "julia" ? t.options : d ? 0 : n.markers, p = f * i, m = e.block.norm.eps, h = {
		hidden: s,
		heads: e.heads,
		headDim: e.headDim,
		intermediate: e.intermediate,
		rows: u,
		length: r,
		eps: m
	}, g = (e, t, n, r = s, i = "rows") => ({
		name: e,
		kernel: "layernorm",
		constants: {
			N: r,
			MODE: t,
			EPS: m
		},
		bind: n,
		dispatch: [i, 1]
	}), _ = (e, t, n, r, i, a, o) => ({
		name: e,
		kernel: "matmul",
		constants: {
			M: t,
			N: n,
			K: r,
			ACT: i
		},
		bind: a,
		dispatch: [Math.ceil(n / 16), o]
	}), v = Math.ceil(p / 16), y = [], b = 0, x = (...e) => e.map((e) => ({
		buffer: e,
		slot: b++
	})), S = `w:${o.embedNormW}`, C = o.embedNormB ? `w:${o.embedNormB}` : "zero";
	if (e.embed.positions === "absolute") {
		let t = e.embed.typeVocab > 0 ? "w:embeddings.type.weight" : "zero", n = e.embed.project ? "embE" : "x", i = [{
			name: "embed",
			kernel: "embln",
			constants: {
				N: c,
				L: r,
				OFFSET: e.embed.positionOffset,
				MAXPOS: e.embed.maxPositions,
				EPS: e.embed.norm.eps,
				MASKMUL: +!!e.embed.maskMultiply
			},
			bind: [
				"emb",
				"w:embeddings.position.weight",
				t,
				"typeIds",
				S,
				C,
				"mask",
				n
			],
			dispatch: ["rows", 1]
		}];
		e.embed.project && i.push(_("project", u, s, c, 0, [
			"embE",
			"w:embeddings.project.weight",
			"w:embeddings.project.bias",
			"x"
		], "rows16")), y.push({
			name: "embed",
			prefix: "",
			ops: i,
			capture: x("x")
		});
	} else e.embed.maskMultiply ? y.push({
		name: "embed",
		prefix: "",
		ops: [g("embed", 1, [
			"emb",
			"dummy",
			S,
			C,
			"mask",
			"x"
		])],
		captureOps: [g("embedPlain", 0, [
			"emb",
			"dummy",
			S,
			C,
			"mask",
			"tmp"
		])],
		capture: x("tmp", "x")
	}) : y.push({
		name: "embed",
		prefix: "",
		ops: [g("embed", 0, [
			"emb",
			"dummy",
			S,
			C,
			"mask",
			"x"
		])],
		capture: x("x")
	});
	let w = e.conv;
	if (w) {
		if (!o.conv) throw Error("conv without tensor names");
		y.push({
			name: "convIn",
			prefix: "ConvIn.",
			ops: [{
				name: "im2col",
				kernel: "im2col",
				constants: {
					N: s,
					L: r,
					KS: w.kernel
				},
				bind: [
					"x",
					"mask",
					"cols"
				],
				dispatch: ["rows", 1]
			}, _("conv", u, s, w.kernel * s, R(w.act), [
				"cols",
				`w:${o.conv.w}`,
				`w:${o.conv.b}`,
				"convOut"
			], "rows16")]
		});
	}
	let T = e.attention.kind === "deberta-relative" ? e.attention.rel : void 0, E;
	if (e.attention.kind === "deberta-relative") {
		if (!T) throw Error("deberta-relative attention needs attention.rel");
		let t = Math.sqrt((1 + T.types.length) * (s / e.heads));
		E = () => ({
			scale: t,
			window: 0
		});
	} else {
		let t = e.attention.window, n = (s / e.heads) ** -.5;
		E = (e) => ({
			scale: n,
			window: !t || e % t.globalEvery === 0 ? 0 : t.half
		});
	}
	let D = e.attention.rope, ee = !!D && D.thetaGlobal !== D.thetaLocal;
	for (let t = 0; t < e.layers; t += 1) {
		let n = o.layer(t), i = E(t), a;
		if (e.block.order === "post") a = re(h, n, {
			act: R(e.ffn.act),
			attnScale: i.scale,
			standard: e.attention.kind === "standard",
			rel: T && {
				moff: de(0, r - 1, T.buckets, T.maxPositions),
				nm: de(r - 1, 0, T.buckets, T.maxPositions) - de(0, r - 1, T.buckets, T.maxPositions) + 1
			}
		});
		else {
			let r = e.block.firstNormIdentity && t === 0, o = e.ffn.kind === "geglu" ? { kind: "geglu" } : {
				kind: "mlp",
				width: e.intermediate,
				act: R(e.ffn.act)
			};
			a = ie(h, r ? {
				...n,
				attnNormW: void 0
			} : n, {
				stream: "x",
				firstNorm: !r,
				attnScale: i.scale,
				window: i.window,
				rope: !!D,
				ffn: o,
				ropeTable: ee ? i.window === 0 ? "cossinG" : "cossinL" : "cossin"
			});
		}
		let c = t === 0 && w;
		y.push({
			name: `layer${t}`,
			prefix: `L${t}.`,
			ops: a,
			skippable: !0,
			...c ? {} : { capture: x("x") }
		}), c && o.conv && y.push({
			name: "conv",
			prefix: "Conv.",
			ops: [{
				name: "add",
				kernel: "add",
				constants: {
					TOTAL: u * s,
					N: s,
					MODE: 0,
					L: r
				},
				bind: ["convOut", "x"],
				dispatch: [te(u, s), 1]
			}, g("lnConv", 1, [
				"convOut",
				"dummy",
				`w:${o.conv.normW}`,
				`w:${o.conv.normB}`,
				"mask",
				"x"
			])],
			capture: x("x")
		});
	}
	let O = "x";
	if (e.block.finalNorm) {
		if (!o.finalNormW) throw Error("finalNorm without a tensor name");
		y.push({
			name: "final",
			prefix: "",
			ops: [g("final", 0, [
				"x",
				"dummy",
				`w:${o.finalNormW}`,
				o.finalNormB ? `w:${o.finalNormB}` : "zero",
				"mask",
				"tmp"
			])],
			capture: x("tmp")
		}), O = "tmp";
	}
	let k, A = [], j = p, M = 1, ne = "logits", N = !1;
	if (t.type === "classify" || t.type === "token" || t.type === "embed") {
		let e = t.type === "token", n = e ? u : i, a = e ? "rows16" : Math.ceil(i / 16), o = O, c = s, l = [];
		if (!e) {
			let e = t.pool === "cls" ? "first" : t.pool;
			e === "first" ? (N = !0, l.push({
				name: "gather",
				kernel: "gather",
				constants: {
					K: 1,
					D: s,
					L: r
				},
				bind: [
					"first",
					O,
					"states"
				],
				dispatch: [1, 1]
			})) : l.push({
				name: "pool",
				kernel: "pool",
				constants: {
					L: r,
					N: s,
					MODE: e === "mean" ? 0 : 1
				},
				bind: [
					O,
					"mask",
					"states"
				],
				dispatch: [Math.ceil(s / 64), i]
			}), A.push({
				id: "states",
				elements: i * s,
				final: !1
			}), o = "states";
		}
		if (t.steps.forEach((r, s) => {
			if (r.op === "norm") {
				let a = s === t.steps.length - 1, u = a ? "out" : `hd${s}`;
				l.push({
					name: `step${s}`,
					kernel: "layernorm",
					constants: {
						N: c,
						MODE: 0,
						EPS: r.eps
					},
					bind: [
						o,
						"dummy",
						`w:${r.name}.weight`,
						r.bias ? `w:${r.name}.bias` : "zero",
						"mask",
						u
					],
					dispatch: [e ? "rows" : i, 1]
				}), A.push({
					id: u,
					elements: n * c,
					final: a
				}), o = u;
				return;
			}
			if (r.in !== c) throw Error(`head step ${r.name}: input ${r.in}, previous width ${c}`);
			let u = s === t.steps.length - 1, d = u ? "out" : `hd${s}`;
			l.push(_(`step${s}`, n, r.out, r.in, r.act === "none" ? 0 : R(r.act), [
				o,
				`w:${r.name}.weight`,
				r.bias ? `w:${r.name}.bias` : "zero",
				d
			], a)), A.push({
				id: d,
				elements: n * r.out,
				final: u
			}), o = d, c = r.out;
		}), !t.steps.length) {
			if (e) throw Error("a token head needs at least one dense step");
			A.find((e) => e.id === "states").final = !0;
		}
		ne = o, j = n, M = c, y.push({
			name: "head",
			prefix: "head.",
			ops: l
		});
	} else if (t.type === "gliner2") {
		let e = t.hidden;
		y.push({
			name: "head",
			prefix: "head.",
			ops: [
				{
					name: "gather",
					kernel: "gather",
					constants: {
						K: f,
						D: s,
						L: r
					},
					bind: [
						"packed",
						O,
						"states"
					],
					dispatch: [1, 1]
				},
				_("fc1", p, e, s, 1, [
					"states",
					"w:head.fc1.weight",
					"w:head.fc1.bias",
					"h1"
				], v),
				_("fc2", p, 1, e, 0, [
					"h1",
					"w:head.fc2.weight",
					"w:head.fc2.bias",
					"raw"
				], v),
				{
					name: "maskl",
					kernel: "masklogits",
					constants: {
						K: f,
						TEMP: t.temperature
					},
					bind: [
						"raw",
						"packed",
						"logits"
					],
					dispatch: [1, 1]
				}
			]
		});
	} else {
		if (O !== "tmp") throw Error("julia head expects a final norm");
		y.push({
			name: "typed",
			prefix: "",
			ops: [{
				name: "type",
				kernel: "add",
				constants: {
					TOTAL: u * s,
					N: s,
					MODE: 1,
					L: r
				},
				bind: ["tmp", "typeRow"],
				dispatch: [te(u, s), 1]
			}],
			capture: x("tmp")
		}), k = {
			table: "w:type_emb.weight",
			dst: "typeRow",
			rowBytes: l(s)
		};
		for (let n = 0; n < t.layers; n += 1) {
			let r = `head.${n}`, i = {
				qkvW: `${r}.in_proj.weight`,
				qkvB: `${r}.in_proj.bias`,
				attnOutW: `${r}.out_proj.weight`,
				attnOutB: `${r}.out_proj.bias`,
				attnNormW: `${r}.norm1.weight`,
				attnNormB: `${r}.norm1.bias`,
				ffnNormW: `${r}.norm2.weight`,
				ffnNormB: `${r}.norm2.bias`,
				ffnInW: `${r}.linear1.weight`,
				ffnInB: `${r}.linear1.bias`,
				ffnOutW: `${r}.linear2.weight`,
				ffnOutB: `${r}.linear2.bias`
			};
			y.push({
				name: `head${n}`,
				prefix: `head${n}.`,
				ops: ie(h, i, {
					stream: "tmp",
					firstNorm: !0,
					attnScale: (s / e.heads) ** -.5,
					window: 0,
					rope: !1,
					ffn: {
						kind: "mlp",
						width: t.ffn,
						act: 1
					}
				}),
				capture: x("tmp")
			});
		}
		y.push({
			name: "scorer",
			prefix: "scorer.",
			ops: [
				{
					name: "gather",
					kernel: "gather",
					constants: {
						K: f,
						D: s,
						L: r
					},
					bind: [
						"packed",
						"tmp",
						"states"
					],
					dispatch: [1, 1]
				},
				g("ln", 0, [
					"states",
					"dummy",
					"w:scorer.norm.weight",
					"w:scorer.norm.bias",
					"mask",
					"kNormed"
				], s, p),
				_("fc1", p, s, s, 2, [
					"kNormed",
					"w:scorer.fc1.weight",
					"w:scorer.fc1.bias",
					"h1"
				], v),
				_("fc2", p, 1, s, 0, [
					"h1",
					"w:scorer.fc2.weight",
					"w:scorer.fc2.bias",
					"raw"
				], v),
				{
					name: "maskl",
					kernel: "masklogits",
					constants: {
						K: f,
						TEMP: 1
					},
					bind: [
						"raw",
						"packed",
						"logits"
					],
					dispatch: [1, 1]
				}
			]
		});
	}
	let P = y.flatMap((e) => [...e.captureOps ?? [], ...e.ops]), F = /* @__PURE__ */ new Map();
	for (let e of P) {
		let t = e.bind[me(e.kernel, e.bind.length)[0]], n = _e(e);
		n > 0 && F.set(t, Math.max(F.get(t) ?? 0, n));
	}
	let I = /* @__PURE__ */ new Set([
		"states",
		"kNormed",
		"h1",
		"raw"
	]), ae = [], L = (e, t, n = "rw", r) => {
		ae.push(r ? {
			id: e,
			bytes: t,
			usage: n,
			init: r
		} : {
			id: e,
			bytes: t,
			usage: n
		});
	}, oe = (e, t = "rw") => {
		let n = F.get(e);
		n !== void 0 && L(e, l((I.has(e) ? p : u) * n), t);
	};
	L("emb", l(u * c)), e.embed.project && L("embE", l(u * c)), e.embed.positions === "absolute" && L("typeIds", u * 4);
	for (let e of ["x", "tmp"]) F.has(e) && L(e, l(u * F.get(e)), "rwSrc");
	w && (L("cols", l(u * w.kernel * s)), L("convOut", l(u * s)));
	for (let e of [
		"normed",
		"qkv",
		"ctx",
		"attnOut",
		"mid",
		"gate",
		"ffnOut",
		"states",
		"kNormed",
		"h1",
		"raw"
	]) d && I.has(e) || oe(e);
	L("mask", u * 4), P.some((e) => e.bind.includes("kinfo")) && L("kinfo", 8 * i);
	let se = P.filter((e) => e.kernel === "attscore");
	for (let e of ["c2p", "p2c"]) {
		let t = P.filter((t) => t.kernel === "attrel" && t.bind[4] === e);
		t.length && L(e, Math.max(...t.map((e) => e.constants.ROWS * e.constants.H * e.constants.NM * 4)));
	}
	if (se.length && L("scores", Math.max(...se.map((e) => e.constants.ROWS * e.constants.H * e.constants.L * 4))), P.some((e) => e.bind.includes("dummy")) && L("dummy", 4, "storage"), d) {
		for (let e of A) L(e.id, l(e.elements), e.final ? "logits" : "rw");
		N && L("first", 3 * i * 4, "rw", new Uint32Array(3 * i));
	} else L("packed", 3 * p * 4), L("logits", p * 4, "logits");
	if (T) {
		let e = fe(r, T.buckets, T.maxPositions);
		L("relidx", e.byteLength, "rw", e);
	}
	if (D && !ee) {
		let t = pe(r, D.thetaGlobal, e.headDim);
		L("cossin", t.byteLength, "rw", t);
	} else if (D) {
		let t = pe(r, D.thetaGlobal, e.headDim), n = pe(r, D.thetaLocal, e.headDim);
		L("cossinG", t.byteLength, "rw", t), L("cossinL", n.byteLength, "rw", n);
	}
	k && L("typeRow", l(i * s));
	let le = P.filter((e) => e.bind.includes("zero"));
	if (le.length) {
		let e = Math.max(...le.map((e) => e.constants.N));
		L("zero", l(e), "rw", new Uint8Array(l(e)));
	}
	return he(y, new Set(ae.map((e) => e.id))), {
		f16: a,
		length: r,
		batch: i,
		markers: f,
		embeddingSize: c,
		buffers: ae,
		segments: y,
		inputs: d ? {
			embeddings: "emb",
			mask: "mask",
			...e.embed.positions === "absolute" ? { typeIds: "typeIds" } : {}
		} : {
			embeddings: "emb",
			mask: "mask",
			markers: "packed"
		},
		rowSelect: k,
		output: d ? {
			buffer: ne,
			bytes: l(j * M),
			dtype: "storage",
			rows: j,
			cols: M
		} : {
			buffer: "logits",
			bytes: p * 4,
			dtype: "f32",
			rows: p,
			cols: 1
		},
		captureSlots: b,
		captureSlotBytes: r * s * 4
	};
}
//#endregion
//#region src/cache.ts
var z = [
	1,
	4,
	8,
	16
], B = z[z.length - 1];
function ye(e) {
	for (let t of z) if (e <= t) return t;
	throw Error(`batch of ${e} exceeds the maximum of ${B}`);
}
var V = class {
	map = /* @__PURE__ */ new Map();
	maxEntries;
	constructor(e) {
		this.maxEntries = e;
	}
	get size() {
		return this.map.size;
	}
	get enabled() {
		return this.maxEntries > 0;
	}
	get(e) {
		let t = this.map.get(e);
		if (t !== void 0) return this.map.delete(e), this.map.set(e, t), t;
	}
	set(e, t) {
		if (!(this.maxEntries <= 0)) {
			if (this.map.has(e)) this.map.delete(e);
			else if (this.map.size >= this.maxEntries) {
				let e = this.map.keys().next().value;
				e !== void 0 && this.map.delete(e);
			}
			this.map.set(e, t);
		}
	}
	clear() {
		this.map.clear();
	}
};
function H(e) {
	return `${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markerIndices.join(",")}|${e.markerMask.join(",")}|${e.markerGroups.join(",")}`;
}
function U(e) {
	return `${e.qtype}|${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markers.join(",")}`;
}
function be(e) {
	return Math.ceil(e / 64) * 64;
}
function xe(e, t, n) {
	let r = e.map((e, t) => t).sort((t, r) => n(e[t]) - n(e[r])), i = Array(e.length);
	return r.forEach((e, t) => {
		i[e] = t;
	}), {
		unique: r.map((t) => e[t]),
		slot: t.map((e) => i[e])
	};
}
function W(e, t) {
	let n = /* @__PURE__ */ new Map(), r = [], i = [];
	for (let a of e) {
		let e = t(a), o = n.get(e);
		o === void 0 && (o = r.length, n.set(e, o), r.push(a)), i.push(o);
	}
	return {
		unique: r,
		slot: i
	};
}
//#endregion
//#region src/plan/check.ts
var Se = {
	add: () => 0,
	attention: (e) => 4 * e + 256,
	attpv: () => 8200,
	attrel: () => 12292,
	attscore: () => 12292,
	attsoftmax: () => 256,
	attsoftrel: () => 256,
	embln: () => 256,
	gather: () => 0,
	geglu: () => 0,
	im2col: () => 0,
	layernorm: () => 256,
	masklogits: () => 0,
	matmul: () => 2048,
	mbattention: (e) => 4 * e + 256,
	mbflash: () => 16384,
	mmtile: () => 12288,
	mmtile16: () => 6144,
	mmtile8: () => 5120,
	pool: () => 0,
	rope: () => 0
}, Ce = (e, t) => e === "rows" ? t : e === "rows8" ? Math.ceil(t / 8) : e === "rows16" ? Math.ceil(t / 16) : e === "rows32" ? Math.ceil(t / 32) : e;
function we(t, n, r, i = {}) {
	let a = n.maxStorageBufferBindingSize ?? e.maxStorageBufferBindingSize, o = n.maxBufferSize ?? e.maxBufferSize, s = n.maxComputeWorkgroupsPerDimension ?? e.maxComputeWorkgroupsPerDimension, c = n.maxComputeWorkgroupStorageSize ?? e.maxComputeWorkgroupStorageSize, l = n.maxStorageBuffersPerShaderStage ?? e.maxStorageBuffersPerShaderStage, u = `bucket L${t.length} B${t.batch}`, d = [], f = (e, t, n, r) => {
		t > r && d.push(`${u}: ${e} needs ${t}, ${n} is ${r}`);
	}, p = t.segments.flatMap((e) => [...e.captureOps ?? [], ...e.ops].map((t) => ({
		op: t,
		at: `${e.prefix}${t.name}`
	})));
	i.batchOnly || (!Number.isInteger(t.length) || t.length <= 0 || t.length % 4 != 0) && d.push(`${u}: the length must be a positive multiple of 4 (the attention kernels read keys in groups of 4)`), (!Number.isInteger(t.batch) || t.batch <= 0) && d.push(`${u}: the batch must be a positive integer`);
	let m = t.f16 ? 2 : 4, h = t.length * t.batch * t.embeddingSize * m;
	h % 4 != 0 && d.push(`${u}: the word row upload of ${h} B is not a multiple of 4 (writeBuffer needs it)`);
	for (let e of t.buffers) f(`buffer ${e.id}`, e.bytes, "maxStorageBufferBindingSize", a), f(`buffer ${e.id}`, e.bytes, "maxBufferSize", o);
	f("the output staging buffer", t.output.bytes, "maxBufferSize", o);
	let g = t.length * t.batch;
	for (let { op: e, at: n } of p) {
		for (let t of e.dispatch) f(`dispatch ${n}`, Ce(t, g), "maxComputeWorkgroupsPerDimension", s);
		for (let t of e.alts ?? []) for (let e of t.dispatch) f(`dispatch ${n} (${t.kernel})`, Ce(e, g), "maxComputeWorkgroupsPerDimension", s);
		i.batchOnly || (f(`op ${n}`, e.bind.length, "maxStorageBuffersPerShaderStage", l), f(`workgroup memory of ${n} (${e.kernel})`, Se[e.kernel](t.length), "maxComputeWorkgroupStorageSize", c), e.kernel === "attention" && e.constants.D % 4 != 0 && d.push(`${u}: relative attention with head width ${e.constants.D} is not supported (needs a multiple of 4)`));
	}
	if (r && !i.batchOnly) {
		let e = /* @__PURE__ */ new Set();
		for (let { op: t } of p) for (let n of t.bind) {
			if (!n.startsWith("w:") || e.has(n)) continue;
			e.add(n);
			let t = r(n.slice(2));
			t !== void 0 && f(`weight ${n.slice(2)}`, t, "maxStorageBufferBindingSize", a);
		}
	}
	return [...new Set(d)];
}
function Te(e, t, n) {
	let r = we(e, t, n);
	if (r.length) throw Error(r.join("; "));
}
function Ee(e, t, n) {
	for (let r of [...z].reverse()) {
		if (r === 1 || r > n) continue;
		let i = e(r);
		if (we(i, t, void 0, { batchOnly: !0 }).length === 0) return i;
	}
}
//#endregion
//#region src/half.ts
function De(e) {
	let t = e & 32768 ? -1 : 1, n = e >> 10 & 31, r = e & 1023;
	return n === 0 ? t * r * 2 ** -24 : n === 31 ? r === 0 ? t * Infinity : NaN : t * (1 + r / 1024) * 2 ** (n - 15);
}
function Oe(e, t = !0) {
	let n = new Float32Array(e.length), r = globalThis.Float16Array;
	if (t && r) {
		let t = new r(e.buffer, e.byteOffset, e.length);
		for (let r = 0; r < e.length; r += 1) n[r] = t[r];
		return n;
	}
	for (let t = 0; t < e.length; t += 1) n[t] = De(e[t]);
	return n;
}
//#endregion
//#region src/kernels/add.wgsl?raw
var ke = "{{ENABLE}}// Elementwise residual accumulation. MODE 0: dst[i] += src[i].\n// MODE 1: dst[i] += src[b * N + i % N] where b = row / L is the batch\n// sequence (per-row type embeddings; at B = 1 this is src[i % N]).\n// Workgroups of 64 threads stride over the elements (grid-stride loop), so the dispatch can be\n// capped at 65535 workgroups in x whatever TOTAL is.\n\noverride TOTAL: u32 = 1u;\noverride N: u32 = 384u;\noverride MODE: u32 = 0u;\noverride L: u32 = 128u;\n\n@group(0) @binding(0) var<storage, read_write> dst: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> src: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n  @builtin(num_workgroups) nwg: vec3<u32>,\n) {\n  for (var i = gid.x; i < TOTAL; i += 64u * nwg.x) {\n    let s = select(src[i], src[(i / (N * L)) * N + i % N], MODE == 1u);\n    dst[i] = {{F}}(f32(dst[i]) + f32(s));\n  }\n}\n", Ae = "{{ENABLE}}// DeBERTa-v2 relative attention, one workgroup per (head, query row).\n// scores[i,j] = (q_i.k_j + c2p[i,j] + p2c[i,j]) / SCALE over key positions,\n// masked softmax (masked pairs -> -1e30, fully masked rows give a uniform\n// distribution like the fp32 reference, never NaN), then . v.\n//   c2p[i,j] = q_i . pos_key[idx[i,j]]   (content -> position)\n//   p2c[i,j] = k_j . pos_query[idx[i,j]] (position -> content)\n// pos_key/pos_query are stored [2*SPAN, H*D] row-major (m-major).\n// The q row is hoisted into registers once (it would otherwise be re-read\n// for every key); fully masked query rows write zeros and skip the loop.\n// Batch (K16): B sequences are packed as B*L global rows; row r = b*L + i\n// belongs to sequence b, whose keys/values live at global rows b*L + j\n// while relative positions and the sliding table stay local (i, j).\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 13.856406; // sqrt(64 * 3)\n\n@group(0) @binding(0) var<storage, read> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> posKey: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> posQuery: array<{{F}}>;\n@group(0) @binding(3) var<storage, read> relidx: array<u32>;\n@group(0) @binding(4) var<storage, read> mask: array<f32>;\n@group(0) @binding(5) var<storage, read_write> ctx: array<{{F}}>;\n\n// One score per bucket position; the array follows the L override, so a\n// L1024 pipeline takes 4 KiB (well under the 16 KiB minimum limit).\nvar<workgroup> scores: array<f32, L>;\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let b = i / L;\n  let il = i - b * L;\n  let kbase = b * L;\n  let hd = H * D;\n  let qb = i * 3u * hd + h * D;\n  var qreg: array<f32, 64>;\n  for (var d = 0u; d < D; d += 1u) {\n    qreg[d] = f32(qkv[qb + d]);\n  }\n  // Masked query rows contribute nothing downstream (their ctx feeds only\n  // their own row), so they can be zeroed and skipped.\n  if (mask[i] <= 0.5) {\n    for (var d = lid.x; d < D; d += 64u) {\n      ctx[i * hd + h * D + d] = {{F}}(0.0);\n    }\n    return;\n  }\n  for (var j = lid.x; j < L; j += 64u) {\n    // Masked keys land at -1e30 regardless of the dot products.\n    // The sentinel sits below any real score; the softmax max also starts at -1e30.\n    if (mask[kbase + j] <= 0.5) {\n      scores[j] = -1e30;\n      continue;\n    }\n    var s = 0.0;\n    let p = relidx[il * L + j] * hd;\n    let kb = (kbase + j) * 3u * hd + (H + h) * D;\n    let pk = p + h * D;\n    let pq = p + h * D;\n    for (var d = 0u; d < D; d += 4u) {\n      let q0 = qreg[d];\n      let q1 = qreg[d + 1u];\n      let q2 = qreg[d + 2u];\n      let q3 = qreg[d + 3u];\n      let k0 = f32(qkv[kb + d]);\n      let k1 = f32(qkv[kb + d + 1u]);\n      let k2 = f32(qkv[kb + d + 2u]);\n      let k3 = f32(qkv[kb + d + 3u]);\n      s += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;\n      s += q0 * f32(posKey[pk + d]) + q1 * f32(posKey[pk + d + 1u])\n        + q2 * f32(posKey[pk + d + 2u]) + q3 * f32(posKey[pk + d + 3u]);\n      s += k0 * f32(posQuery[pq + d]) + k1 * f32(posQuery[pq + d + 1u])\n        + k2 * f32(posQuery[pq + d + 2u]) + k3 * f32(posQuery[pq + d + 3u]);\n    }\n    scores[j] = s / SCALE;\n  }\n  workgroupBarrier();\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) { m = max(m, scores[j]); }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    let e = exp(scores[j] - mx);\n    scores[j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let invTotal = 1.0 / red[0];\n  workgroupBarrier();\n  for (var d = lid.x; d < D; d += 64u) {\n    let vb = (2u * H + h) * D + d;\n    // The softmax normalization folds into the epilogue (ctx = acc / total):\n    // exp() underflows to exactly 0 for masked keys, so the sv != 0 branch\n    // still keeps stale rows out, and four independent accumulators shorten\n    // the serial FMA dependency.\n    var acc0 = 0.0;\n    var acc1 = 0.0;\n    var acc2 = 0.0;\n    var acc3 = 0.0;\n    for (var j = 0u; j + 3u < L; j += 4u) {\n      let s0 = scores[j];\n      let s1 = scores[j + 1u];\n      let s2 = scores[j + 2u];\n      let s3 = scores[j + 3u];\n      if (s0 != 0.0) { acc0 += s0 * f32(qkv[(kbase + j + 0u) * 3u * hd + vb]); }\n      if (s1 != 0.0) { acc1 += s1 * f32(qkv[(kbase + j + 1u) * 3u * hd + vb]); }\n      if (s2 != 0.0) { acc2 += s2 * f32(qkv[(kbase + j + 2u) * 3u * hd + vb]); }\n      if (s3 != 0.0) { acc3 += s3 * f32(qkv[(kbase + j + 3u) * 3u * hd + vb]); }\n    }\n    var acc = (acc0 + acc1 + acc2 + acc3) * invTotal;\n    ctx[i * hd + h * D + d] = {{F}}(acc);\n  }\n}\n", je = "{{ENABLE}}// Attention context as a matmul (K27): ctx[i][h * D + d] = sum_j P[i][h][j] * v_j[h * D + d],\n// P the f32 probabilities of attsoftmax.wgsl. Register-blocked: 8x8 threads, 4x4 per thread,\n// tile 32 query rows x 32 head dims, keys in steps of 32 through workgroup memory. Value rows\n// of masked keys load as zero, so stale padding rows never enter the sum (their P is 0 too).\n// Workgroup x = h * (D / 32) + dim tile, y = query block of 32 rows. Needs D % 32 == 0 and\n// L % 32 == 0. Every output sums j in order in f32.\n\noverride L: u32 = 128u;\noverride H: u32 = 12u;\noverride D: u32 = 64u;\noverride ROWS: u32 = 128u;   // B * L\n\n@group(0) @binding(0) var<storage, read> probs: array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read> qkv: array<vec4<{{F}}>>;\n@group(0) @binding(2) var<storage, read> mask: array<f32>;\n@group(0) @binding(3) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence\n@group(0) @binding(4) var<storage, read_write> ctx: array<{{F}}>;\n\nvar<workgroup> sa: array<vec4<f32>, 256>;   // [row][key/4], 32 x 8\nvar<workgroup> sb: array<vec4<{{F}}>, 256>; // [key][d/4], 32 x 8\nvar<workgroup> keyFirst: u32;\nvar<workgroup> keyLast: u32;\n\n@compute @workgroup_size(8, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let nd = D / 32u;\n  let h = wid.x / nd;\n  let dt = wid.x % nd;\n  let tid = lid.y * 8u + lid.x;\n  let row0 = wid.y * 32u;\n  let kbase = (row0 / L) * L;\n  let tileRow = lid.y * 4u;\n  let L4 = L / 4u;\n  let D4 = D / 4u;\n  let row4 = 3u * H * D4;\n  var acc: array<vec4<f32>, 4>;\n  // 32 keys outside the valid keys of the sequence add exactly zero (P is 0 there): skip them.\n  if (tid == 0u) {\n    keyFirst = kinfo[2u * (row0 / L)];\n  }\n  let first = workgroupUniformLoad(&keyFirst);\n  if (tid == 0u) {\n    keyLast = kinfo[2u * (row0 / L) + 1u];\n  }\n  let last = workgroupUniformLoad(&keyLast);\n  for (var t4 = 0u; t4 < L4; t4 += 8u) {\n    if (first >= L || t4 * 4u > last || t4 * 4u + 31u < first) { continue; }\n    for (var run = tid; run < 256u; run += 64u) {\n      let r = row0 + run / 8u;\n      sa[run] = select(vec4<f32>(0.0), probs[(r * H + h) * L4 + t4 + run % 8u], r < ROWS);\n      let key = kbase + t4 * 4u + run / 8u;\n      sb[run] = select(vec4<{{F}}>(0.0), qkv[key * row4 + (2u * H + h) * D4 + dt * 8u + run % 8u],\n        mask[key] > 0.5);\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sb[(k4 * 4u) * 8u + lid.x]);\n      let b1 = vec4<f32>(sb[(k4 * 4u + 1u) * 8u + lid.x]);\n      let b2 = vec4<f32>(sb[(k4 * 4u + 2u) * 8u + lid.x]);\n      let b3 = vec4<f32>(sb[(k4 * 4u + 3u) * 8u + lid.x]);\n      for (var i = 0u; i < 4u; i += 1u) {\n        let av = sa[(tileRow + i) * 8u + k4];\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let col = h * D + dt * 32u + lid.x * 4u;\n  for (var i = 0u; i < 4u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= ROWS) { continue; }\n    for (var c = 0u; c < 4u; c += 1u) {\n      ctx[row * H * D + col + c] = {{F}}(acc[i][c]);\n    }\n  }\n}\n", Me = "{{ENABLE}}// DeBERTa relative terms as a matmul (K27): OUT[r][h][m] = x_r . T[MOFF + m] in f32, x the\n// q part (PART 0, content -> position: c2p with T = pos_key) or the k part (PART 1, position\n// -> content: p2c with T = pos_query) of qkv row r, for the NM position rows the bucket can\n// reach (MOFF .. MOFF + NM - 1). T is [2 * SPAN, H * D] row-major. Register-blocked like\n// attscore.wgsl: 16x8 threads, 4x4 per thread, tile 32 rows x 64 positions, the head\n// dimension in steps of 32. Workgroup x = h * ceil(NM / 64) + position tile, y = block of 32\n// rows. Needs D % 32 == 0. Position tiles no valid pair of the sequence reaches write nothing.\n\noverride H: u32 = 12u;\noverride D: u32 = 64u;\noverride ROWS: u32 = 128u;   // B * L\noverride NM: u32 = 256u;\noverride MOFF: u32 = 0u;\noverride PART: u32 = 0u;\noverride L: u32 = 128u;      // bucket length (relidx is L x L)\n\n@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> table: array<vec4<{{F}}>>;\n@group(0) @binding(2) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence\n@group(0) @binding(3) var<storage, read> relidx: array<u32>;\n@group(0) @binding(4) var<storage, read_write> rel: array<f32>;\n\nvar<workgroup> sa: array<vec4<{{F}}>, 256>; // [row][d/4], 32 rows x 8\nvar<workgroup> sw: array<vec4<{{F}}>, 512>; // [d][m/4], 32 x 16\nvar<workgroup> tileLive: u32;\n\n@compute @workgroup_size(16, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let nmt = (NM + 63u) / 64u;\n  let h = wid.x / nmt;\n  let m0 = (wid.x % nmt) * 64u;\n  let tid = lid.y * 16u + lid.x;\n  let row0 = wid.y * 32u;\n  let tileRow = lid.y * 4u;\n  let D4 = D / 4u;\n  let hd4 = H * D4;\n  let row4 = 3u * hd4;\n  // Valid keys of this sequence span first .. last; their pairs reach the positions from\n  // relidx[first][last] to relidx[last][first] (the bucket is monotone in i - j). A position\n  // tile outside that range is never read (attsoftrel reads valid pairs only): skip it.\n  if (tid == 0u) {\n    let b = row0 / L;\n    let first = kinfo[2u * b];\n    let last = kinfo[2u * b + 1u];\n    var live = 0u;\n    if (first < L) {\n      let lo = relidx[first * L + last] - MOFF;\n      let hi = relidx[last * L + first] - MOFF;\n      live = select(0u, 1u, m0 <= hi && m0 + 63u >= lo);\n    }\n    tileLive = live;\n  }\n  if (workgroupUniformLoad(&tileLive) == 0u) { return; }\n  var acc: array<vec4<f32>, 4>;\n  for (var t4 = 0u; t4 < D4; t4 += 8u) {\n    for (var run = tid; run < 256u; run += 128u) {\n      let r = row0 + run / 8u;\n      sa[run] = select(vec4<{{F}}>(0.0), qkv[r * row4 + (PART * H + h) * D4 + t4 + run % 8u], r < ROWS);\n    }\n    let ng = tid / 8u;\n    let kv = tid % 8u;\n    var m: array<vec4<{{F}}>, 4>;\n    for (var q = 0u; q < 4u; q += 1u) {\n      let mm = m0 + ng * 4u + q;\n      m[q] = select(vec4<{{F}}>(0.0), table[(MOFF + mm) * hd4 + h * D4 + t4 + kv], mm < NM);\n    }\n    for (var e = 0u; e < 4u; e += 1u) {\n      sw[(kv * 4u + e) * 16u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sw[(k4 * 4u) * 16u + lid.x]);\n      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 16u + lid.x]);\n      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 16u + lid.x]);\n      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 16u + lid.x]);\n      for (var i = 0u; i < 4u; i += 1u) {\n        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let mc = m0 + lid.x * 4u;\n  for (var i = 0u; i < 4u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= ROWS) { continue; }\n    for (var c = 0u; c < 4u; c += 1u) {\n      if (mc + c < NM) {\n        rel[(row * H + h) * NM + mc + c] = acc[i][c];\n      }\n    }\n  }\n}\n", Ne = "{{ENABLE}}// Attention scores as a matmul (K27): S[i][h][j] = SCALE * (q_i . k_j) in f32 for every\n// head h, query row i and key j of the same sequence. Register-blocked like mmtile.wgsl:\n// 16x8 threads, 4x4 per thread, tile 32 query rows x 64 keys, k (the head dimension) in\n// steps of 32 through workgroup memory. Workgroup x = h * ceil(L / 64) + key tile, y = query\n// block of 32 rows. Layout of qkv: row i holds [q | k | v] of H * D each, B sequences packed\n// as B * L rows. Needs D % 32 == 0 and L % 32 == 0. Masking happens in attsoftmax.wgsl; a key\n// tile with no valid key writes nothing (its scores are never read).\n\noverride L: u32 = 128u;\noverride H: u32 = 12u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 0.125;\noverride ROWS: u32 = 128u;   // B * L\n\n@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence\n@group(0) @binding(2) var<storage, read_write> scores: array<f32>;\n\nvar<workgroup> sa: array<vec4<{{F}}>, 256>; // [row][d/4], 32 rows x 8\nvar<workgroup> sw: array<vec4<{{F}}>, 512>; // [d][key/4], 32 x 16\nvar<workgroup> tileLive: u32;\n\n@compute @workgroup_size(16, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let nkt = (L + 63u) / 64u;\n  let h = wid.x / nkt;\n  let j0 = (wid.x % nkt) * 64u;\n  let tid = lid.y * 16u + lid.x;\n  let row0 = wid.y * 32u;\n  let kbase = (row0 / L) * L;\n  let tileRow = lid.y * 4u;\n  let D4 = D / 4u;\n  let row4 = 3u * H * D4;\n  // A key tile outside the valid keys of the sequence is never read (attsoftmax skips masked\n  // keys): skip it.\n  if (tid == 0u) {\n    let b = row0 / L;\n    let first = kinfo[2u * b];\n    let last = kinfo[2u * b + 1u];\n    tileLive = select(0u, 1u, first < L && j0 <= last && j0 + 63u >= first);\n  }\n  if (workgroupUniformLoad(&tileLive) == 0u) { return; }\n  var acc: array<vec4<f32>, 4>;\n  for (var t4 = 0u; t4 < D4; t4 += 8u) {\n    for (var run = tid; run < 256u; run += 128u) {\n      let r = row0 + run / 8u;\n      sa[run] = select(vec4<{{F}}>(0.0), qkv[r * row4 + h * D4 + t4 + run % 8u], r < ROWS);\n    }\n    // keys in groups of 4 (transposed in registers), 16 groups x 8 d chunks\n    let ng = tid / 8u;\n    let kv = tid % 8u;\n    var m: array<vec4<{{F}}>, 4>;\n    for (var q = 0u; q < 4u; q += 1u) {\n      let j = j0 + ng * 4u + q;\n      m[q] = select(vec4<{{F}}>(0.0), qkv[(kbase + j) * row4 + (H + h) * D4 + t4 + kv], j < L);\n    }\n    for (var e = 0u; e < 4u; e += 1u) {\n      sw[(kv * 4u + e) * 16u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sw[(k4 * 4u) * 16u + lid.x]);\n      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 16u + lid.x]);\n      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 16u + lid.x]);\n      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 16u + lid.x]);\n      for (var i = 0u; i < 4u; i += 1u) {\n        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let j = j0 + lid.x * 4u;\n  for (var i = 0u; i < 4u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= ROWS) { continue; }\n    for (var c = 0u; c < 4u; c += 1u) {\n      if (j + c < L) {\n        scores[(row * H + h) * L + j + c] = acc[i][c] * SCALE;\n      }\n    }\n  }\n}\n", Pe = "// Attention softmax over the rows of attscore.wgsl (K27), in place and in f32: one workgroup\n// per (head, query row). Keys with mask 0 and keys outside the sliding window (WINDOW > 0,\n// |i - j| <= WINDOW) get probability exactly 0; a masked query row gets 0 everywhere (its\n// context row is zero, as in mbattention.wgsl). Every unmasked query has at least itself as\n// a key, so the sum is never 0 there.\n\noverride L: u32 = 128u;\noverride H: u32 = 12u;\noverride WINDOW: u32 = 0u; // 0 = global attention\n\n@group(0) @binding(0) var<storage, read> mask: array<f32>;\n@group(0) @binding(1) var<storage, read_write> scores: array<f32>;\n\nvar<workgroup> red: array<f32, 64>;\n\nfn keep(il: u32, j: u32, kbase: u32) -> bool {\n  var k = mask[kbase + j] > 0.5;\n  if (WINDOW > 0u) {\n    let dist = select(il - j, j - il, j > il);\n    k = k && dist <= WINDOW;\n  }\n  return k;\n}\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let kbase = (i / L) * L;\n  let il = i - kbase;\n  let base = (i * H + h) * L;\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) {\n    if (keep(il, j, kbase)) { m = max(m, scores[base + j]); }\n  }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    var e = 0.0;\n    if (keep(il, j, kbase)) { e = exp(scores[base + j] - mx); }\n    scores[base + j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let inv = select(0.0, 1.0 / red[0], mask[i] > 0.5 && red[0] > 0.0);\n  for (var j = lid.x; j < L; j += 64u) {\n    scores[base + j] *= inv;\n  }\n}\n", Fe = "// DeBERTa attention softmax (K27), in place and in f32: one workgroup per (head, query row).\n// The scores of attscore.wgsl hold (q_i . k_j) / SCALE; this kernel adds the relative terms\n// (c2p[i][h][m] + p2c[j][h][m]) / SCALE with m = relidx[i][j] - MOFF (attrel.wgsl), then\n// runs the masked softmax: keys with mask 0 get probability exactly 0, a masked query row\n// gets 0 everywhere. Same score as attention.wgsl: (q.k + q.pos_key[m] + k.pos_query[m]) / SCALE.\n\noverride L: u32 = 128u;\noverride H: u32 = 12u;\noverride NM: u32 = 256u;\noverride MOFF: u32 = 0u;\noverride INVSCALE: f32 = 0.0721688; // 1 / sqrt(64 * 3)\n\n@group(0) @binding(0) var<storage, read> mask: array<f32>;\n@group(0) @binding(1) var<storage, read> relidx: array<u32>;\n@group(0) @binding(2) var<storage, read> c2p: array<f32>;\n@group(0) @binding(3) var<storage, read> p2c: array<f32>;\n@group(0) @binding(4) var<storage, read_write> scores: array<f32>;\n\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let kbase = (i / L) * L;\n  let il = i - kbase;\n  let base = (i * H + h) * L;\n  let cbase = (i * H + h) * NM;\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) {\n    if (mask[kbase + j] > 0.5) {\n      let p = relidx[il * L + j] - MOFF;\n      let s = scores[base + j] + (c2p[cbase + p] + p2c[((kbase + j) * H + h) * NM + p]) * INVSCALE;\n      scores[base + j] = s;\n      m = max(m, s);\n    }\n  }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    var e = 0.0;\n    if (mask[kbase + j] > 0.5) { e = exp(scores[base + j] - mx); }\n    scores[base + j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let inv = select(0.0, 1.0 / red[0], mask[i] > 0.5 && red[0] > 0.0);\n  for (var j = lid.x; j < L; j += 64u) {\n    scores[base + j] *= inv;\n  }\n}\n", Ie = "{{ENABLE}}// Embedding sum and LayerNorm, one workgroup per row, f32 accumulation.\n// out[row, :] = LN(word[row, :] + pos[(row % L) + OFFSET, :] + typ[tt[row], :])\n//   * (MASKMUL == 1 ? mask[row] : 1)\n// The three-table sum is formed in f32; the result is rounded once on write.\n// L is the bucket length: in a batch, row r = b * L + i reads position i.\n// Rows past the position table (a bucket longer than MAXPOS - OFFSET) are padding and masked: the\n// position index is clamped to the last row, valid rows (i < MAXPOS - OFFSET) never reach the clamp.\n// Families without a type table bind a zero row and zero ids.\n\noverride N: u32 = 384u;\noverride L: u32 = 128u;\noverride OFFSET: u32 = 0u;\noverride MAXPOS: u32 = 0xffffffffu;\noverride EPS: f32 = 1e-12;\noverride MASKMUL: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> word: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> pos: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> typ: array<{{F}}>;\n@group(0) @binding(3) var<storage, read> tt: array<u32>;\n@group(0) @binding(4) var<storage, read> weight: array<{{F}}>;\n@group(0) @binding(5) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(6) var<storage, read> mask: array<f32>;\n@group(0) @binding(7) var<storage, read_write> out: array<{{F}}>;\n\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let base = row * N;\n  let pbase = min((row % L) + OFFSET, MAXPOS - 1u) * N;\n  let tbase = tt[row] * N;\n  var s = 0.0;\n  var sq = 0.0;\n  for (var i = lid.x; i < N; i += 64u) {\n    let v = f32(word[base + i]) + f32(pos[pbase + i]) + f32(typ[tbase + i]);\n    s += v;\n    sq += v * v;\n  }\n  red[lid.x] = s;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let mean = red[0] / f32(N);\n  workgroupBarrier();\n  red[lid.x] = sq;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let variance = red[0] / f32(N) - mean * mean;\n  let inv = 1.0 / sqrt(variance + EPS);\n  for (var i = lid.x; i < N; i += 64u) {\n    let v = f32(word[base + i]) + f32(pos[pbase + i]) + f32(typ[tbase + i]);\n    var y = (v - mean) * inv * f32(weight[i]) + f32(bias[i]);\n    if (MASKMUL == 1u) { y *= mask[row]; }\n    out[base + i] = {{F}}(y);\n  }\n}\n", Le = "{{ENABLE}}// states[g, :] = x[markers[g], :] — collect hidden states at the\n// classification marker positions. One workgroup, threads stride over rows*D.\n// Batch (K16): markers hold B blocks of 3*K (indices, mask bits, groups);\n// output row g belongs to sequence b = g / K and reads x at the\n// sequence-local marker index plus b * L.\n\noverride K: u32 = 16u;\noverride D: u32 = 384u;\noverride L: u32 = 128u;\n\n@group(0) @binding(0) var<storage, read> markers: array<u32>;\n@group(0) @binding(1) var<storage, read> x: array<{{F}}>;\n@group(0) @binding(2) var<storage, read_write> states: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  // The bound size encodes the batch: 3*K u32 per sequence.\n  let total = arrayLength(&markers) / 3u;\n  for (var i = lid.x; i < total * D; i += 64u) {\n    let g = i / D;\n    let b = g / K;\n    let d = i - g * D;\n    states[i] = x[(b * L + markers[b * 3u * K + (g - b * K)]) * D + d];\n  }\n}\n", Re = "{{ENABLE}}// GeGLU epilogue for the ModernBERT FFN: mid is the raw [L, 2*I]\n// projection; out[row, j] = gelu(mid[row, j]) * mid[row, I + j] for j < I.\n// gelu is the exact erf form via Abramowitz-Stegun 7.1.26 (same constants\n// as matmul.wgsl ACT=2). One workgroup per row.\n\noverride I: u32 = 1152u;\n\n@group(0) @binding(0) var<storage, read> mid: array<{{F}}>;\n@group(0) @binding(1) var<storage, read_write> gate: array<{{F}}>;\n\nfn gelu(v: f32) -> f32 {\n  let u = v * 0.7071067811865476;\n  let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n  let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n    - 0.284496736) * t + 0.254829592) * t;\n  let e = 1.0 - p * exp(-u * u);\n  return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n}\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let base = row * 2u * I;\n  for (var j = lid.x; j < I; j += 64u) {\n    gate[row * I + j] = {{F}}(\n      gelu(f32(mid[base + j])) * f32(mid[base + I + j]));\n  }\n}\n", ze = "{{ENABLE}}// im2col for a 1-D convolution over packed sequences, one workgroup per row.\n// Row r = b * L + i of out holds the KS rows emb[b * L + i + t - PAD], t = 0..KS-1, side by side\n// (block t at columns t * N); a source position outside [0, L) of the own sequence reads as zero, and so\n// does a source row with mask 0: the executor dispatches only the rows below seqLen, so a padding row\n// of emb can hold stale values of an earlier call instead of the zeros of the embedding.\n// emb is [rows, N], out is [rows, KS * N]. Pure copy, no arithmetic.\n\noverride N: u32 = 768u;\noverride L: u32 = 128u;\noverride KS: u32 = 3u;\n\n@group(0) @binding(0) var<storage, read> emb: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> out: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let seq = row / L;\n  let i = i32(row % L);\n  let pad = i32((KS - 1u) / 2u);\n  for (var t = 0u; t < KS; t += 1u) {\n    let j = i + i32(t) - pad;\n    let srcRow = seq * L + u32(max(j, 0));\n    let ok = j >= 0 && j < i32(L) && mask[srcRow] != 0.0;\n    let src = srcRow * N;\n    for (var c = lid.x; c < N; c += 64u) {\n      var v = {{F}}(0.0);\n      if (ok) { v = emb[src + c]; }\n      out[row * KS * N + t * N + c] = v;\n    }\n  }\n}\n", Be = "{{ENABLE}}// LayerNorm per row, one workgroup per row, f32 accumulation.\n// MODE 0: out = LN(a)\n// MODE 1: out = LN(a) * mask[row]   (embedding masking)\n// MODE 2: out = LN(a + b)          (residual add fused)\n// mean/var over the last dimension with manifest epsilon (1e-7).\n\noverride N: u32 = 384u;\noverride MODE: u32 = 0u;\noverride EPS: f32 = 1e-7;\n\n@group(0) @binding(0) var<storage, read> a: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> b: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> weight: array<{{F}}>;\n@group(0) @binding(3) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(4) var<storage, read> mask: array<f32>;\n@group(0) @binding(5) var<storage, read_write> out: array<{{F}}>;\n\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let base = row * N;\n  var s = 0.0;\n  var sq = 0.0;\n  for (var i = lid.x; i < N; i += 64u) {\n    var v = f32(a[base + i]);\n    if (MODE == 2u) { v += f32(b[base + i]); }\n    s += v;\n    sq += v * v;\n  }\n  red[lid.x] = s;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let mean = red[0] / f32(N);\n  workgroupBarrier();\n  red[lid.x] = sq;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let variance = red[0] / f32(N) - mean * mean;\n  let inv = 1.0 / sqrt(variance + EPS);\n  for (var i = lid.x; i < N; i += 64u) {\n    var v = f32(a[base + i]);\n    if (MODE == 2u) { v += f32(b[base + i]); }\n    var y = (v - mean) * inv * f32(weight[i]) + f32(bias[i]);\n    if (MODE == 1u) { y *= mask[row]; }\n    out[base + i] = {{F}}(y);\n  }\n}\n", Ve = "{{ENABLE}}// logits[k] = raw[k] / TEMP where markerMask[k] > 0.5 else -1e4.\n// Marker payloads are packed u32: [0..K) marker_indices, [K..2K)\n// marker_mask as f32 bits, [2K..3K) marker_groups (resolved on the CPU).\n// Batch (K16): packed holds B blocks of 3*K; output k maps to sequence\n// b = k / K and mask element b*3K + K + (k mod K).\n\noverride K: u32 = 16u;\noverride TEMP: f32 = 1.0;\n\n@group(0) @binding(0) var<storage, read> raw: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> packed: array<u32>;\n@group(0) @binding(2) var<storage, read_write> logits: array<f32>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let total = arrayLength(&packed) / 3u;\n  for (var k = lid.x; k < total; k += 64u) {\n    let b = k / K;\n    let m = bitcast<f32>(packed[b * 3u * K + K + (k - b * K)]);\n    logits[k] = select(-1e4, f32(raw[k]) / TEMP, m > 0.5);\n  }\n}\n", He = "{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N]. Row-major storage, 16x16 tiles,\n// f32 accumulation. ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26), 3 tanh, 4 silu.\n\noverride M: u32 = 1u;\noverride N: u32 = 1u;\noverride K: u32 = 1u;\noverride ACT: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> a: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> w: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;\n\nvar<workgroup> ta: array<f32, 256>;\nvar<workgroup> tw: array<f32, 256>;\n\nfn activate(v: f32) -> f32 {\n  if (ACT == 1u) { return max(v, 0.0); }\n  if (ACT == 2u) {\n    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26\n    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.\n    let u = v * 0.7071067811865476;\n    let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n      - 0.284496736) * t + 0.254829592) * t;\n    let e = 1.0 - p * exp(-u * u);\n    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n  }\n  if (ACT == 3u) { return tanh(v); }\n  if (ACT == 4u) { return v / (1.0 + exp(-v)); }\n  return v;\n}\n\n@compute @workgroup_size(16, 16)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.y * 16u + lid.y;\n  let col = wid.x * 16u + lid.x;\n  var acc = 0.0;\n  for (var t = 0u; t < K; t += 16u) {\n    let ai = t + lid.x;\n    let wi = t + lid.y;\n    ta[lid.y * 16u + lid.x] = select(0.0, f32(a[row * K + ai]), row < M && ai < K);\n    tw[lid.y * 16u + lid.x] = select(0.0, f32(w[col * K + wi]), col < N && wi < K);\n    workgroupBarrier();\n    for (var k = 0u; k < 16u; k += 1u) {\n      acc += ta[lid.y * 16u + k] * tw[k * 16u + lid.x];\n    }\n    workgroupBarrier();\n  }\n  if (row < M && col < N) {\n    c[row * N + col] = {{F}}(activate(acc + f32(bias[col])));\n  }\n}\n", Ue = "{{ENABLE}}// ModernBERT multi-head attention, one workgroup per (head, query\n// row). scores[i,j] = (q_i . k_j) / sqrt(D) over key positions, masked by the\n// key mask and (WINDOW > 0) the sliding window |i - j| <= WINDOW; softmax in\n// f32 (fully masked rows give a uniform distribution, never NaN), then . v.\n// RoPE is applied upstream to q and k inside qkv; layout matches the DeBERTa\n// kernel: row i holds [q | k | v] of H * D each.\n// The q row is hoisted into registers once; masked query rows write zeros\n// and skip the loop.\n// Batch (K16): B sequences are packed as B*L global rows; row r = b*L + i\n// attends to keys b*L + j and the sliding window compares local i to j.\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 0.125; // 64^-0.5\noverride WINDOW: u32 = 0u;   // 0 = global attention\n\n@group(0) @binding(0) var<storage, read> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> ctx: array<{{F}}>;\n\n// One score per bucket position; the array follows the L override, so a\n// L1024 pipeline takes 4 KiB (well under the 16 KiB minimum limit).\nvar<workgroup> scores: array<f32, L>;\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let b = i / L;\n  let il = i - b * L;\n  let kbase = b * L;\n  let hd = H * D;\n  let qb = i * 3u * hd + h * D;\n  var qreg: array<f32, 64>;\n  for (var d = 0u; d < D; d += 1u) {\n    qreg[d] = f32(qkv[qb + d]);\n  }\n  // Masked query rows contribute nothing downstream (their ctx feeds only\n  // their own row), so they can be zeroed and skipped.\n  if (mask[i] <= 0.5) {\n    for (var d = lid.x; d < D; d += 64u) {\n      ctx[i * hd + h * D + d] = {{F}}(0.0);\n    }\n    return;\n  }\n  for (var j = lid.x; j < L; j += 64u) {\n    var outside = mask[kbase + j] <= 0.5;\n    if (WINDOW > 0u) {\n      let dist = select(il - j, j - il, j > il);\n      outside = outside || dist > WINDOW;\n    }\n    // Masked keys land at -1e30 regardless of the dot product.\n    // The sentinel sits below any real score; the softmax max also starts at -1e30.\n    if (outside) {\n      scores[j] = -1e30;\n      continue;\n    }\n    var s = 0.0;\n    let kb = (kbase + j) * 3u * hd + (H + h) * D;\n    for (var d = 0u; d < D; d += 4u) {\n      s += qreg[d] * f32(qkv[kb + d])\n        + qreg[d + 1u] * f32(qkv[kb + d + 1u])\n        + qreg[d + 2u] * f32(qkv[kb + d + 2u])\n        + qreg[d + 3u] * f32(qkv[kb + d + 3u]);\n    }\n    scores[j] = s * SCALE;\n  }\n  workgroupBarrier();\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) { m = max(m, scores[j]); }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    let e = exp(scores[j] - mx);\n    scores[j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let invTotal = 1.0 / red[0];\n  workgroupBarrier();\n  for (var d = lid.x; d < D; d += 64u) {\n    let vb = (2u * H + h) * D + d;\n    // Softmax normalization folds into the epilogue (ctx = acc / total):\n    // exp() underflows to exactly 0 for masked keys, so the sv != 0 branch\n    // keeps stale rows out, and four accumulators shorten the serial chain.\n    var acc0 = 0.0;\n    var acc1 = 0.0;\n    var acc2 = 0.0;\n    var acc3 = 0.0;\n    for (var j = 0u; j + 3u < L; j += 4u) {\n      let s0 = scores[j];\n      let s1 = scores[j + 1u];\n      let s2 = scores[j + 2u];\n      let s3 = scores[j + 3u];\n      if (s0 != 0.0) { acc0 += s0 * f32(qkv[(kbase + j + 0u) * 3u * hd + vb]); }\n      if (s1 != 0.0) { acc1 += s1 * f32(qkv[(kbase + j + 1u) * 3u * hd + vb]); }\n      if (s2 != 0.0) { acc2 += s2 * f32(qkv[(kbase + j + 2u) * 3u * hd + vb]); }\n      if (s3 != 0.0) { acc3 += s3 * f32(qkv[(kbase + j + 3u) * 3u * hd + vb]); }\n    }\n    var acc = (acc0 + acc1 + acc2 + acc3) * invTotal;\n    ctx[i * hd + h * D + d] = {{F}}(acc);\n  }\n}\n", We = "{{ENABLE}}// Standard multi-head attention, flash form (K27): one workgroup per (head, block of 32\n// query rows), one thread per query row. Keys and values stream through workgroup memory in\n// blocks of 32 (vec4 of the storage type, 2 x 32 x D values); every thread keeps its q row\n// and its output row in registers and runs an online softmax in f32. Masked keys and keys\n// outside the sliding window (WINDOW > 0, |i - j| <= WINDOW) are skipped, so stale padding\n// rows never enter a sum; masked query rows write zeros. Layout as mbattention.wgsl: row i\n// holds [q | k | v] of H * D each, B sequences packed as B * L rows. Needs D % 4 == 0,\n// D <= 64 and L % 32 == 0 (a block never straddles two sequences).\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 0.125; // 64^-0.5\noverride WINDOW: u32 = 0u;   // 0 = global attention\noverride ROWS: u32 = 128u;   // B * L, rows of qkv\n\n@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> ctx: array<vec4<{{F}}>>;\n\nvar<workgroup> sk: array<vec4<{{F}}>, 512>; // [key][d/4], 32 keys x 16\nvar<workgroup> sv: array<vec4<{{F}}>, 512>;\n\n@compute @workgroup_size(32)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y * 32u + lid.x;\n  let b = (wid.y * 32u) / L;   // the whole block lies in sequence b\n  let il = i - b * L;\n  let kbase = b * L;\n  let D4 = D / 4u;\n  let hd4 = H * D4;            // vec4 per q, k or v part of a row\n  let row4 = 3u * hd4;\n  let live = i < ROWS && mask[min(i, ROWS - 1u)] > 0.5;\n  var q: array<vec4<f32>, 16>;\n  var o: array<vec4<f32>, 16>;\n  if (live) {\n    for (var d = 0u; d < D4; d += 1u) {\n      q[d] = vec4<f32>(qkv[i * row4 + h * D4 + d]) * SCALE;\n    }\n  }\n  var m = -1e30;\n  var l = 0.0;\n  for (var j0 = 0u; j0 < L; j0 += 32u) {\n    // 32 keys x D4 vec4 per operand, 32 threads.\n    for (var e = lid.x; e < 32u * D4; e += 32u) {\n      let kj = kbase + j0 + e / D4;\n      let d = e % D4;\n      sk[e] = qkv[kj * row4 + (H + h) * D4 + d];\n      sv[e] = qkv[kj * row4 + (2u * H + h) * D4 + d];\n    }\n    workgroupBarrier();\n    if (live) {\n      for (var jj = 0u; jj < 32u; jj += 1u) {\n        let j = j0 + jj;\n        var skip = mask[kbase + j] <= 0.5;\n        if (WINDOW > 0u) {\n          let dist = select(il - j, j - il, j > il);\n          skip = skip || dist > WINDOW;\n        }\n        if (skip) { continue; }\n        var s = 0.0;\n        for (var d = 0u; d < D4; d += 1u) {\n          s += dot(q[d], vec4<f32>(sk[jj * D4 + d]));\n        }\n        if (s > m) {\n          let c = exp(m - s);\n          l *= c;\n          for (var d = 0u; d < D4; d += 1u) { o[d] *= c; }\n          m = s;\n        }\n        let p = exp(s - m);\n        l += p;\n        for (var d = 0u; d < D4; d += 1u) {\n          o[d] += p * vec4<f32>(sv[jj * D4 + d]);\n        }\n      }\n    }\n    workgroupBarrier();\n  }\n  if (i < ROWS) {\n    let inv = select(0.0, 1.0 / l, live);\n    for (var d = 0u; d < D4; d += 1u) {\n      ctx[i * hd4 + h * D4 + d] = vec4<{{F}}>(o[d] * inv);\n    }\n  }\n}\n", Ge = "{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N], register-blocked (K27, tools/k27_mm_gen.py mmtile 16 8 4 32).\n// Workgroup 16x8 threads, 4x4 outputs per thread, tile 32x64, K step 32; tiles as vec4 of\n// the storage type in workgroup memory (A [row][k/4], W [k][col/4]). Every output sums k in order\n// in f32 and adds the bias last, like matmul.wgsl. Needs K % 32 == 0.\n// dispatch: 32x64 (tile rows x cols)\n// ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26), 3 tanh, 4 silu.\n\noverride M: u32 = 1u;\noverride N: u32 = 1u;\noverride K: u32 = 32u;\noverride ACT: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> a: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> w: array<vec4<{{F}}>>;\n@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;\n\nvar<workgroup> sa: array<vec4<{{F}}>, 256>;\nvar<workgroup> sw: array<vec4<{{F}}>, 512>;\n\nfn activate(v: f32) -> f32 {\n  if (ACT == 1u) { return max(v, 0.0); }\n  if (ACT == 2u) {\n    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26\n    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.\n    let u = v * 0.7071067811865476;\n    let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n      - 0.284496736) * t + 0.254829592) * t;\n    let e = 1.0 - p * exp(-u * u);\n    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n  }\n  if (ACT == 3u) { return tanh(v); }\n  if (ACT == 4u) { return v / (1.0 + exp(-v)); }\n  return v;\n}\n\n@compute @workgroup_size(16, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let tid = lid.y * 16u + lid.x;\n  let row0 = wid.y * 32u;\n  let col0 = wid.x * 64u;\n  let tileRow = lid.y * 4u;\n  let K4 = K / 4u;\n  var acc: array<vec4<f32>, 4>;\n  for (var t4 = 0u; t4 < K4; t4 += 8u) {\n    for (var run = tid; run < 256u; run += 128u) {\n      let ar = row0 + run / 8u;\n      sa[run] = select(vec4<{{F}}>(0.0), a[ar * K4 + t4 + run % 8u], ar < M);\n    }\n    for (var g = tid; g < 128u; g += 128u) {\n      let ng = g / 8u;\n      let kv = g % 8u;\n      var m: array<vec4<{{F}}>, 4>;\n      for (var q = 0u; q < 4u; q += 1u) {\n        let wr = col0 + ng * 4u + q;\n        m[q] = select(vec4<{{F}}>(0.0), w[wr * K4 + t4 + kv], wr < N);\n      }\n      for (var e = 0u; e < 4u; e += 1u) {\n        sw[(kv * 4u + e) * 16u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);\n      }\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sw[(k4 * 4u) * 16u + lid.x]);\n      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 16u + lid.x]);\n      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 16u + lid.x]);\n      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 16u + lid.x]);\n      for (var i = 0u; i < 4u; i += 1u) {\n        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let col = col0 + lid.x * 4u;\n  for (var i = 0u; i < 4u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= M) { continue; }\n    for (var j = 0u; j < 4u; j += 1u) {\n      if (col + j < N) {\n        c[row * N + col + j] = {{F}}(activate(acc[i][j] + f32(bias[col + j])));\n      }\n    }\n  }\n}\n", Ke = "{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N], register-blocked (K27, tools/k27_mm_gen.py mmtile16 8 8 2 32).\n// Workgroup 8x8 threads, 2x4 outputs per thread, tile 16x32, K step 32; tiles as vec4 of\n// the storage type in workgroup memory (A [row][k/4], W [k][col/4]). Every output sums k in order\n// in f32 and adds the bias last, like matmul.wgsl. Needs K % 32 == 0.\n// dispatch: 16x32 (tile rows x cols)\n// ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26), 3 tanh, 4 silu.\n\noverride M: u32 = 1u;\noverride N: u32 = 1u;\noverride K: u32 = 32u;\noverride ACT: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> a: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> w: array<vec4<{{F}}>>;\n@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;\n\nvar<workgroup> sa: array<vec4<{{F}}>, 128>;\nvar<workgroup> sw: array<vec4<{{F}}>, 256>;\n\nfn activate(v: f32) -> f32 {\n  if (ACT == 1u) { return max(v, 0.0); }\n  if (ACT == 2u) {\n    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26\n    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.\n    let u = v * 0.7071067811865476;\n    let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n      - 0.284496736) * t + 0.254829592) * t;\n    let e = 1.0 - p * exp(-u * u);\n    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n  }\n  if (ACT == 3u) { return tanh(v); }\n  if (ACT == 4u) { return v / (1.0 + exp(-v)); }\n  return v;\n}\n\n@compute @workgroup_size(8, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let tid = lid.y * 8u + lid.x;\n  let row0 = wid.y * 16u;\n  let col0 = wid.x * 32u;\n  let tileRow = lid.y * 2u;\n  let K4 = K / 4u;\n  var acc: array<vec4<f32>, 2>;\n  for (var t4 = 0u; t4 < K4; t4 += 8u) {\n    for (var run = tid; run < 128u; run += 64u) {\n      let ar = row0 + run / 8u;\n      sa[run] = select(vec4<{{F}}>(0.0), a[ar * K4 + t4 + run % 8u], ar < M);\n    }\n    for (var g = tid; g < 64u; g += 64u) {\n      let ng = g / 8u;\n      let kv = g % 8u;\n      var m: array<vec4<{{F}}>, 4>;\n      for (var q = 0u; q < 4u; q += 1u) {\n        let wr = col0 + ng * 4u + q;\n        m[q] = select(vec4<{{F}}>(0.0), w[wr * K4 + t4 + kv], wr < N);\n      }\n      for (var e = 0u; e < 4u; e += 1u) {\n        sw[(kv * 4u + e) * 8u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);\n      }\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sw[(k4 * 4u) * 8u + lid.x]);\n      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 8u + lid.x]);\n      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 8u + lid.x]);\n      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 8u + lid.x]);\n      for (var i = 0u; i < 2u; i += 1u) {\n        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let col = col0 + lid.x * 4u;\n  for (var i = 0u; i < 2u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= M) { continue; }\n    for (var j = 0u; j < 4u; j += 1u) {\n      if (col + j < N) {\n        c[row * N + col + j] = {{F}}(activate(acc[i][j] + f32(bias[col + j])));\n      }\n    }\n  }\n}\n", qe = "{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N], register-blocked (K27, tools/k27_mm_gen.py mmtile8 8 8 1 32).\n// Workgroup 8x8 threads, 1x4 outputs per thread, tile 8x32, K step 32; tiles as vec4 of\n// the storage type in workgroup memory (A [row][k/4], W [k][col/4]). Every output sums k in order\n// in f32 and adds the bias last, like matmul.wgsl. Needs K % 32 == 0.\n// dispatch: 8x32 (tile rows x cols)\n// ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26), 3 tanh, 4 silu.\n\noverride M: u32 = 1u;\noverride N: u32 = 1u;\noverride K: u32 = 32u;\noverride ACT: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> a: array<vec4<{{F}}>>;\n@group(0) @binding(1) var<storage, read> w: array<vec4<{{F}}>>;\n@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;\n\nvar<workgroup> sa: array<vec4<{{F}}>, 64>;\nvar<workgroup> sw: array<vec4<{{F}}>, 256>;\n\nfn activate(v: f32) -> f32 {\n  if (ACT == 1u) { return max(v, 0.0); }\n  if (ACT == 2u) {\n    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26\n    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.\n    let u = v * 0.7071067811865476;\n    let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n      - 0.284496736) * t + 0.254829592) * t;\n    let e = 1.0 - p * exp(-u * u);\n    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n  }\n  if (ACT == 3u) { return tanh(v); }\n  if (ACT == 4u) { return v / (1.0 + exp(-v)); }\n  return v;\n}\n\n@compute @workgroup_size(8, 8)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let tid = lid.y * 8u + lid.x;\n  let row0 = wid.y * 8u;\n  let col0 = wid.x * 32u;\n  let tileRow = lid.y * 1u;\n  let K4 = K / 4u;\n  var acc: array<vec4<f32>, 1>;\n  for (var t4 = 0u; t4 < K4; t4 += 8u) {\n    for (var run = tid; run < 64u; run += 64u) {\n      let ar = row0 + run / 8u;\n      sa[run] = select(vec4<{{F}}>(0.0), a[ar * K4 + t4 + run % 8u], ar < M);\n    }\n    for (var g = tid; g < 64u; g += 64u) {\n      let ng = g / 8u;\n      let kv = g % 8u;\n      var m: array<vec4<{{F}}>, 4>;\n      for (var q = 0u; q < 4u; q += 1u) {\n        let wr = col0 + ng * 4u + q;\n        m[q] = select(vec4<{{F}}>(0.0), w[wr * K4 + t4 + kv], wr < N);\n      }\n      for (var e = 0u; e < 4u; e += 1u) {\n        sw[(kv * 4u + e) * 8u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);\n      }\n    }\n    workgroupBarrier();\n    for (var k4 = 0u; k4 < 8u; k4 += 1u) {\n      let b0 = vec4<f32>(sw[(k4 * 4u) * 8u + lid.x]);\n      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 8u + lid.x]);\n      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 8u + lid.x]);\n      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 8u + lid.x]);\n      for (var i = 0u; i < 1u; i += 1u) {\n        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);\n        acc[i] = b0 * av.x + acc[i];\n        acc[i] = b1 * av.y + acc[i];\n        acc[i] = b2 * av.z + acc[i];\n        acc[i] = b3 * av.w + acc[i];\n      }\n    }\n    workgroupBarrier();\n  }\n  let col = col0 + lid.x * 4u;\n  for (var i = 0u; i < 1u; i += 1u) {\n    let row = row0 + tileRow + i;\n    if (row >= M) { continue; }\n    for (var j = 0u; j < 4u; j += 1u) {\n      if (col + j < N) {\n        c[row * N + col + j] = {{F}}(activate(acc[i][j] + f32(bias[col + j])));\n      }\n    }\n  }\n}\n", Je = "{{ENABLE}}// Masked pooling per sequence, f32 accumulation. One workgroup per\n// (64 columns, sequence); each thread owns one column and loops over the L\n// rows of its sequence. Rows of sequence b are b*L .. b*L + L - 1.\n// MODE 0: out[b, d] = sum_i mask * x / max(sum_i mask, 1e-9)   (mean)\n// MODE 1: out[b, d] = max over i with mask > 0.5 of x           (max)\n// The 1e-9 floor is the clamp of sentence-transformers. MODE 1 starts at the\n// first valid row (a found flag, no sentinel), so every finite value is a\n// valid maximum; a sequence without a valid row gives 0.\n\noverride L: u32 = 128u;\noverride N: u32 = 384u;\noverride MODE: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> x: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> out: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let d = wid.x * 64u + lid.x;\n  let b = wid.y;\n  if (d >= N) { return; }\n  let rbase = b * L;\n  if (MODE == 0u) {\n    var acc = 0.0;\n    var cnt = 0.0;\n    for (var i = 0u; i < L; i += 1u) {\n      let m = mask[rbase + i];\n      // Rows past seqLen hold stale values; skip them instead of 0 * stale.\n      if (m != 0.0) {\n        acc += m * f32(x[(rbase + i) * N + d]);\n        cnt += m;\n      }\n    }\n    out[b * N + d] = {{F}}(acc / max(cnt, 1e-9));\n  } else {\n    var best = 0.0;\n    var found = false;\n    for (var i = 0u; i < L; i += 1u) {\n      if (mask[rbase + i] > 0.5) {\n        let v = f32(x[(rbase + i) * N + d]);\n        best = select(v, max(best, v), found);\n        found = true;\n      }\n    }\n    out[b * N + d] = {{F}}(best);\n  }\n}\n", Ye = "{{ENABLE}}// RoPE (rotate-half convention) applied in place to the q and k\n// sections of the qkv buffer. One workgroup of 32 threads per (position,\n// head, section): thread d handles the pair (d, d + D/2), so no shared\n// state is needed (threads d >= D / 2 return, so head widths below 64 work). cossin holds per-position tables [L, 2 * D]: cos at\n// i * 2 * D + d and sin at i * 2 * D + D + d.\n// out[d]      = x[d] * cos_d - x[d + half] * sin_d\n// out[d + half] = x[d + half] * cos_d + x[d] * sin_d\n// (cos/sin are indexed by d mod half, which is why both halves use the\n// same table entries.)\n// Batch (K16): i is a global row in B*L packed space; the rotary table is\n// indexed by the sequence-local position i mod L.\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\n\n@group(0) @binding(0) var<storage, read_write> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> cossin: array<f32>;\n\n@compute @workgroup_size(32)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let i = wid.x;\n  let il = i - (i / L) * L;\n  let h = wid.y / 2u;\n  let seg = wid.y % 2u; // 0 = q, 1 = k (v keeps absolute position, no RoPE)\n  let hd = H * D;\n  let base = i * 3u * hd + (seg * H + h) * D;\n  let half = D / 2u;\n  let d = lid.x;\n  if (d >= half) { return; }\n  let a = f32(qkv[base + d]);\n  let b = f32(qkv[base + d + half]);\n  let c = cossin[il * 2u * D + d];\n  let s = cossin[il * 2u * D + D + d];\n  qkv[base + d] = {{F}}(a * c - b * s);\n  qkv[base + d + half] = {{F}}(b * c + a * s);\n}\n";
//#endregion
//#region src/kernels/index.ts
function Xe(e, t) {
	return e.replace("{{ENABLE}}", t ? "enable f16;\n" : "").replaceAll("{{F}}", t ? "f16" : "f32");
}
var Ze = {
	add: ke,
	attention: Ae,
	attpv: je,
	attrel: Me,
	attscore: Ne,
	attsoftmax: Pe,
	attsoftrel: Fe,
	embln: Ie,
	gather: Le,
	geglu: Re,
	im2col: ze,
	layernorm: Be,
	masklogits: Ve,
	matmul: He,
	mbattention: Ue,
	mbflash: We,
	mmtile: Ge,
	mmtile16: Ke,
	mmtile8: qe,
	pool: Je,
	rope: Ye
}, Qe = {
	rw: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	rwSrc: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	storage: () => GPUBufferUsage.STORAGE,
	logits: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
}, $e = (e) => Object.keys(e).sort().map((t) => `${t}=${e[t]}`).join(","), G = class {
	device;
	plan;
	tensors;
	length;
	markers;
	batch;
	gpuBytes = 0;
	buffers = /* @__PURE__ */ new Map();
	ownedBufs = [];
	staging;
	capBuf;
	segments = [];
	typeTable;
	typeRow;
	constructor(e, t, n) {
		this.device = e, this.plan = t, this.tensors = n, this.length = t.length, this.markers = t.markers, this.batch = t.batch, this.buildBuffers(), this.buildSegments(), t.rowSelect && (this.typeTable = this.resolve(t.rowSelect.table), this.typeRow = this.resolve(t.rowSelect.dst));
	}
	track(e, t) {
		return this.ownedBufs.push(e), this.gpuBytes += t, e;
	}
	resolve(e) {
		if (e.startsWith("w:")) {
			let t = this.tensors.get(e.slice(2));
			if (!t) throw Error(`weight tensor ${e.slice(2)} missing`);
			return t;
		}
		let t = this.buffers.get(e);
		if (!t) throw Error(`plan buffer ${e} missing`);
		return t;
	}
	buildBuffers() {
		for (let e of this.plan.buffers) {
			let t = this.track(this.device.createBuffer({
				size: e.bytes,
				usage: Qe[e.usage]()
			}), e.bytes);
			this.buffers.set(e.id, t), e.init && this.device.queue.writeBuffer(t, 0, e.init);
		}
		let e = this.plan.output.bytes;
		this.staging = this.track(this.device.createBuffer({
			size: e,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		}), e);
	}
	buildSegments() {
		let e = /* @__PURE__ */ new Map(), t = (t) => {
			let n = `${t.kernel}|${$e(t.constants)}`, r = e.get(n);
			return r || (r = this.device.createComputePipeline({
				layout: "auto",
				compute: {
					module: this.device.createShaderModule({ code: Xe(Ze[t.kernel], this.plan.f16) }),
					entryPoint: "main",
					constants: t.constants
				}
			}), e.set(n, r)), r;
		}, n = (e, t) => this.device.createBindGroup({
			layout: e.getBindGroupLayout(0),
			entries: t.bind.map((e, t) => ({
				binding: t,
				resource: { buffer: this.resolve(e) }
			}))
		}), r = (e) => {
			let r = t(e);
			return {
				pipeline: r,
				bindGroup: n(r, e),
				dispatch: e.dispatch,
				name: e.name,
				alts: (e.alts ?? []).map((r) => {
					let i = t({
						kernel: r.kernel,
						constants: e.constants
					});
					return {
						pipeline: i,
						bindGroup: n(i, e),
						dispatch: r.dispatch,
						maxRows: r.maxRows
					};
				})
			};
		};
		for (let e of this.plan.segments) this.segments.push({
			name: e.name,
			prefix: e.prefix,
			ops: e.ops.map(r),
			captureOps: (e.captureOps ?? []).map(r),
			capture: (e.capture ?? []).map((e) => ({
				buffer: this.resolve(e.buffer),
				slot: e.slot
			})),
			skippable: !!e.skippable
		});
	}
	assertCapturable() {
		if (this.plan.f16) throw Error("capture needs an f32 plan: the capture copies and readCapture are f32");
		if (this.plan.batch !== 1) throw Error(`capture needs a plan of batch 1, this one has ${this.plan.batch}`);
		let t = this.plan.captureSlots * this.plan.captureSlotBytes, n = this.device.limits?.maxBufferSize ?? e.maxBufferSize;
		if (t > n) throw Error(`capture buffer needs ${t} B, maxBufferSize is ${n}`);
	}
	captureBuffer() {
		if (!this.capBuf) {
			let e = this.plan.captureSlots * this.plan.captureSlotBytes;
			this.capBuf = this.track(this.device.createBuffer({
				size: e,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			}), e);
		}
		return this.capBuf;
	}
	destroy() {
		for (let e of this.ownedBufs) e.destroy();
		this.ownedBufs.length = 0;
	}
	dim(e, t) {
		return e === "rows" ? t : e === "rows8" ? Math.ceil(t / 8) : e === "rows16" ? Math.ceil(t / 16) : e === "rows32" ? Math.ceil(t / 32) : e;
	}
	run(e, t, n) {
		let r = t.alts.find((e) => n <= e.maxRows) ?? t;
		e.setPipeline(r.pipeline), e.setBindGroup(0, r.bindGroup), e.dispatchWorkgroups(this.dim(r.dispatch[0], n), this.dim(r.dispatch[1], n));
	}
	encode(e, t = {}) {
		let { ts: n, skip: r, dispatchNames: i } = t;
		e && this.assertCapturable();
		let a = this.device.createCommandEncoder(), o = 0, s = () => {
			let e = n ? {
				querySet: n.querySet,
				beginningOfPassWriteIndex: o * 2,
				endOfPassWriteIndex: o * 2 + 1
			} : void 0;
			return o += 1, a.beginComputePass(e ? { timestampWrites: e } : void 0);
		}, c = e ? this.length : t.seqLen ?? this.length, l = (e) => e.skippable && r ? e.ops.filter((e, t) => !r.has(t)) : e.ops, u = this.plan.rowSelect;
		if (u) {
			let e = Array.isArray(t.qtype) ? t.qtype : [t.qtype ?? 0];
			for (let t = 0; t < this.batch; t += 1) a.copyBufferToBuffer(this.typeTable, (e[t] ?? 0) * u.rowBytes, this.typeRow, t * u.rowBytes, u.rowBytes);
		}
		if (i) {
			if (!n) throw Error("dispatch profile needs timestamp resources");
			for (let e of this.segments) for (let t of e.ops) {
				i.push(`${e.prefix}${t.name}`);
				let n = s();
				this.run(n, t, c), n.end();
			}
		} else if (!e && !n) {
			let e = s();
			for (let t of this.segments) for (let n of l(t)) this.run(e, n, c);
			e.end();
		} else {
			let t = this.plan.captureSlotBytes;
			for (let n of this.segments) {
				let r = s();
				if (e) for (let e of n.captureOps) this.run(r, e, c);
				for (let e of l(n)) this.run(r, e, c);
				if (r.end(), e) for (let e of n.capture) a.copyBufferToBuffer(e.buffer, 0, this.captureBuffer(), e.slot * t, t);
			}
		}
		return a.copyBufferToBuffer(this.resolve(this.plan.output.buffer), 0, this.staging, 0, this.plan.output.bytes), n && (a.resolveQuerySet(n.querySet, 0, o * 2, n.resolve, 0), a.copyBufferToBuffer(n.resolve, 0, n.staging, 0, o * 16)), a.finish();
	}
	submit(e, t = {}) {
		this.device.queue.submit([this.encode(e, t)]);
	}
	async kernelTimesMs(e, t = "pass", n = this.length) {
		let r = await this.profileForward(e, t, n);
		return r && r.times;
	}
	async profileForward(e, t = "pass", n = this.length) {
		if (!this.device.features.has("timestamp-query")) return null;
		let r = t === "dispatch", i = r ? this.segments.reduce((e, t) => e + t.ops.length, 0) : this.segments.length, a = this.timestampResources(i);
		this.upload(e);
		let o = r ? [] : this.segments.map((e) => e.name);
		if (this.device.queue.submit([this.encode(!1, {
			seqLen: r ? n : this.length,
			qtype: e.qtype,
			ts: a,
			dispatchNames: r ? o : void 0
		})]), o.length !== i) throw Error(`profile: ${o.length} timed passes, expected ${i}`);
		await a.staging.mapAsync(GPUMapMode.READ);
		let s = new BigUint64Array(a.staging.getMappedRange().slice(0));
		a.staging.unmap();
		let c = {};
		for (let e = 0; e < i; e += 1) c[o[e]] = Number(s[e * 2 + 1] - s[e * 2]) / 1e6;
		return {
			times: c,
			logits: await this.readLogits()
		};
	}
	tsResources;
	timestampResources(e) {
		if (this.tsResources && this.tsResources.count === e) return this.tsResources;
		this.tsResources && (this.tsResources.querySet.destroy(), this.tsResources.resolve.destroy(), this.tsResources.staging.destroy());
		let t = e * 16;
		return this.tsResources = {
			count: e,
			querySet: this.device.createQuerySet({
				type: "timestamp",
				count: e * 2
			}),
			resolve: this.device.createBuffer({
				size: t,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
			}),
			staging: this.device.createBuffer({
				size: t,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			})
		}, this.tsResources;
	}
	upload(e) {
		let t = this.plan.inputs;
		if (this.device.queue.writeBuffer(this.resolve(t.embeddings), 0, e.embeddings), this.device.queue.writeBuffer(this.resolve(t.mask), 0, e.mask), this.plan.buffers.some((e) => e.id === "kinfo")) {
			let t = this.plan.length, n = new Uint32Array(2 * this.plan.batch);
			for (let r = 0; r < this.plan.batch; r += 1) {
				let i = t, a = 0;
				for (let n = 0; n < t; n += 1) e.mask[r * t + n] > .5 && (i === t && (i = n), a = n);
				n[2 * r] = i, n[2 * r + 1] = a;
			}
			this.device.queue.writeBuffer(this.resolve("kinfo"), 0, n);
		}
		if (t.markers) {
			if (!e.packedMarkers) throw Error("plan needs packedMarkers");
			this.device.queue.writeBuffer(this.resolve(t.markers), 0, e.packedMarkers);
		}
		if (t.typeIds) {
			if (!e.typeIds) throw Error("plan needs typeIds");
			this.device.queue.writeBuffer(this.resolve(t.typeIds), 0, e.typeIds);
		}
	}
	async readOutput() {
		let e = this.plan.output;
		await this.staging.mapAsync(GPUMapMode.READ);
		let t = this.staging.getMappedRange().slice(0);
		this.staging.unmap();
		let n = e.rows * e.cols;
		return e.dtype === "storage" && this.plan.f16 ? Oe(new Uint16Array(t, 0, n)) : new Float32Array(t, 0, n);
	}
	async readLogits() {
		await this.staging.mapAsync(GPUMapMode.READ);
		let e = new Float32Array(this.staging.getMappedRange().slice(0));
		return this.staging.unmap(), e;
	}
	async readCapture() {
		let e = this.captureBuffer();
		await e.mapAsync(GPUMapMode.READ);
		let t = new Float32Array(e.getMappedRange().slice(0));
		return e.unmap(), t;
	}
};
//#endregion
//#region src/tasks.ts
function K(e, t) {
	let n = 0;
	for (let t = 0; t < e.length; t += 1) Number.isFinite(e[t]) || (n += 1);
	if (n) throw Error(`${t}: ${n} of ${e.length} values are not finite`);
}
function q(e, t, n) {
	if (!Number.isInteger(e) || e < 0 || (e + 1) * t > n) throw Error(`token id ${e} is outside the vocabulary of ${n / t} rows (ids are integers from 0)`);
	return e * t;
}
function et(e) {
	let t = new Float32Array(e.length), n = -Infinity;
	for (let t = 0; t < e.length; t += 1) n = Math.max(n, e[t]);
	let r = 0;
	for (let i = 0; i < e.length; i += 1) t[i] = Math.exp(e[i] - n), r += t[i];
	for (let e = 0; e < t.length; e += 1) t[e] /= r;
	return t;
}
function tt(e) {
	let t = 0;
	for (let n = 0; n < e.length; n += 1) t += e[n] * e[n];
	let n = Math.max(Math.sqrt(t), 1e-12), r = new Float32Array(e.length);
	for (let t = 0; t < e.length; t += 1) r[t] = e[t] / n;
	return r;
}
function nt(e, t) {
	let n = Array.from({ length: t }, (e, t) => `LABEL_${t}`);
	for (let [r, i] of Object.entries(e ?? {})) {
		let e = Number(r);
		if (!Number.isInteger(e) || String(e) !== r || e < 0 || e >= t) throw Error(`label id '${r}' is not an integer from 0 to ${t - 1}`);
		n[e] = i;
	}
	return n;
}
function rt(e, t) {
	if (!Number.isInteger(t) || t <= 0) throw Error(`argmaxRows needs a positive integer column count, got ${t}`);
	let n = Math.floor(e.length / t), r = [];
	for (let i = 0; i < n; i += 1) {
		let n = 0;
		for (let r = 1; r < t; r += 1) e[i * t + r] > e[i * t + n] && (n = r);
		r.push(n);
	}
	return r;
}
function it(e) {
	return e.startsWith("B-") ? {
		begin: !0,
		tag: e.slice(2)
	} : e.startsWith("I-") ? {
		begin: !1,
		tag: e.slice(2)
	} : {
		begin: !1,
		tag: e
	};
}
function at(e, t, n, r, i = ["O"]) {
	let a = t.length, o = [], s = rt(e, a);
	for (let i = 0; i < s.length; i += 1) r[i] || o.push({
		label: t[s[i]],
		score: e[i * a + s[i]],
		start: n[i][0],
		end: n[i][1]
	});
	let c = [], l = [], u = () => {
		if (!l.length) return;
		let e = l[0].label, t = e.indexOf("-"), n = 0;
		for (let e of l) n += e.score;
		c.push({
			group: t < 0 ? e : e.slice(t + 1),
			score: n / l.length,
			start: l[0].start,
			end: l[l.length - 1].end
		});
	};
	for (let e of o) {
		if (l.length) {
			let t = it(e.label), n = it(l[l.length - 1].label);
			(t.tag !== n.tag || t.begin) && (u(), l = []);
		}
		l.push(e);
	}
	return u(), c.filter((e) => !i.includes(e.group));
}
//#endregion
//#region src/tokenizer/bpe.ts
var ot = "▁";
function st(e) {
	let t = [];
	for (let n of new TextEncoder().encode(e)) t.push(`<0x${n.toString(16).toUpperCase().padStart(2, "0")}>`);
	return t;
}
var ct = class {
	vocab;
	ranks;
	ids;
	added;
	addedList;
	unkId;
	padTokenId = 0;
	bosTokenId = 2;
	eosTokenId = 1;
	maskTokenId = 4;
	maskToken = "<mask>";
	constructor(e) {
		this.vocab = new Map(Object.entries(e.model.vocab)), this.ids = new Map([...this.vocab].map(([e, t]) => [t, e])), this.ranks = /* @__PURE__ */ new Map(), e.model.merges.forEach(([e, t], n) => {
			this.ranks.set(`${e} ${t}`, n);
		}), this.unkId = this.vocab.get(e.model.unk_token ?? "<unk>") ?? 3, this.added = /* @__PURE__ */ new Map(), this.addedList = [];
		for (let t of e.added_tokens) this.added.set(t.content, t.id), this.addedList.push(t.content), t.content === "<pad>" ? this.padTokenId = t.id : t.content === "<bos>" ? this.bosTokenId = t.id : t.content === "<eos>" ? this.eosTokenId = t.id : t.content === "<mask>" && (this.maskTokenId = t.id);
		this.addedList.sort((e, t) => t.length - e.length);
	}
	bpe(e) {
		let t = [];
		for (let n of e) this.vocab.has(n) ? t.push(n) : t.push(...st(n));
		for (;;) {
			let e = Infinity, n = "";
			for (let r = 0; r + 1 < t.length; r += 1) {
				let i = this.ranks.get(`${t[r]} ${t[r + 1]}`) ?? Infinity;
				i < e && (e = i, n = `${t[r]} ${t[r + 1]}`);
			}
			if (e === Infinity) break;
			let r = n.replace(" ", ""), i = [];
			for (let e = 0; e < t.length; e += 1) e + 1 < t.length && `${t[e]} ${t[e + 1]}` === n ? (i.push(r), e += 1) : i.push(t[e]);
			t.length = 0, t.push(...i);
		}
		return t;
	}
	encodePlain(e) {
		let t = e.replaceAll(" ", ot);
		t.startsWith(ot) || (t = ot + t);
		let n = [];
		for (let e of t.matchAll(/▁[^▁]*/g)) for (let t of this.bpe(e[0])) n.push(this.vocab.get(t) ?? this.unkId);
		let r = [];
		for (let e of n) (e !== this.unkId || r[r.length - 1] !== this.unkId) && r.push(e);
		return r;
	}
	encodeIds(e) {
		let t = [], n = 0;
		for (; n < e.length;) {
			let r = -1, i = 0;
			for (let t of this.addedList) {
				let a = e.indexOf(t, n);
				a !== -1 && (r === -1 || a < r) && (r = a, i = t.length);
			}
			let a = r === -1 ? e.length : r;
			if (a > n && t.push(...this.encodePlain(e.slice(n, a))), r === -1) break;
			t.push(this.added.get(e.slice(r, r + i))), n = r + i;
		}
		return t;
	}
	idToToken(e) {
		return this.ids.get(e);
	}
}, lt = {
	choice: 0,
	score: 1,
	noul: 2
};
function ut(e, t, n = 1024, r = 256, i = !1) {
	let a = typeof t.state == "string" ? t.state : JSON.stringify(t.state);
	if (i && [
		a,
		t.question,
		...t.options
	].some((t) => t.includes(e.maskToken))) throw Error("reserved model marker in request");
	let o = (t) => t.replaceAll(e.maskToken, " "), s = t.type ?? "choice", c = e.encodeIds(`${s} question: ${o(t.question)}`), l = t.options.map((t) => e.encodeIds(` ${o(t)}`));
	if (i && l.some((e) => e.length > 48)) throw Error("option exceeds 48-token model contract");
	let u = l.map((t) => [e.maskTokenId, ...t.slice(0, 48)]), d = r - u.reduce((e, t) => e + t.length, 0);
	if (d < 16) {
		let e = Math.max(4, Math.floor((r - 16) / u.length));
		u = u.map((t) => t.slice(0, e)), d = r - u.reduce((e, t) => e + t.length, 0);
	}
	if (i && (c.length > d || u.some((e, t) => e.length !== l[t].length + 1))) throw Error("question/options exceed lossless head budget");
	let f = [
		e.bosTokenId,
		...c.slice(0, Math.max(8, d)),
		e.eosTokenId
	], p = [];
	for (let e of u) p.push(f.length), f.push(...e);
	f.push(e.eosTokenId);
	let m = e.encodeIds(o(a)), h = n - f.length - 1;
	if (h < 1) throw Error("question/options exceed sequence budget");
	let g = m.length > h;
	if (i && g) throw Error("state exceeds lossless context budget");
	return f.push(...m.slice(0, h), e.eosTokenId), {
		inputIds: Int32Array.from(f),
		markers: p,
		qtype: lt[s] ?? 0,
		seqLen: f.length,
		truncated: g
	};
}
//#endregion
//#region src/julia.ts
function J(e) {
	return {
		logits: e.logits.slice(),
		probabilities: e.probabilities.slice()
	};
}
function dt(e, t) {
	let n = new Float32Array(t), r = -Infinity;
	for (let n = 0; n < t; n += 1) r = Math.max(r, e[n]);
	let i = 0;
	for (let n = 0; n < t; n += 1) i += Math.exp(e[n] - r);
	for (let a = 0; a < t; a += 1) n[a] = Math.exp(e[a] - r) / i;
	return n;
}
var ft = 8, pt = class e {
	kh;
	weights;
	tokenizer;
	spec;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	batchFit = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	tokenizerBytes = 0;
	weightGpuBytes = 0;
	loadTiming = {};
	cache = new V(0);
	constructor(e, t, n, r, i) {
		this.kh = e, this.weights = t, this.tokenizer = n, this.spec = r, this.precision = i;
	}
	static async load(n, r) {
		let a = n.precision !== "f32", o = n.manifestUrl.slice(0, n.manifestUrl.lastIndexOf("/") + 1), c = performance.now(), l = r ? Promise.resolve(r) : s(n.manifestUrl), d = await t(a, n.limits !== "default"), f = await l, p = performance.now() - c, m = (async () => (await fetch(o + f.tokenizer)).arrayBuffer())(), h = await u(d.device, n.manifestUrl, f), g = h.manifest.tensors.find((e) => e.name.endsWith("wqkv.weight"))?.dtype ?? "f32";
		if (g === "f16" && !d.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n.precision && n.precision !== "auto" && n.precision !== g) throw Error(`precision ${n.precision} requested but manifest is ${g}`);
		let _ = g, v = performance.now(), y = await m, b = new ct(JSON.parse(new TextDecoder().decode(y))), x = performance.now() - v, S = f.encoder, C = new e(d, h, b, S, _);
		C.downloadBytes = h.downloadBytes + y.byteLength, C.tokenizerBytes = y.byteLength, C.weightGpuBytes = h.gpuBytes, C.loadTiming = {
			...h.timing,
			manifestMs: p,
			tokenizerMs: x,
			planMs: 0
		};
		let w = performance.now();
		return await i(d, () => {
			for (let e of n.buckets ?? [512]) {
				let t = typeof e == "number" ? e : e.length, n = C.planFor(t, 1);
				Te(n, d.device.limits, (e) => h.tensors.get(e)?.size), C.plans.set(t, new G(d.device, n, h.tensors));
			}
		}), C.cache = new V(n.cacheSize ?? 256), C.loadTiming.planMs = performance.now() - w, C;
	}
	planFor(e, t) {
		let { spec: n, head: r } = ue(this.spec);
		return ve(n, r, {
			length: e,
			batch: t,
			markers: this.spec.options,
			f16: this.precision === "f16"
		});
	}
	pickBucket(e) {
		let t = [...this.plans.keys()].sort((e, t) => e - t).find((t) => e <= t);
		if (t === void 0) throw new _(`seqLen ${e} exceeds largest bucket`);
		return this.plans.get(t);
	}
	embeddingRows(e, t) {
		let n = t.length, r = this.spec.hiddenSize;
		if (this.weights.embeddings instanceof Float32Array) {
			let t = new Float32Array(n * r), i = this.weights.embeddings;
			for (let n = 0; n < e.seqLen; n += 1) {
				let a = q(e.inputIds[n], r, i.length);
				t.set(i.subarray(a, a + r), n * r);
			}
			return t;
		}
		let i = new Uint16Array(n * r), a = this.weights.embeddings;
		for (let t = 0; t < e.seqLen; t += 1) {
			let n = q(e.inputIds[t], r, a.length);
			i.set(a.subarray(n, n + r), t * r);
		}
		return i;
	}
	packedMarkers(e) {
		let t = this.spec.options, n = new Uint32Array(3 * t);
		for (let r = 0; r < e.markers.length && r < t; r += 1) n[r] = e.markers[r];
		let r = new Float32Array(t);
		return r.fill(1, 0, Math.min(e.markers.length, t)), n.set(new Uint32Array(r.buffer), t), n;
	}
	maskOf(e, t) {
		let n = new Float32Array(t);
		return n.fill(1, 0, e.seqLen), n;
	}
	enqueue(e) {
		let t = this.queue.then(e);
		return this.queue = t.catch(() => {}), t;
	}
	async runPrepared(e, t = !1, n) {
		let i = n === void 0 ? this.pickBucket(e.seqLen) : this.plans.get(n);
		if (!i) throw Error(`bucket ${n} not loaded`);
		if (e.seqLen > i.length) throw new _(`seqLen ${e.seqLen} exceeds bucket ${i.length}`);
		t && i.assertCapturable();
		let a = U(e), o = t ? void 0 : this.cache.get(a);
		return o ? J(o) : this.enqueue(() => r(this.kh, () => {
			i.upload({
				embeddings: this.embeddingRows(e, i),
				mask: this.maskOf(e, i.length),
				packedMarkers: this.packedMarkers(e),
				qtype: e.qtype
			}), i.submit(t, {
				seqLen: e.seqLen,
				qtype: e.qtype
			});
		}, async () => {
			let n = await i.readLogits();
			K(n, "logits");
			let r = dt(n, Math.min(e.markers.length, this.spec.options));
			return t || this.cache.set(a, J({
				logits: n,
				probabilities: r
			})), {
				logits: n,
				probabilities: r,
				captureData: t ? await i.readCapture() : void 0
			};
		}));
	}
	batchPlan(e, t) {
		let n = `${e}:${t}`, r = this.batchFit.get(n);
		if (r === 1) return;
		let i = this.batchPlans.get(`${e}:${r ?? t}`);
		if (i) return i;
		let a = r === void 0 ? Ee((t) => this.planFor(e, t), this.kh.device.limits, t) : this.planFor(e, r);
		if (this.batchFit.set(n, a ? a.batch : 1), !a) return;
		if (this.batchPlans.size >= ft) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let o = new G(this.kh.device, a, this.weights.tensors);
		return this.batchPlans.set(`${e}:${a.batch}`, o), o;
	}
	padInput() {
		return {
			inputIds: /* @__PURE__ */ new Int32Array(),
			markers: [],
			qtype: 0,
			seqLen: 0
		};
	}
	batchEmbeddingRows(e, t) {
		let n = this.spec.hiddenSize, r = this.weights.embeddings, i = r instanceof Float32Array ? new Float32Array(t.length * t.batch * n) : new Uint16Array(t.length * t.batch * n);
		for (let [a, o] of e.entries()) {
			let e = a * t.length * n;
			for (let t = 0; t < o.seqLen; t += 1) {
				let a = q(o.inputIds[t], n, r.length);
				i.set(r.subarray(a, a + n), e + t * n);
			}
		}
		return i;
	}
	batchMask(e, t) {
		let n = new Float32Array(t.length * t.batch);
		for (let [r, i] of e.entries()) n.fill(1, r * t.length, r * t.length + i.seqLen);
		return n;
	}
	batchPackedMarkers(e) {
		let t = new Uint32Array(e.length * 3 * this.spec.options);
		for (let [n, r] of e.entries()) t.set(this.packedMarkers(r), n * 3 * this.spec.options);
		return t;
	}
	async runBatchChunk(e, t) {
		let n = Math.max(...e.map((e) => e.seqLen)), i = t === void 0 ? this.pickBucket(n) : this.plans.get(t);
		if (!i) throw Error(`bucket ${t} not loaded`);
		if (n > i.length) throw new _(`batch seqLen ${n} exceeds bucket ${i.length}`);
		let a = ye(e.length), o = Math.min(i.length, be(n));
		return this.enqueue(async () => {
			let t = [], n;
			for (; t.length < e.length;) {
				let s = await r(this.kh, () => {
					n ??= (a === 1 ? void 0 : this.batchPlan(o, a)) ?? i;
					let r = e.slice(t.length, t.length + n.batch);
					return this.submitPiece(r, n), {
						rows: r,
						plan: n
					};
				}, (e) => this.readPiece(e.rows, e.plan));
				t.push(...s);
			}
			return t;
		});
	}
	submitPiece(e, t) {
		let n = e.length === t.batch ? e : [...e, ...Array.from({ length: t.batch - e.length }, () => this.padInput())], r = t.batch === 1 ? Math.max(...e.map((e) => e.seqLen)) : t.length * t.batch;
		t.upload({
			embeddings: this.batchEmbeddingRows(n, t),
			mask: this.batchMask(n, t),
			packedMarkers: this.batchPackedMarkers(n),
			qtype: 0
		}), t.submit(!1, {
			seqLen: r,
			qtype: n.map((e) => e.qtype)
		});
	}
	async readPiece(e, t) {
		let n = await t.readLogits(), r = this.spec.options;
		return e.map((e, t) => {
			let i = n.slice(t * r, (t + 1) * r);
			K(i, "logits");
			let a = {
				logits: i,
				probabilities: dt(i, Math.min(e.markers.length, r))
			};
			return this.cache.set(U(e), J(a)), a;
		});
	}
	async runPreparedBatch(e, t) {
		let n = Array(e.length), r = [], i = [];
		for (let [t, a] of e.entries()) {
			let e = this.cache.get(U(a));
			e ? n[t] = J(e) : (r.push(a), i.push(t));
		}
		let { unique: a, slot: o } = W(r, U), s = xe(a, o, (e) => e.seqLen), c = [];
		for (let e = 0; e < s.unique.length; e += B) {
			let n = await this.runBatchChunk(s.unique.slice(e, e + B), t);
			c.push(...n);
		}
		let l = /* @__PURE__ */ new Set();
		for (let [e] of r.entries()) {
			let t = c[s.slot[e]];
			n[i[e]] = l.has(s.slot[e]) ? J(t) : t, l.add(s.slot[e]);
		}
		return n;
	}
	prepare(e, t = 1024, n = 512, r = !1) {
		return ut(this.tokenizer, e, t, n, r);
	}
	async decide(e) {
		let t = performance.now(), n = ut(this.tokenizer, e, this.maxBucket(), 256), r = performance.now() - t, i = performance.now(), { logits: a, probabilities: o } = await this.runPrepared({
			inputIds: n.inputIds,
			markers: n.markers,
			qtype: n.qtype,
			seqLen: n.seqLen
		}), s = performance.now() - i, c = 0;
		for (let e = 1; e < n.markers.length; e += 1) a[e] > a[c] && (c = e);
		return {
			logits: a,
			probabilities: o,
			choice: c,
			timings: {
				tokenizeMs: r,
				gpuMs: s,
				totalMs: performance.now() - t
			}
		};
	}
	async decideBatch(e) {
		let t = performance.now(), n = e.map((e) => {
			let t = ut(this.tokenizer, e, this.maxBucket(), 256);
			return {
				inputIds: t.inputIds,
				markers: t.markers,
				qtype: t.qtype,
				seqLen: t.seqLen
			};
		}), r = performance.now() - t, i = performance.now(), a = await this.runPreparedBatch(n), o = {
			tokenizeMs: r,
			gpuMs: performance.now() - i,
			totalMs: performance.now() - t
		};
		return e.map((e, t) => {
			let { logits: r, probabilities: i } = a[t], s = 0;
			for (let e = 1; e < n[t].markers.length; e += 1) r[e] > r[s] && (s = e);
			return {
				logits: r,
				probabilities: i,
				choice: s,
				timings: o
			};
		});
	}
	maxBucket() {
		return Math.max(...this.plans.keys());
	}
	async profile(e, t = {}) {
		let n = await this.profileDetailed(e, t);
		return n && n.times;
	}
	async profileDetailed(e, t = {}) {
		let n = this.pickBucket(e.seqLen);
		return this.enqueue(() => n.profileForward({
			embeddings: this.embeddingRows(e, n),
			mask: this.maskOf(e, n.length),
			packedMarkers: this.packedMarkers(e),
			qtype: e.qtype
		}, t.granularity ?? "pass", e.seqLen));
	}
	liveGpuBytes() {
		let e = this.weightGpuBytes;
		for (let t of this.plans.values()) e += t.gpuBytes;
		for (let t of this.batchPlans.values()) e += t.gpuBytes;
		return e;
	}
	info() {
		return {
			precision: this.precision,
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			gpuBytes: this.liveGpuBytes(),
			downloadBytes: this.downloadBytes,
			tokenizerBytes: this.tokenizerBytes,
			weightBytes: this.downloadBytes - this.tokenizerBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.kh.device.destroy();
	}
}, mt = class e {
	worker;
	seq = 0;
	pending = /* @__PURE__ */ new Map();
	infoData = {};
	cache = new V(0);
	constructor() {}
	static async load(t) {
		let n = new e();
		return n.cache = new V(t.cacheSize ?? 256), n.worker = new Worker(new URL(
			/* @vite-ignore */
			"" + new URL("assets/wasm-worker-DXLL-ExG.js", import.meta.url).href,
			"" + import.meta.url
		), { type: "module" }), n.worker.onerror = (e) => {
			for (let t of n.pending.values()) t.reject(/* @__PURE__ */ Error(`wasm worker failed: ${e.message ?? "script error"}`));
			n.pending.clear();
		}, n.worker.onmessage = (e) => {
			let t = e.data, r = n.pending.get(t.id);
			r && (n.pending.delete(t.id), t.type === "error" ? r.reject(Error(t.message)) : r.resolve(t.result ?? t.info));
		}, n.infoData = await n.call("load", { options: t }), n;
	}
	call(e, t) {
		let n = this.seq += 1;
		return new Promise((r, i) => {
			this.pending.set(n, {
				resolve: r,
				reject: i
			}), this.worker.postMessage({
				type: e,
				id: n,
				...t
			});
		});
	}
	async classify(e, t) {
		return await this.call("classify", {
			text: e,
			tasks: t
		});
	}
	async classifyBatch(e) {
		let t = (e) => JSON.stringify([e.text, e.tasks]), n = Array(e.length), r = [], i = [];
		for (let [a, o] of e.entries()) {
			let e = this.cache.get(t(o));
			e ? n[a] = e : (r.push(o), i.push(a));
		}
		let { unique: a, slot: o } = W(r, t), s = [];
		for (let e of a) {
			let n = await this.classify(e.text, e.tasks);
			s.push(n), this.cache.set(t(e), n);
		}
		for (let [e] of r.entries()) n[i[e]] = s[o[e]];
		return n;
	}
	async runPrepared(e, t = !1, n) {
		return await this.call("runPrepared", {
			input: e,
			capture: t,
			bucket: n
		});
	}
	async runPreparedBatch(e, t) {
		let { unique: n, slot: r } = W(e, H), i = [];
		for (let e of n) i.push(await this.runPrepared(e, !1, t));
		return e.map((e, t) => i[r[t]]);
	}
	info() {
		return {
			...this.infoData,
			worker: !0
		};
	}
	dispose() {
		this.worker.terminate();
		for (let e of this.pending.values()) e.reject(/* @__PURE__ */ Error("worker disposed"));
		this.pending.clear();
	}
};
//#endregion
//#region src/tokenizer/hf/normalized.ts
function ht(e, t = 0) {
	let n = [], r = [], i = [], a = t;
	for (let t of e) n.push(t), r.push(a), a += t.length, i.push(a);
	return {
		chars: n,
		starts: r,
		ends: i,
		origin: t
	};
}
function Y(e) {
	let t = e.codePointAt(0);
	return t < 128 ? 1 : t < 2048 ? 2 : t < 65536 ? 3 : 4;
}
function gt(e, t, n = 0) {
	let r = [], i = [], a = [], o = e.chars.length - 1, s = e.origin ?? 0, c = n;
	for (let [n, l] of t) {
		let t, u;
		if (l > 0) c < 1 ? (t = s, u = s) : (t = e.starts[c - 1], u = e.ends[c - 1]);
		else {
			let n = Math.min(c, o);
			t = e.starts[n], u = e.ends[n], c += 1, l < 0 && (c += -l);
		}
		r.push(n), i.push(t), a.push(u);
	}
	return {
		chars: r,
		starts: i,
		ends: a,
		origin: e.origin
	};
}
function X(e, t, n) {
	return {
		chars: e.chars.slice(t, n),
		starts: e.starts.slice(t, n),
		ends: e.ends.slice(t, n),
		origin: t < e.starts.length ? e.starts[t] : e.origin
	};
}
function _t(e, t) {
	if (e.length === 0) return [[
		0,
		0,
		!1
	]];
	let n = [], r = 0;
	for (let i = 0; i < e.length; i += 1) t(e[i]) && (r < i && n.push([
		r,
		i,
		!1
	]), n.push([
		i,
		i + 1,
		!0
	]), r = i + 1);
	return e.length > r && n.push([
		r,
		e.length,
		!1
	]), n;
}
function vt(e) {
	let t = [], n = 0;
	for (let r = 0; r < e.length; r += 1) {
		for (; t.length < n;) t.push(r - 1);
		t.push(r), n += e[r].length;
	}
	for (; t.length <= n;) t.push(e.length);
	return t[n] = e.length, t;
}
function yt(e, t) {
	if (e.length === 0) return [[
		0,
		0,
		!1
	]];
	let n = e.join(""), r = vt(e), i = [], a = 0;
	t.lastIndex = 0;
	for (let e = t.exec(n); e !== null; e = t.exec(n)) {
		let n = r[e.index], o = r[e.index + e[0].length];
		e[0].length === 0 && (t.lastIndex += 1), a !== n && i.push([
			a,
			n,
			!1
		]), i.push([
			n,
			o,
			!0
		]), a = o;
	}
	return a !== e.length && i.push([
		a,
		e.length,
		!1
	]), i;
}
function bt(e, t, n) {
	let r;
	if (n === "removed") r = t;
	else if (n === "isolated") r = t.map(([e, t]) => [
		e,
		t,
		!1
	]);
	else {
		let e = !1, n = [];
		for (let r = t.length - 1; r >= 0; --r) {
			let [i, a, o] = t[r];
			o && !e && n.length ? n[n.length - 1][0] = i : n.push([
				i,
				a,
				!1
			]), e = o;
		}
		n.reverse(), r = n;
	}
	let i = [];
	for (let [t, n, a] of r) a || i.push(X(e, t, n));
	return i;
}
function xt(e, t, n) {
	let r = [], i = [], a = [], o = (t, n) => {
		for (let o = t; o < n; o += 1) r.push(e.chars[o]), i.push(e.starts[o]), a.push(e.ends[o]);
	}, s = [...n];
	for (let [n, c, l] of t) {
		if (!l) {
			o(n, c);
			continue;
		}
		let t = c >= 1 ? e.starts[c - 1] : e.origin ?? 0, u = c >= 1 ? e.ends[c - 1] : e.origin ?? 0;
		for (let e of s) r.push(e), i.push(t), a.push(u);
	}
	return {
		chars: r,
		starts: i,
		ends: a,
		origin: e.origin
	};
}
function St(e, t) {
	if (e.chars.length === 0) return e;
	let n = [...t];
	return {
		chars: [...n, ...e.chars],
		starts: [...n.map(() => e.starts[0]), ...e.starts],
		ends: [...n.map(() => e.ends[0]), ...e.ends],
		origin: e.origin
	};
}
function Ct(e) {
	return e.chars.join("");
}
//#endregion
//#region src/tokenizer/hf/bpe.ts
var wt = class e {
	vocab;
	vocabR = [];
	base;
	merges = /* @__PURE__ */ new Map();
	opt;
	constructor(e, t, n) {
		this.vocab = e, this.opt = n;
		let r = 0;
		for (let [t, n] of e) {
			if (!Number.isInteger(n) || n < 0) throw Error(`BPE: vocab id ${n} of ${t} is not a non-negative integer`);
			this.vocabR[n] = t, r = Math.max(r, n);
		}
		if (this.base = r + 1, this.base * this.base > 2 ** 53 - 1) throw Error(`BPE: largest vocab id ${r} is too large for a collision-free pair key`);
		let i = n.continuingSubwordPrefix === null ? 0 : n.continuingSubwordPrefix.length;
		if (t.forEach(([t, n], r) => {
			let a = e.get(t), o = e.get(n);
			if (a === void 0) throw Error(`BPE: merge token ${t} is not in the vocab`);
			if (o === void 0) throw Error(`BPE: merge token ${n} is not in the vocab`);
			let s = e.get(t + n.slice(i));
			if (s === void 0) throw Error(`BPE: merged token ${t + n.slice(i)} is not in the vocab`);
			this.merges.set(a * this.base + o, [r, s]);
		}), n.unkToken !== null && !e.has(n.unkToken)) throw Error(`BPE: unk token ${n.unkToken} is not in the vocab`);
	}
	static fromJson(t) {
		if (t.dropout !== null && t.dropout !== void 0) throw Error("BPE: dropout is not supported");
		let n = new Map(Object.entries(t.vocab)), r = t.merges.map((e) => {
			if (typeof e != "string") return [e[0], e[1]];
			let t = e.split(" ");
			if (t.length !== 2) throw Error(`BPE: invalid merge ${e}`);
			return [t[0], t[1]];
		});
		return new e(n, r, {
			unkToken: t.unk_token ?? null,
			continuingSubwordPrefix: t.continuing_subword_prefix ?? null,
			endOfWordSuffix: t.end_of_word_suffix ?? null,
			fuseUnk: t.fuse_unk === !0,
			byteFallback: t.byte_fallback === !0,
			ignoreMerges: t.ignore_merges === !0
		});
	}
	tokenize(e) {
		if (e.length === 0) return [];
		let t = e.join("");
		if (this.opt.ignoreMerges) {
			let n = this.vocab.get(t);
			if (n !== void 0) return [{
				id: n,
				start: 0,
				end: e.length,
				value: t
			}];
		}
		let n = [];
		e.forEach((e, t) => {
			for (let r = Y(e); r > 0; --r) n.push(t);
		});
		let r = [], i = 0;
		for (let t of this.mergeWord(e)) {
			let e = n[Math.min(i, n.length - 1)], a = n[Math.min(i + t.len, n.length) - 1] + 1;
			r.push({
				id: t.id,
				start: e,
				end: Math.max(a, e + 1),
				value: this.vocabR[t.id]
			}), i += t.len;
		}
		return r;
	}
	mergeWord(e) {
		let t = this.opt, n = [], r = null;
		for (let i = 0; i < e.length; i += 1) {
			let a = e[i];
			i > 0 && t.continuingSubwordPrefix !== null && (a = t.continuingSubwordPrefix + a), i === e.length - 1 && t.endOfWordSuffix !== null && (a += t.endOfWordSuffix);
			let o = this.vocab.get(a);
			if (o !== void 0) {
				r &&= (n.push(r), null), n.push({
					id: o,
					len: Y(e[i])
				});
				continue;
			}
			if (t.byteFallback) {
				let e = [], t = !0;
				for (let n of new TextEncoder().encode(a)) {
					let r = this.vocab.get(`<0x${n.toString(16).toUpperCase().padStart(2, "0")}>`);
					if (r === void 0) {
						t = !1;
						break;
					}
					e.push(r);
				}
				if (t) {
					for (let t of e) n.push({
						id: t,
						len: 1
					});
					continue;
				}
			}
			if (t.unkToken !== null) {
				let a = this.vocab.get(t.unkToken);
				r && t.fuseUnk ? r = {
					id: r.id,
					len: r.len + Y(e[i])
				} : (r && n.push(r), r = {
					id: a,
					len: Y(e[i])
				});
			}
		}
		return r && n.push(r), this.mergeAll(n);
	}
	mergeAll(e) {
		let t = e, n = this.base;
		for (;;) {
			let e = Infinity, r = -1, i = 0;
			for (let a = 0; a + 1 < t.length; a += 1) {
				let o = this.merges.get(t[a].id * n + t[a + 1].id);
				o !== void 0 && o[0] < e && (e = o[0], r = a, i = o[1]);
			}
			if (r < 0) return t;
			t = [
				...t.slice(0, r),
				{
					id: i,
					len: t[r].len + t[r + 1].len
				},
				...t.slice(r + 2)
			];
		}
	}
}, Tt = /^[\p{Cc}\p{Cf}\p{Cs}\p{Co}]$/u, Et = /^\p{White_Space}$/u, Dt = /^\p{Mn}$/u;
function Z(e) {
	let t = e.charCodeAt(0);
	return t < 128 ? t === 32 || t >= 9 && t <= 13 : Et.test(e);
}
function Ot(e) {
	return e === 2192 || e === 2193 || e === 2274 || e === 69837 || e >= 78896 && e <= 78911;
}
function kt(e) {
	return e === "	" || e === "\n" || e === "\r" ? !1 : Tt.test(e) && !Ot(e.codePointAt(0));
}
function At(e) {
	return e >= 19968 && e <= 40959 || e >= 13312 && e <= 19903 || e >= 131072 && e <= 173791 || e >= 173824 && e <= 177983 || e >= 177984 && e <= 178207 || e >= 178464 && e <= 183983 || e >= 63744 && e <= 64255 || e >= 194560 && e <= 195103;
}
function jt(e) {
	return {
		cleanText: e.clean_text !== !1,
		handleChineseChars: e.handle_chinese_chars !== !1,
		stripAccents: e.strip_accents ?? null,
		lowercase: e.lowercase !== !1
	};
}
function Mt(e, t) {
	let n = [], r = [], i = [];
	for (let a = 0; a < e.chars.length; a += 1) {
		let o = e.chars[a];
		(!t.cleanText || o !== "\0" && o !== "�" && !kt(o)) && (n.push(t.cleanText && Z(o) ? " " : o), r.push(e.starts[a]), i.push(e.ends[a]));
	}
	if (t.handleChineseChars) {
		let e = [], t = [], a = [];
		for (let o = 0; o < n.length; o += 1) {
			let s = n[o];
			At(s.codePointAt(0)) ? (e.push(" ", s, " "), t.push(r[o], r[o], r[o]), a.push(i[o], i[o], i[o])) : (e.push(s), t.push(r[o]), a.push(i[o]));
		}
		n = e, r = t, i = a;
	}
	if (t.stripAccents ?? t.lowercase) {
		let e = [], t = [], a = [];
		for (let o = 0; o < n.length; o += 1) for (let s of n[o].normalize("NFD")) Dt.test(s) || (e.push(s), t.push(r[o]), a.push(i[o]));
		n = e, r = t, i = a;
	}
	if (t.lowercase) {
		let e = [], t = [], a = [];
		for (let o = 0; o < n.length; o += 1) for (let s of n[o].toLowerCase()) e.push(s), t.push(r[o]), a.push(i[o]);
		n = e, r = t, i = a;
	}
	return {
		chars: n,
		starts: r,
		ends: i,
		origin: e.origin
	};
}
var Nt = /* @__PURE__ */ new Map(), Pt = /* @__PURE__ */ new Map();
function Ft(e) {
	let t = Nt.get(e);
	return t === void 0 && (t = e === "ͅ" || `x\u0345${e}`.normalize("NFD") !== `x\u0345${e}`, Nt.set(e, t)), t;
}
function It(e, t) {
	let n = e + t, r = Pt.get(n);
	return r === void 0 && (r = `x${t}${e}`.normalize("NFD") !== `x${t}${e}`, Pt.set(n, r)), r;
}
var Lt = /* @__PURE__ */ new Map();
function Rt(e, t) {
	let n = e + "\0" + t, r = Lt.get(n);
	if (r === void 0) {
		let i = (e + t).normalize("NFC");
		r = [...i].length === 1 ? i : null, Lt.set(n, r);
	}
	return r;
}
function zt(e, t) {
	let n = t === "NFC" ? "NFD" : "NFKD", r = [];
	for (let t of e.chars) [...t.normalize(n)].forEach((e, t) => r.push([e, t === 0 ? 0 : 1]));
	for (let e = 0; e < r.length;) {
		if (!Ft(r[e][0])) {
			e += 1;
			continue;
		}
		let t = e;
		for (; t < r.length && Ft(r[t][0]);) t += 1;
		for (let n = e + 1; n < t; n += 1) {
			let t = r[n], i = n - 1;
			for (; i >= e && It(t[0], r[i][0]);) r[i + 1] = r[i], --i;
			r[i + 1] = t;
		}
		e = t;
	}
	let i = [], a = -1, o = null, s = !1;
	for (let [e, t] of r) {
		let n = Ft(e);
		if (a >= 0) {
			let r = Rt(i[a][0], e), c = s || o !== null && n && It(o, e);
			if (r !== null && c) {
				i[a] = [r, i[a][1] + t - 1];
				continue;
			}
		}
		n || (a = i.length), o = e, s = !n, i.push([e, t]), i.length === 1 && n && (o = null, a = -1);
	}
	return gt(e, i);
}
function Bt(e, t, n) {
	let r = e.chars.length, i = 0;
	if (t) for (; i < r && Z(e.chars[i]);) i += 1;
	let a = 0;
	if (n) for (; a < r && Z(e.chars[r - 1 - a]);) a += 1;
	if (i === 0 && a === 0) return e;
	let o = Math.min(i, r), s = Math.max(o, r - a);
	return {
		chars: e.chars.slice(o, s),
		starts: e.starts.slice(o, s),
		ends: e.ends.slice(o, s),
		origin: e.origin
	};
}
var Vt = "\\p{L}\\p{M}\\p{Nd}\\p{Pc}", Ht = "\\p{White_Space}";
function Ut(e) {
	let t = "", n = !1;
	for (let r = 0; r < e.length; r += 1) {
		let i = e[r];
		if (i === "\\") {
			let a = e[r + 1];
			r += 1;
			let o = {
				w: Vt,
				s: Ht
			}, s = {
				W: Vt,
				S: Ht
			};
			if (a in o) t += n ? o[a] : `[${o[a]}]`;
			else if (a in s) {
				if (n) throw Error(`Replace: unsupported \\${a} inside a character class in ${e}`);
				t += `[^${s[a]}]`;
			} else if (a === "d" || a === "D") throw Error(`Replace: unsupported \\${a} in ${e}`);
			else if ((a === "b" || a === "B") && !n) throw Error(`Replace: unsupported \\${a} outside a character class in ${e}`);
			else t += i + a;
		} else i === "[" ? n = !0 : i === "]" && (n = !1), t += i;
	}
	return t;
}
function Wt(e) {
	return e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
var Gt = class {
	trie;
	blob;
	cache = /* @__PURE__ */ new Map();
	encoder = new TextEncoder();
	decoder = new TextDecoder();
	constructor(e) {
		let t = Uint8Array.from(atob(e), (e) => e.charCodeAt(0)), n = new DataView(t.buffer, t.byteOffset, t.byteLength), r = n.getUint32(0, !0);
		this.trie = new Uint32Array(r / 4);
		for (let e = 0; e < this.trie.length; e += 1) this.trie[e] = n.getUint32(4 + 4 * e, !0);
		this.blob = t.subarray(4 + r);
	}
	transform(e) {
		let t = this.cache.get(e);
		if (t !== void 0) return t;
		let n = this.firstPrefix(this.encoder.encode(e)), r = null;
		if (n !== null) {
			let e = n;
			for (; e < this.blob.length && this.blob[e] !== 0;) e += 1;
			r = this.decoder.decode(this.blob.subarray(n, e));
		}
		return this.cache.set(e, r), r;
	}
	firstPrefix(e) {
		let t = this.trie, n = (e) => e >>> 10 << ((e & 512) >> 6), r = 0, i = t[r];
		r ^= n(i);
		for (let a of e) {
			if (a === 0) break;
			if (r ^= a, i = t[r], i === void 0 || (i & 2147483903) >>> 0 !== a) return null;
			if (r ^= n(i), i >>> 8 & 1) return t[r] & 2147483647;
		}
		return null;
	}
	normalize(e, t) {
		let n = [], r = !1, i = (e, t) => {
			let r = [...e].length, i = [...t], a = i.length - r;
			for (let e of i) n.push([e, 0]);
			if (a > 0) for (let e = 0; e < a; e += 1) n[n.length - 1 - e][1] = 1;
			else a < 0 && n.length && (n[n.length - 1][1] += a);
		}, a = e.chars.join("");
		for (let { segment: e } of t.segment(a)) {
			if (Kt(e) < 6) {
				let t = this.transform(e);
				if (t !== null) {
					r = !0, i(e, t);
					continue;
				}
			}
			for (let t of e) {
				let e = this.transform(t);
				e === null ? n.push([t, 0]) : (r = !0, i(t, e));
			}
		}
		return r ? gt(e, n) : e;
	}
};
function Kt(e) {
	let t = 0;
	for (let n of e) t += Y(n);
	return t;
}
function qt(e) {
	if (e == null) return null;
	let t = String(e.type);
	switch (t) {
		case "Sequence": {
			let t = e.normalizers.map((e) => qt(e));
			return (e) => t.reduce((e, t) => t(e), e);
		}
		case "BertNormalizer": {
			let t = jt(e);
			return (e) => Mt(e, t);
		}
		case "NFC":
		case "NFKC": return (e) => zt(e, t);
		case "Strip": {
			let t = e.strip_left !== !1, n = e.strip_right !== !1;
			return (e) => Bt(e, t, n);
		}
		case "Replace": {
			let t = e.pattern, n = t.String === void 0 ? Ut(t.Regex) : Wt(t.String);
			if (n === "") throw Error("Replace: empty pattern");
			let r = new RegExp(n, "gu"), i = e.content;
			return (e) => xt(e, yt(e.chars, r), i);
		}
		case "Precompiled": {
			let t = new Gt(e.precompiled_charsmap), n = new Intl.Segmenter("en", { granularity: "grapheme" });
			return (e) => t.normalize(e, n);
		}
		default: throw Error(`tokenizer.json: unsupported normalizer type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/postprocessors.ts
var Jt = class e {
	single;
	pair;
	constructor(e, t) {
		this.single = e, this.pair = t;
	}
	static fromJson(t) {
		let n = t.special_tokens ?? {}, r = (e) => e.map((e) => {
			if (e.Sequence) {
				let t = e.Sequence.id;
				if (t !== "A" && t !== "B") throw Error(`TemplateProcessing: sequence id ${String(t)}`);
				return {
					kind: "sequence",
					which: t === "A" ? 0 : 1,
					typeId: e.Sequence.type_id
				};
			}
			if (e.SpecialToken) {
				let t = n[e.SpecialToken.id];
				if (!t) throw Error(`TemplateProcessing: unknown special token ${String(e.SpecialToken.id)}`);
				return {
					kind: "special",
					ids: t.ids,
					typeId: e.SpecialToken.type_id
				};
			}
			throw Error(`TemplateProcessing: template item ${Object.keys(e).join(",")}`);
		});
		return new e(r(t.single), r(t.pair));
	}
	addedTokens(e) {
		let t = 0;
		for (let n of e ? this.pair : this.single) n.kind === "special" && (t += n.ids.length);
		return t;
	}
	apply(e, t) {
		let n = t ? this.pair : this.single, r = {
			ids: [],
			typeIds: [],
			attentionMask: [],
			offsets: [],
			wordIds: [],
			sequenceIds: [],
			specialTokensMask: []
		};
		for (let i of n) {
			if (i.kind === "special") {
				for (let e of i.ids) r.ids.push(e), r.typeIds.push(i.typeId), r.attentionMask.push(1), r.offsets.push([0, 0]), r.wordIds.push(-1), r.sequenceIds.push(-1), r.specialTokensMask.push(1);
				continue;
			}
			let n = i.which === 0 ? e : t;
			if (!n) throw Error("TemplateProcessing: template needs a second sequence");
			for (let e = 0; e < n.ids.length; e += 1) r.ids.push(n.ids[e]), r.typeIds.push(i.typeId), r.attentionMask.push(1), r.offsets.push([n.offsets[2 * e], n.offsets[2 * e + 1]]), r.wordIds.push(n.wordIds[e]), r.sequenceIds.push(i.which), r.specialTokensMask.push(0);
		}
		return r;
	}
}, Yt = "Ġ";
function Xt(e, t) {
	let n = e.offsets.slice();
	for (let r = 0; r < e.ids.length; r += 1) {
		let i = [...e.tokens[r]], a = 0;
		for (; a < i.length && (i[a] === Yt || Z(i[a]));) a += 1;
		let o = 0;
		for (; o < i.length && (i[i.length - 1 - o] === Yt || Z(i[i.length - 1 - o]));) o += 1;
		if (a === 0 && o === 0) continue;
		let s = n[2 * r], c = n[2 * r + 1];
		a > 0 && ((r === 0 || s === 0) && t && a === 1 && (a = 0), s = Math.min(s + a, c)), o > 0 && c >= o && (c = Math.max(c - o, s)), n[2 * r] = s, n[2 * r + 1] = c;
	}
	return n;
}
function Zt(e, t) {
	e.ids.push(t), e.typeIds.push(0), e.attentionMask.push(1), e.offsets.push([0, 0]), e.wordIds.push(-1), e.sequenceIds.push(-1), e.specialTokensMask.push(1);
}
function Q(e, t, n, r, i) {
	for (let a = 0; a < t.ids.length; a += 1) e.ids.push(t.ids[a]), e.typeIds.push(i), e.attentionMask.push(1), e.offsets.push([n[2 * a], n[2 * a + 1]]), e.wordIds.push(t.wordIds[a]), e.sequenceIds.push(r), e.specialTokensMask.push(0);
}
function Qt() {
	return {
		ids: [],
		typeIds: [],
		attentionMask: [],
		offsets: [],
		wordIds: [],
		sequenceIds: [],
		specialTokensMask: []
	};
}
var $t = class e {
	sep;
	cls;
	trim;
	addPrefixSpace;
	constructor(e, t, n, r) {
		this.sep = e, this.cls = t, this.trim = n, this.addPrefixSpace = r;
	}
	static fromJson(t) {
		return new e(t.sep[1], t.cls[1], t.trim_offsets !== !1, t.add_prefix_space !== !1);
	}
	addedTokens(e) {
		return e ? 4 : 2;
	}
	apply(e, t) {
		let n = Qt();
		return Zt(n, this.cls), Q(n, e, this.trim ? Xt(e, this.addPrefixSpace) : e.offsets, 0, 0), Zt(n, this.sep), t && (Zt(n, this.sep), Q(n, t, this.trim ? Xt(t, this.addPrefixSpace) : t.offsets, 1, 0), Zt(n, this.sep)), n;
	}
}, en = class e {
	trim;
	addPrefixSpace;
	constructor(e, t) {
		this.trim = e, this.addPrefixSpace = t;
	}
	static fromJson(t) {
		return new e(t.trim_offsets !== !1, t.add_prefix_space !== !1);
	}
	addedTokens() {
		return 0;
	}
	apply(e, t) {
		let n = Qt();
		return Q(n, e, this.trim ? Xt(e, this.addPrefixSpace) : e.offsets, 0, 0), t && Q(n, t, this.trim ? Xt(t, this.addPrefixSpace) : t.offsets, 1, 1), n;
	}
};
function tn(e) {
	let t = String(e?.type);
	switch (t) {
		case "TemplateProcessing": return Jt.fromJson(e);
		case "RobertaProcessing": return $t.fromJson(e);
		case "ByteLevel": return en.fromJson(e);
		default: throw Error(`tokenizer.json: unsupported post_processor type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/pretokenizers.ts
var nn = /^\p{P}$/u, rn = /^[!-/:-@[-`{-~]$/;
function an(e) {
	return e.charCodeAt(0) < 128 ? rn.test(e) : nn.test(e);
}
function on(e) {
	let t = [], n = -1;
	for (let r = 0; r < e.chars.length; r += 1) {
		let i = e.chars[r];
		Z(i) ? (n >= 0 && t.push({
			start: n,
			end: r
		}), n = -1) : an(i) ? (n >= 0 && t.push({
			start: n,
			end: r
		}), t.push({
			start: r,
			end: r + 1
		}), n = -1) : n < 0 && (n = r);
	}
	return n >= 0 && t.push({
		start: n,
		end: e.chars.length
	}), t;
}
function sn(e) {
	return on(e).map((t) => X(e, t.start, t.end));
}
var cn = /* @__PURE__ */ RegExp("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+|\\p{White_Space}+(?!\\P{White_Space})|\\p{White_Space}+", "gu"), ln = (() => {
	let e = [];
	for (let t = 33; t <= 126; t += 1) e.push(t);
	for (let t = 161; t <= 172; t += 1) e.push(t);
	for (let t = 174; t <= 255; t += 1) e.push(t);
	let t = Array(256);
	for (let n of e) t[n] = String.fromCharCode(n);
	let n = 0;
	for (let e = 0; e < 256; e += 1) t[e] === void 0 && (t[e] = String.fromCharCode(256 + n), n += 1);
	return t;
})();
ln[32];
var un = new TextEncoder();
function dn(e) {
	let t = [], n = [], r = [];
	for (let i = 0; i < e.chars.length; i += 1) {
		let a = Y(e.chars[i]) === 1 ? [e.chars[i].charCodeAt(0)] : un.encode(e.chars[i]);
		for (let o of a) t.push(ln[o]), n.push(e.starts[i]), r.push(e.ends[i]);
	}
	return {
		chars: t,
		starts: n,
		ends: r,
		origin: e.origin
	};
}
function fn(e, t) {
	return (n) => {
		let r = n;
		return e && r.chars[0] !== " " && (r = St(r, " ")), (t ? bt(r, yt(r.chars, cn), "isolated") : [r]).map(dn);
	};
}
function pn(e, t, n) {
	let r = (e) => e === " ";
	return (i) => {
		let a = xt(i, _t(i.chars, r), e);
		return (t === "always" && a.chars[0] !== e || t === "first" && a.chars[0] !== e && (a.origin ?? 0) === 0) && (a = St(a, e)), n ? bt(a, _t(a.chars, (t) => t === e), "merged_with_next") : [a];
	};
}
function mn(e) {
	return bt(e, _t(e.chars, Z), "removed");
}
function hn(e) {
	if (e == null) return null;
	let t = String(e.type);
	switch (t) {
		case "Sequence": {
			let t = e.pretokenizers.map((e) => hn(e));
			return (e) => t.reduce((e, t) => e.flatMap((e) => t(e)).filter((e) => e.chars.length > 0), [e]);
		}
		case "BertPreTokenizer": return sn;
		case "WhitespaceSplit": return mn;
		case "ByteLevel": return fn(e.add_prefix_space !== !1, e.use_regex !== !1);
		case "Metaspace": {
			let t = e.replacement;
			if (e.add_prefix_space === !1 && e.prepend_scheme !== void 0 && e.prepend_scheme !== "never") throw Error("Metaspace: add_prefix_space does not match declared prepend_scheme");
			return pn(t, e.add_prefix_space === !1 ? "never" : e.prepend_scheme ?? "always", e.split !== !1);
		}
		default: throw Error(`tokenizer.json: unsupported pre_tokenizer type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/truncation.ts
var gn = "Truncation error: Sequence to truncate too short to respect the provided max_length";
function _n(e, t, n, r) {
	let i = t ?? 0, a = e + i;
	if (n <= 0) return [0, 0];
	if (a <= n) return [e, i];
	let o = a - n;
	if (r === "longest_first") {
		let t = Math.abs(e - i), n = Math.min(t, o), r = o - n, a = e <= i, s = Math.ceil(r / 2), c = n + Math.floor(r / 2);
		return [e - (a ? s : c), i - (a ? c : s)];
	}
	if (r === "only_first") {
		if (e <= o) throw Error(gn);
		return [e - o, i];
	}
	if (t === null) throw Error("Truncation error: Second sequence not provided");
	if (t <= o) throw Error(gn);
	return [e, t - o];
}
//#endregion
//#region src/tokenizer/hf/unigram.ts
var vn = 10, yn = class e {
	scores = /* @__PURE__ */ new Map();
	ids = /* @__PURE__ */ new Map();
	minScore = Infinity;
	maxChars = 0;
	unkId;
	byteFallback;
	constructor(e, t, n) {
		if (this.unkId = t, this.byteFallback = n, t !== null && (e.length === 0 || t >= e.length)) throw Error("Unigram: unk_id is not in the vocab");
		e.forEach(([e, t], n) => {
			this.ids.set(e, n), this.scores.set(e, t), t < this.minScore && (this.minScore = t), this.maxChars = Math.max(this.maxChars, [...e].length);
		}), this.byId = e.map((e) => e[1]);
	}
	byId;
	static fromJson(t) {
		return new e(t.vocab, t.unk_id ?? null, t.byte_fallback === !0);
	}
	tokenize(e) {
		if (e.length === 0) return [];
		let t = e.length, n = this.minScore - vn, r = Array(t + 1).fill(0), i = Array(t + 1).fill(-1), a = Array(t + 1).fill(0);
		for (let o = 0; o < t; o += 1) {
			let s = r[o], c = !1, l = "";
			for (let n = 1; n <= this.maxChars && o + n <= t; n += 1) {
				l += e[o + n - 1];
				let t = this.ids.get(l);
				if (t === void 0) continue;
				let u = o + n, d = this.byId[t] + s;
				(i[u] < 0 || d > r[u]) && (r[u] = d, i[u] = o, a[u] = t), n === 1 && (c = !0);
			}
			if (!c) {
				if (this.unkId === null) throw Error("Unigram: the vocab has no unk id");
				let e = n + s;
				(i[o + 1] < 0 || e > r[o + 1]) && (r[o + 1] = e, i[o + 1] = o, a[o + 1] = this.unkId);
			}
		}
		let o = [], s = t, c = -1, l = -1;
		for (; s > 0;) {
			let e = i[s];
			this.unkId !== null && a[s] === this.unkId ? (c < 0 && (c = s), l = e) : (c >= 0 && (o.push([l, c]), c = -1), o.push([e, s])), s = e;
		}
		c >= 0 && o.push([l, c]), o.reverse();
		let u = [];
		for (let [t, n] of o) {
			let r = e.slice(t, n).join(""), i = this.ids.get(r);
			if (i !== void 0) {
				u.push({
					id: i,
					start: t,
					end: n
				});
				continue;
			}
			if (this.byteFallback) {
				let e = new TextEncoder().encode(r), i = [], a = !0;
				for (let t of e) {
					let e = this.ids.get(`<0x${t.toString(16).toUpperCase().padStart(2, "0")}>`);
					if (e === void 0) {
						a = !1;
						break;
					}
					i.push(e);
				}
				if (a) {
					for (let e of i) u.push({
						id: e,
						start: t,
						end: n
					});
					continue;
				}
			}
			if (this.unkId === null) throw Error("Unigram: the vocab has no unk id");
			u.push({
				id: this.unkId,
				start: t,
				end: n
			});
		}
		return u;
	}
}, bn = class e {
	vocab;
	unkToken;
	prefix;
	maxInputCharsPerWord;
	constructor(e, t, n, r) {
		if (this.vocab = e, this.unkToken = t, this.prefix = n, this.maxInputCharsPerWord = r, !e.has(t)) throw Error(`WordPiece: unk token ${t} is not in the vocab`);
	}
	static fromJson(t) {
		let n = new Map(Object.entries(t.vocab));
		return new e(n, t.unk_token ?? "[UNK]", t.continuing_subword_prefix ?? "##", t.max_input_chars_per_word ?? 100);
	}
	id(e) {
		return this.vocab.get(e);
	}
	tokenize(e) {
		let t = {
			id: this.vocab.get(this.unkToken),
			start: 0,
			end: e.length
		};
		if (e.length > this.maxInputCharsPerWord) return [t];
		let n = [], r = 0;
		for (; r < e.length;) {
			let i = e.length, a = null;
			for (; r < i;) {
				let t = (r > 0 ? this.prefix : "") + e.slice(r, i).join(""), n = this.vocab.get(t);
				if (n !== void 0) {
					a = {
						id: n,
						start: r,
						end: i
					};
					break;
				}
				--i;
			}
			if (!a) return [t];
			n.push(a), r = i;
		}
		return n;
	}
}, xn = /^[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]$/u;
function Sn(e) {
	return e == null ? null : String(e.type);
}
function Cn(e) {
	return e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function wn(e) {
	return new TextEncoder().encode(e).length;
}
function Tn(e) {
	let t = Sn(e);
	if (t === "WordPiece") return bn.fromJson(e);
	if (t === "BPE") return wt.fromJson(e);
	if (t === "Unigram") return yn.fromJson(e);
	throw Error(`tokenizer.json: unsupported model type ${t}`);
}
function En(e, t) {
	if (e.length === 0) return [{
		token: null,
		start: 0,
		end: 0
	}];
	let n = [], r = 0;
	if (t.regex) {
		t.regex.lastIndex = 0;
		for (let i = t.regex.exec(e); i !== null; i = t.regex.exec(e)) {
			if (i[0].length === 0) {
				t.regex.lastIndex += 1;
				continue;
			}
			let a = i.index, o = i.index + i[0].length, s = t.tokens[i.slice(1).findIndex((e) => e !== void 0)];
			if (s.singleWord) {
				let t = [...e.slice(0, a)].pop(), n = [...e.slice(o)][0], r = a === 0 || !xn.test(t);
				if (o !== e.length && xn.test(n) || !r) continue;
			}
			if (s.lstrip) {
				let t = a;
				for (; t > 0;) {
					let n = [...e.slice(Math.max(0, t - 2), t)].pop();
					if (!Z(n)) break;
					t -= n.length;
				}
				a = Math.max(t, r);
			}
			if (s.rstrip) {
				let t = o;
				for (; t < e.length;) {
					let n = String.fromCodePoint(e.codePointAt(t));
					if (!Z(n)) break;
					t += n.length;
				}
				o = t;
			}
			r < a && n.push({
				token: null,
				start: r,
				end: a
			}), n.push({
				token: s,
				start: a,
				end: o
			}), r = o;
		}
	}
	return r !== e.length && n.push({
		token: null,
		start: r,
		end: e.length
	}), n;
}
function Dn(e, t) {
	if (e.length === 0) return {
		tokens: e,
		regex: null
	};
	let n = e.map((e, t) => t).sort((e, n) => wn(t[n]) - wn(t[e]) || e - n);
	return {
		tokens: n.map((t) => e[t]),
		regex: new RegExp(n.map((e) => `(${Cn(t[e])})`).join("|"), "g")
	};
}
var On = class e {
	normalizer;
	preTokenizer;
	model;
	post;
	plain;
	normalizedSet;
	constructor(e, t, n, r, i, a) {
		this.normalizer = e, this.preTokenizer = t, this.model = n, this.post = r, this.plain = i, this.normalizedSet = a;
	}
	static fromJson(t) {
		let n = qt(t.normalizer), r = hn(t.pre_tokenizer), i = Tn(t.model), a = tn(t.post_processor), o = (t.added_tokens ?? []).map((e) => ({
			id: e.id,
			content: e.content,
			lstrip: e.lstrip === !0,
			rstrip: e.rstrip === !0,
			singleWord: e.single_word === !0,
			normalized: e.normalized === !0,
			special: e.special === !0
		})).filter((e) => e.content !== ""), s = [...o.filter((e) => e.special), ...o.filter((e) => !e.special)], c = s.filter((e) => !e.normalized), l = s.filter((e) => e.normalized), u = l.map((e) => {
			let t = ht(e.content);
			return Ct(n ? n(t) : t);
		});
		return u.forEach((e, t) => {
			if (e === "") throw Error(`tokenizer.json: added token ${l[t].id} is empty after normalization`);
		}), new e(n, r, i, a, Dn(c, c.map((e) => e.content)), Dn(l, u));
	}
	static fromString(t) {
		return e.fromJson(JSON.parse(t));
	}
	split(e) {
		let t = [];
		for (let n of En(e, this.plain)) {
			let r = ht(e.slice(n.start, n.end), n.start);
			if (n.token) {
				t.push({
					token: n.token,
					span: r
				});
				continue;
			}
			let i = this.normalizer ? this.normalizer(r) : r;
			if (i.chars.length === 0) continue;
			let a = vt(i.chars);
			for (let e of En(Ct(i), this.normalizedSet)) {
				let n = a[e.start], r = a[e.end];
				n !== r && t.push(e.token ? {
					token: e.token,
					span: X(i, n, r)
				} : { text: X(i, n, r) });
			}
		}
		return t;
	}
	encodeSequence(e) {
		let t = {
			ids: [],
			offsets: [],
			wordIds: [],
			tokens: []
		}, n = 0;
		for (let r of this.split(e)) {
			if ("token" in r) {
				let e = r.span;
				t.ids.push(r.token.id), t.offsets.push(e.starts[0], e.ends[e.chars.length - 1]), t.wordIds.push(n), t.tokens.push(Ct(e)), n += 1;
				continue;
			}
			let e = this.preTokenizer ? this.preTokenizer(r.text) : [r.text];
			for (let r of e) if (r.chars.length !== 0) {
				for (let e of this.model.tokenize(r.chars)) t.ids.push(e.id), t.offsets.push(r.starts[e.start], r.ends[e.end - 1]), t.wordIds.push(n), t.tokens.push(e.value ?? "");
				n += 1;
			}
		}
		return t;
	}
	encode(e, t, n = {}) {
		let r = this.encodeSequence(e), i = t == null ? null : this.encodeSequence(t), a = n.truncation === !1 ? null : n.truncation === void 0 || n.truncation === !0 ? "longest_first" : n.truncation;
		if (n.maxLength !== void 0 && a) {
			let e = n.maxLength - this.post.addedTokens(i !== null), [t, o] = _n(r.ids.length, i ? i.ids.length : null, e, a), s = (e, t) => t >= e.ids.length ? e : {
				ids: e.ids.slice(0, t),
				offsets: e.offsets.slice(0, 2 * t),
				wordIds: e.wordIds.slice(0, t),
				tokens: e.tokens.slice(0, t)
			};
			r = s(r, t), i &&= s(i, o);
		}
		return this.post.apply(r, i);
	}
}, kn = 16, An = [128, 512], jn = 8;
async function Mn(e) {
	let t;
	try {
		let n = await fetch(e);
		if (!n.ok) throw Error(`fetch ${e}: ${n.status}`);
		t = await n.arrayBuffer();
	} catch (e) {
		return {
			bytes: 0,
			error: String(e instanceof Error ? e.message : e)
		};
	}
	try {
		return {
			bytes: t.byteLength,
			tokenizer: On.fromString(new TextDecoder().decode(t))
		};
	} catch (e) {
		return {
			bytes: t.byteLength,
			error: String(e instanceof Error ? e.message : e)
		};
	}
}
var Nn = class e {
	kh;
	weights;
	spec;
	head;
	task;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	batchFit = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	weightGpuBytes = 0;
	loadTiming = {};
	tokenizer = null;
	tokenizerError = null;
	precisionNote;
	recommendedPrecision;
	labels = [];
	textMaxLength = 0;
	constructor(e, t, n, r, i, a) {
		this.kh = e, this.weights = t, this.spec = n, this.head = r, this.task = i, this.precision = a;
	}
	static async load(n) {
		let r = n.precision !== "f32", a = performance.now(), o = s(n.manifestUrl), l = await t(r, n.limits !== "default"), d = await c(n.manifestUrl, n.precision, await o), f = d.manifest, p = d.url, m = performance.now() - a;
		if (f.format !== "kleinhirn-weights-2" || !f.spec) throw Error(`EncoderModel needs kleinhirn-weights-2, manifest is ${f.format}`);
		let h = p.slice(0, p.lastIndexOf("/") + 1), g = f.tokenizer ? Mn(h + f.tokenizer) : void 0, _ = await u(l.device, p, f), v = _.manifest.tensors[0]?.dtype ?? "f32";
		if (v === "f16" && !l.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n.precision && n.precision !== "auto" && n.precision !== v) throw Error(`precision ${n.precision} requested but manifest is ${v}`);
		let y = f.spec, b = new e(l, _, y, f.head, f.task ?? "", v);
		b.precisionNote = d.note, b.recommendedPrecision = f.recommendedPrecision;
		let x = await g;
		x?.error === void 0 ? x && (b.tokenizer = x.tokenizer) : b.tokenizerError = x.error;
		let S = b.head;
		b.labels = S.type === "classify" || S.type === "token" ? nt(f.labels, S.classes) : Object.entries(f.labels ?? {}).sort((e, t) => Number(e[0]) - Number(t[0])).map(([, e]) => e);
		let C = f.sentenceTransformers, w = Math.max(...n.buckets ?? An);
		b.textMaxLength = n.maxLength ?? Math.min(f.maxLength ?? Math.min(y.embed.maxPositions - y.embed.positionOffset, C?.maxSeqLength ?? Infinity), w), b.downloadBytes = _.downloadBytes + (x?.bytes ?? 0), b.weightGpuBytes = _.gpuBytes, b.loadTiming = {
			..._.timing,
			manifestMs: m,
			planMs: 0
		};
		let T = performance.now();
		return await i(l, () => {
			for (let e of n.buckets ?? An) {
				let t = b.planFor(e, 1);
				Te(t, l.device.limits, (e) => _.tensors.get(e)?.size), b.plans.set(e, new G(l.device, t, _.tensors));
			}
		}), b.loadTiming.planMs = performance.now() - T, b;
	}
	planFor(e, t) {
		return ve(this.spec, this.head, {
			length: e,
			batch: t,
			markers: 0,
			f16: this.precision === "f16"
		});
	}
	makePlan(e, t) {
		return new G(this.kh.device, this.planFor(e, t), this.weights.tensors);
	}
	pickBucket(e) {
		let t = [...this.plans.values()].filter((t) => e <= t.length).sort((e, t) => e.length - t.length);
		if (!t.length) throw new _(`seqLen ${e} exceeds the loaded buckets`);
		return t[0];
	}
	enqueue(e) {
		let t = this.queue.then(e);
		return this.queue = t.catch(() => {}), t;
	}
	wordRows(e, t) {
		let n = this.spec.embeddingSize, r = this.weights.embeddings, i = r instanceof Float32Array ? new Float32Array(t * e.length * n) : new Uint16Array(t * e.length * n);
		for (let [a, o] of e.entries()) for (let e = 0; e < o.inputIds.length; e += 1) {
			let s = q(o.inputIds[e], n, r.length);
			i.set(r.subarray(s, s + n), (a * t + e) * n);
		}
		return i;
	}
	maskAndTypes(e, t) {
		let n = new Float32Array(t * e.length), r = new Uint32Array(t * e.length);
		for (let [i, a] of e.entries()) if (n.fill(1, i * t, i * t + a.inputIds.length), a.typeIds && this.spec.embed.typeVocab > 0) for (let e = 0; e < a.inputIds.length; e += 1) {
			let n = a.typeIds[e] ?? 0;
			if (!Number.isInteger(n) || n < 0 || n >= this.spec.embed.typeVocab) throw Error(`token type ${n} is not an integer inside the ${this.spec.embed.typeVocab} type rows`);
			r[i * t + e] = n;
		}
		return {
			mask: n,
			typeIds: r
		};
	}
	get cols() {
		let e = this.head;
		if (e.type === "classify" || e.type === "token") return e.classes;
		if (e.type === "embed") {
			let t = this.spec.hidden;
			for (let n of e.steps) n.op === "dense" && (t = n.out);
			return t;
		}
		throw Error(`head ${e.type} is not an encoder head`);
	}
	shape(e, t, n, r) {
		let i = this.cols;
		if (this.head.type === "token") {
			let a = t * r.length * i;
			return {
				data: e.slice(a, a + n * i),
				rows: n,
				cols: i,
				seqLen: n
			};
		}
		return {
			data: e.slice(t * i, (t + 1) * i),
			rows: 1,
			cols: i,
			seqLen: n
		};
	}
	checkPositions(e) {
		let { maxPositions: t, positionOffset: n } = this.spec.embed;
		if (e > t - n) throw Error(`input of ${e} tokens exceeds the ${t} position rows minus offset ${n} (${t - n} tokens)`);
	}
	checkPadIds(e) {
		let t = F(this.spec.embed, e);
		if (t < 0) return;
		let { padId: n, positionOffset: r } = this.spec.embed;
		throw Error(r === n && t === 0 ? `token id ${e[0]} at index 0 is not the pad id ${n}: the position rule of this model needs the pad id first and nowhere else` : `token id ${e[t]} at index ${t} is the pad id ${n}: positions of RoBERTa and XLM-R count only the ids that are not the pad id, which the engine does not do`);
	}
	async runIds(e, t = {}) {
		let n = e.inputIds.length;
		if (n === 0) throw Error("empty input");
		this.checkPadIds(e.inputIds), this.checkPositions(n);
		let i = t.bucket === void 0 ? this.pickBucket(n) : this.plans.get(t.bucket);
		if (!i) throw Error(`bucket ${t.bucket} not loaded`);
		if (n > i.length) throw new _(`seqLen ${n} exceeds bucket ${i.length}`);
		return t.capture && i.assertCapturable(), this.enqueue(() => r(this.kh, () => {
			let { mask: r, typeIds: a } = this.maskAndTypes([e], i.length);
			i.upload({
				embeddings: this.wordRows([e], i.length),
				mask: r,
				typeIds: a
			}), i.submit(!!t.capture, { seqLen: n });
		}, async () => {
			let e = await i.readOutput(), r = this.shape(e, 0, n, i);
			return K(r.data, "engine output"), t.capture && (r.capture = await i.readCapture(), r.captureSlotElements = i.length * this.spec.hidden), r;
		}));
	}
	batchPlan(e, t) {
		let n = `${e}:${t}`, r = this.batchFit.get(n);
		if (r === 1) return;
		let i = this.batchPlans.get(`${e}:${r ?? t}`);
		if (i) return i;
		let a = r === void 0 ? Ee((t) => this.planFor(e, t), this.kh.device.limits, t) : this.planFor(e, r);
		if (this.batchFit.set(n, a ? a.batch : 1), !a) return;
		if (this.batchPlans.size >= jn) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let o = new G(this.kh.device, a, this.weights.tensors);
		return this.batchPlans.set(`${e}:${a.batch}`, o), o;
	}
	async runChunk(e, t) {
		let n = Math.max(...e.map((e) => e.inputIds.length));
		if (n === 0) throw Error("empty input");
		this.checkPositions(n);
		let i = t === void 0 ? this.pickBucket(n) : this.plans.get(t);
		if (!i) throw Error(`bucket ${t} not loaded`);
		if (n > i.length) throw new _(`batch exceeds bucket ${i.length}`);
		let a = ye(e.length), o = Math.min(i.length, be(n));
		return this.enqueue(async () => {
			let t = [], n;
			for (; t.length < e.length;) {
				let s = await r(this.kh, () => {
					n ??= (a === 1 ? void 0 : this.batchPlan(o, a)) ?? i;
					let r = e.slice(t.length, t.length + n.batch);
					return this.submitPiece(r, n), {
						rows: r,
						plan: n
					};
				}, (e) => this.readPiece(e.rows, e.plan));
				t.push(...s);
			}
			return t;
		});
	}
	submitPiece(e, t) {
		let n = [...e, ...Array.from({ length: t.batch - e.length }, () => ({ inputIds: [] }))], r = t.batch === 1 ? Math.max(...e.map((e) => e.inputIds.length)) : t.length * t.batch, { mask: i, typeIds: a } = this.maskAndTypes(n, t.length);
		t.upload({
			embeddings: this.wordRows(n, t.length),
			mask: i,
			typeIds: a
		}), t.submit(!1, { seqLen: r });
	}
	async readPiece(e, t) {
		let n = await t.readOutput();
		return e.map((e, r) => {
			let i = this.shape(n, r, e.inputIds.length, t);
			return K(i.data, "engine output"), i;
		});
	}
	async runIdsBatch(e, t = {}) {
		for (let [t, n] of e.entries()) {
			if (n.inputIds.length === 0) throw Error(`empty input (input ${t})`);
			try {
				this.checkPadIds(n.inputIds);
			} catch (e) {
				throw Error(`input ${t}: ${e.message}`);
			}
		}
		let n = e.map((e, t) => t).sort((t, n) => e[t].inputIds.length - e[n].inputIds.length || t - n), r = Array(e.length);
		for (let i = 0; i < n.length; i += B) {
			let a = n.slice(i, i + B), o = await this.runChunk(a.map((t) => e[t]), t.bucket);
			a.forEach((e, t) => {
				r[e] = o[t];
			});
		}
		return r;
	}
	requireTokenizer() {
		if (this.tokenizerError) throw Error(`text input unavailable: ${this.tokenizerError}`);
		if (!this.tokenizer) throw Error("this model has no tokenizer.json in its manifest");
		return this.tokenizer;
	}
	textInput(e, t, n = "longest_first") {
		let r = this.requireTokenizer().encode(e, t, {
			maxLength: this.textMaxLength,
			truncation: n
		});
		return {
			inputIds: r.ids,
			typeIds: r.typeIds
		};
	}
	requireHead(...e) {
		if (!e.includes(this.head.type)) throw Error(`needs a ${e.join(" or ")} head, this model has ${this.head.type}`);
	}
	async classify(e, t) {
		this.requireHead("classify");
		let n = (await this.runIds(this.textInput(e, t ?? null))).data, r = this.head, i = r.problem ?? (r.classes === 1 ? "regression" : "single");
		if (i === "regression") {
			let e = rt(n, n.length)[0];
			return {
				label: this.labels[e] ?? String(e),
				index: e,
				score: n[e],
				scores: n,
				logits: n
			};
		}
		let a;
		a = i === "multi" ? Float32Array.from(n, (e) => 1 / (1 + Math.exp(-e))) : et(n);
		let o = rt(i === "multi" ? n : a, a.length)[0];
		return {
			label: this.labels[o] ?? String(o),
			index: o,
			score: a[o],
			scores: a,
			logits: n
		};
	}
	async zeroShot(e, t, n = {}) {
		this.requireHead("classify");
		let r = this.labels.findIndex((e) => e.toLowerCase().startsWith("entail"));
		if (r < 0) throw Error(`no entailment label in [${this.labels.join(", ")}]`);
		if (!t.length) throw Error("zeroShot needs at least one label");
		let i = n.template ?? "This example is {}.", a = await this.runIdsBatch(t.map((t) => this.textInput(e, i.replace("{}", () => t), "only_first"))), o = Float32Array.from(a, (e) => e.data[r]), s = et(o), c = rt(s, s.length)[0];
		return {
			label: t[c],
			index: c,
			score: s[c],
			scores: s,
			logits: o
		};
	}
	async rerank(e, t) {
		if (this.requireHead("classify"), this.cols !== 1) throw Error(`rerank needs a one-logit head, this model has ${this.cols}`);
		let n = [];
		for (let r = 0; r < t.length; r += kn) {
			let i = await this.runIdsBatch(t.slice(r, r + kn).map((t) => this.textInput(e, t)));
			for (let e of i) n.push(e.data[0]);
		}
		return {
			scores: n,
			order: n.map((e, t) => t).sort((e, t) => n[t] - n[e] || e - t)
		};
	}
	async embed(e, t = {}) {
		this.requireHead("embed");
		let n = t.prompt ?? "", r = await this.runIdsBatch(e.map((e) => this.textInput(n + e, null))), i = this.head.normalize === !0;
		return r.map((e) => i ? tt(e.data) : e.data);
	}
	async tokenClassify(e) {
		this.requireHead("token");
		let t = this.requireTokenizer().encode(e, null, { maxLength: this.textMaxLength }), n = await this.runIds({
			inputIds: t.ids,
			typeIds: t.typeIds
		}), r = n.cols, i = new Float32Array(n.rows * r);
		for (let e = 0; e < n.rows; e += 1) i.set(et(n.data.subarray(e * r, (e + 1) * r)), e * r);
		return at(i, this.labels, t.offsets, t.specialTokensMask).map((t) => ({
			...t,
			text: e.slice(t.start, t.end)
		}));
	}
	liveGpuBytes() {
		let e = this.weightGpuBytes;
		for (let t of this.plans.values()) e += t.gpuBytes;
		for (let t of this.batchPlans.values()) e += t.gpuBytes;
		return e;
	}
	info() {
		return {
			precision: this.precision,
			...this.recommendedPrecision ? { recommendedPrecision: this.recommendedPrecision } : {},
			...this.precisionNote ? { precisionNote: this.precisionNote } : {},
			buildId: "mus637tw",
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps,
			family: this.spec.family,
			task: this.task,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			textMaxLength: this.textMaxLength,
			hasTokenizer: this.tokenizer !== null,
			batchSizes: [...z],
			gpuBytes: this.liveGpuBytes(),
			downloadBytes: this.downloadBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.kh.device.destroy();
	}
};
//#endregion
//#region src/index.ts
async function Pn(e) {
	let t = e.backend ?? "auto";
	if (t === "wasm") {
		let t = await s(e.manifestUrl);
		if (Fn(t)) throw Error("julia-1 needs WebGPU; the wasm fallback covers DeBERTa only");
		if (In(t)) throw Error("kleinhirn-weights-2 models need WebGPU; the wasm backend does not run them yet");
		return mt.load(e);
	}
	if (typeof navigator < "u" && navigator.gpu) {
		let n = await s(e.manifestUrl);
		if (Fn(n)) return pt.load(e, n);
		if (In(n)) return Nn.load({
			manifestUrl: e.manifestUrl,
			precision: e.precision,
			limits: e.limits,
			buckets: e.buckets?.map((e) => typeof e == "number" ? e : e.length)
		});
		try {
			return await zn.load(e);
		} catch (e) {
			if (t === "webgpu") throw e;
		}
	} else {
		let t = await s(e.manifestUrl);
		if (Fn(t)) throw Error("julia-1 needs WebGPU; the wasm fallback covers DeBERTa only");
		if (In(t)) throw Error("kleinhirn-weights-2 models need WebGPU; the wasm backend does not run them yet");
	}
	return mt.load(e);
}
function Fn(e) {
	return e.encoder?.arch === "modernbert-julia";
}
function In(e) {
	return e.format === "kleinhirn-weights-2";
}
var Ln = 16, Rn = 8, zn = class e {
	kh;
	weights;
	tokenizer;
	spec;
	temperature;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	batchFit = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	tokenizerBytes = 0;
	weightGpuBytes = 0;
	loadTiming = {};
	cache = new V(0);
	headHidden = 768;
	constructor(e, t, n, r, i, a) {
		this.kh = e, this.weights = t, this.tokenizer = n, this.spec = r, this.temperature = i, this.precision = a;
	}
	static async load(n) {
		let r = n.precision !== "f32", a = n.manifestUrl.slice(0, n.manifestUrl.lastIndexOf("/") + 1), o = performance.now(), c = s(n.manifestUrl), l = await t(r, n.limits !== "default"), d = await c, f = performance.now() - o, p = (async () => (await fetch(a + d.tokenizer)).arrayBuffer())(), m = await u(l.device, n.manifestUrl, d), h = m.manifest.tensors.find((e) => e.name.endsWith("qkv.weight"))?.dtype ?? "f32";
		if (h === "f16" && !l.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n.precision && n.precision !== "auto" && n.precision !== h) throw Error(`precision ${n.precision} requested but manifest is ${h}`);
		let _ = h, v = performance.now(), y = await p, b = new g(JSON.parse(new TextDecoder().decode(y))), x = performance.now() - v, S = m.manifest.encoder, C = new e(l, m, b, S, m.manifest.head.temperature, _);
		C.downloadBytes = m.downloadBytes + y.byteLength, C.tokenizerBytes = y.byteLength, C.weightGpuBytes = m.gpuBytes, C.loadTiming = {
			...m.timing,
			manifestMs: f,
			tokenizerMs: x,
			planMs: 0
		}, C.headHidden = Number(m.manifest.head.hiddenSize) || 768, C.cache = new V(n.cacheSize ?? 256);
		let w = performance.now();
		return await i(l, () => {
			for (let e of n.buckets ?? [128]) {
				let t = typeof e == "number" ? {
					length: e,
					markers: Ln
				} : {
					length: e.length,
					markers: e.markers ?? Ln
				}, n = C.planFor(t.length, t.markers, 1);
				Te(n, l.device.limits, (e) => m.tensors.get(e)?.size), C.plans.set(t.length, new G(l.device, n, m.tensors));
			}
		}), C.loadTiming.planMs = performance.now() - w, C;
	}
	planFor(e, t, n) {
		let { spec: r, head: i } = le(this.spec, {
			temperature: this.temperature,
			hiddenSize: this.headHidden
		}, t);
		return ve(r, i, {
			length: e,
			batch: n,
			markers: t,
			f16: this.precision === "f16"
		});
	}
	pickBucket(e, t = 1) {
		let n = [...this.plans.values()].filter((n) => e <= n.length && t <= n.markers).sort((e, t) => e.length - t.length || e.markers - t.markers);
		if (!n.length) throw new _(`seqLen ${e} or ${t} markers exceed loaded buckets`);
		return n[0];
	}
	embeddingRows(e, t) {
		let n = t.length, r = this.spec.hiddenSize;
		if (this.weights.embeddings instanceof Float32Array) {
			let t = new Float32Array(n * r), i = this.weights.embeddings;
			for (let n = 0; n < e.seqLen; n += 1) {
				let a = q(e.inputIds[n], r, i.length);
				t.set(i.subarray(a, a + r), n * r);
			}
			return t;
		}
		let i = new Uint16Array(n * r), a = this.weights.embeddings;
		for (let t = 0; t < e.seqLen; t += 1) {
			let n = q(e.inputIds[t], r, a.length);
			i.set(a.subarray(n, n + r), t * r);
		}
		return i;
	}
	packedMarkers(e, t) {
		let n = new Uint32Array(3 * t);
		return n.set(e.markerIndices.subarray(0, t), 0), n.set(new Uint32Array(e.markerMask.buffer).subarray(0, t), t), n.set(e.markerGroups.subarray(0, t), 2 * t), n;
	}
	maskOf(e, t) {
		let n = new Float32Array(t);
		return n.fill(1, 0, e.seqLen), n;
	}
	enqueue(e) {
		let t = this.queue.then(e);
		return this.queue = t.catch(() => {}), t;
	}
	async runPrepared(e, t = !1, n) {
		let i = e.markerMask.reduce((e, t) => e + +(t > .5), 0), a = n === void 0 ? this.pickBucket(e.seqLen, i) : this.plans.get(n);
		if (!a) throw Error(`bucket ${n} not loaded`);
		if (e.seqLen > a.length || i > a.markers) throw new _(`seqLen ${e.seqLen}/${i} markers exceeds bucket ${a.length}/${a.markers}`);
		t && a.assertCapturable();
		let o = H(e), s = t ? void 0 : this.cache.get(o);
		return s ? $(s) : this.enqueue(() => r(this.kh, () => {
			a.upload({
				embeddings: this.embeddingRows(e, a),
				mask: this.maskOf(e, a.length),
				packedMarkers: this.packedMarkers(e, a.markers)
			}), a.submit(t, { seqLen: e.seqLen });
		}, async () => {
			let n = await a.readLogits();
			K(n, "logits");
			let r = Bn(n, e.markerGroups, e.markerMask);
			return t || this.cache.set(o, $({
				logits: n,
				probabilities: r
			})), {
				logits: n,
				probabilities: r,
				captureData: t ? await a.readCapture() : void 0
			};
		}));
	}
	batchPlan(e, t, n) {
		let r = `${e}:${t}:${n}`, i = this.batchFit.get(r);
		if (i === 1) return;
		let a = this.batchPlans.get(`${e}:${t}:${i ?? n}`);
		if (a) return a;
		let o = i === void 0 ? Ee((n) => this.planFor(e, t, n), this.kh.device.limits, n) : this.planFor(e, t, i);
		if (this.batchFit.set(r, o ? o.batch : 1), !o) return;
		if (this.batchPlans.size >= Rn) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let s = new G(this.kh.device, o, this.weights.tensors);
		return this.batchPlans.set(`${e}:${t}:${o.batch}`, s), s;
	}
	padInput() {
		return {
			inputIds: /* @__PURE__ */ new Int32Array(),
			attentionMask: /* @__PURE__ */ new Int32Array(),
			markerIndices: /* @__PURE__ */ new Int32Array(),
			markerMask: /* @__PURE__ */ new Float32Array(),
			markerGroups: /* @__PURE__ */ new Int32Array(),
			seqLen: 0
		};
	}
	batchEmbeddingRows(e, t) {
		let n = this.spec.hiddenSize, r = this.weights.embeddings, i = r instanceof Float32Array ? new Float32Array(t.length * t.batch * n) : new Uint16Array(t.length * t.batch * n);
		for (let [a, o] of e.entries()) {
			let e = a * t.length * n;
			for (let t = 0; t < o.seqLen; t += 1) {
				let a = q(o.inputIds[t], n, r.length);
				i.set(r.subarray(a, a + n), e + t * n);
			}
		}
		return i;
	}
	batchMask(e, t) {
		let n = new Float32Array(t.length * t.batch);
		for (let [r, i] of e.entries()) n.fill(1, r * t.length, r * t.length + i.seqLen);
		return n;
	}
	batchPackedMarkers(e, t) {
		let n = new Uint32Array(e.length * 3 * t.markers);
		for (let [r, i] of e.entries()) n.set(this.packedMarkers(i, t.markers), r * 3 * t.markers);
		return n;
	}
	async runBatchChunk(e, t) {
		let n = Math.max(...e.map((e) => e.seqLen)), i = Math.max(...e.map((e) => e.markerMask.reduce((e, t) => e + +(t > .5), 0))), a = t === void 0 ? this.pickBucket(n, i) : this.plans.get(t);
		if (!a) throw Error(`bucket ${t} not loaded`);
		if (n > a.length || i > a.markers) throw new _(`batch exceeds bucket ${a.length}/${a.markers}`);
		let o = ye(e.length), s = Math.min(a.length, be(n));
		return this.enqueue(async () => {
			let t = [], n;
			for (; t.length < e.length;) {
				let i = await r(this.kh, () => {
					n ??= (o === 1 ? void 0 : this.batchPlan(s, a.markers, o)) ?? a;
					let r = e.slice(t.length, t.length + n.batch);
					return this.submitPiece(r, n), {
						rows: r,
						plan: n
					};
				}, (e) => this.readPiece(e.rows, e.plan));
				t.push(...i);
			}
			return t;
		});
	}
	submitPiece(e, t) {
		let n = e.length === t.batch ? e : [...e, ...Array.from({ length: t.batch - e.length }, () => this.padInput())], r = t.batch === 1 ? Math.max(...e.map((e) => e.seqLen)) : t.length * t.batch;
		t.upload({
			embeddings: this.batchEmbeddingRows(n, t),
			mask: this.batchMask(n, t),
			packedMarkers: this.batchPackedMarkers(n, t)
		}), t.submit(!1, { seqLen: r });
	}
	async readPiece(e, t) {
		let n = await t.readLogits();
		return e.map((e, r) => {
			let i = n.slice(r * t.markers, (r + 1) * t.markers);
			K(i, "logits");
			let a = {
				logits: i,
				probabilities: Bn(i, e.markerGroups, e.markerMask)
			};
			return this.cache.set(H(e), $(a)), a;
		});
	}
	async runPreparedBatch(e, t) {
		let n = Array(e.length), r = [], i = [];
		for (let [t, a] of e.entries()) {
			let e = this.cache.get(H(a));
			e ? n[t] = $(e) : (r.push(a), i.push(t));
		}
		let { unique: a, slot: o } = W(r, H), s = xe(a, o, (e) => e.seqLen), c = [];
		for (let e = 0; e < s.unique.length; e += B) {
			let n = await this.runBatchChunk(s.unique.slice(e, e + B), t);
			c.push(...n);
		}
		let l = /* @__PURE__ */ new Set();
		for (let [e] of r.entries()) {
			let t = c[s.slot[e]];
			n[i[e]] = l.has(s.slot[e]) ? $(t) : t, l.add(s.slot[e]);
		}
		return n;
	}
	async classify(e, t) {
		let n = performance.now(), r = t.map((e) => [e.task, e.labels]), i = r.reduce((e, [, t]) => e + t.length, 0), a = w(this.tokenizer, e, r, this.maxBucket(i), i), o = performance.now() - n, s = performance.now(), { logits: c, probabilities: l } = await this.runPrepared(a), u = performance.now() - s;
		return this.toResult(t, a, c, l, {
			tokenizeMs: o,
			gpuMs: u,
			totalMs: performance.now() - n
		});
	}
	async classifyBatch(e) {
		let t = performance.now(), n = e.map((e) => {
			let t = e.tasks.map((e) => [e.task, e.labels]), n = t.reduce((e, [, t]) => e + t.length, 0);
			return w(this.tokenizer, e.text, t, this.maxBucket(n), n);
		}), r = performance.now() - t, i = performance.now(), a = await this.runPreparedBatch(n), o = {
			tokenizeMs: r,
			gpuMs: performance.now() - i,
			totalMs: performance.now() - t
		};
		return e.map((e, t) => this.toResult(e.tasks, n[t], a[t].logits, a[t].probabilities, o));
	}
	toResult(e, t, n, r, i) {
		return {
			tasks: e.map(({ task: e, labels: i }, a) => ({
				task: e,
				labels: i.map((e, i) => {
					let o = Vn(t.markerGroups, a, i);
					return o < 0 ? {
						label: e,
						probability: 0,
						logit: -1e4
					} : {
						label: e,
						probability: r[o],
						logit: n[o]
					};
				})
			})),
			timings: i
		};
	}
	maxBucket(e = 1) {
		let t = [...this.plans.values()].filter((t) => t.markers >= e).map((e) => e.length);
		if (!t.length) throw new _(`no bucket routes ${e} markers`);
		return Math.max(...t);
	}
	async profile(e, t = {}) {
		let n = await this.profileDetailed(e, t);
		return n && n.times;
	}
	async profileDetailed(e, t = {}) {
		let n = e.markerMask.reduce((e, t) => e + +(t > .5), 0), r = this.pickBucket(e.seqLen, n);
		return this.enqueue(() => r.profileForward({
			embeddings: this.embeddingRows(e, r),
			mask: this.maskOf(e, r.length),
			packedMarkers: this.packedMarkers(e, r.markers)
		}, t.granularity ?? "pass", e.seqLen));
	}
	liveGpuBytes() {
		let e = this.weightGpuBytes;
		for (let t of this.plans.values()) e += t.gpuBytes;
		for (let t of this.batchPlans.values()) e += t.gpuBytes;
		return e;
	}
	info() {
		return {
			precision: this.precision,
			buildId: "mus637tw",
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			gpuBytes: this.liveGpuBytes(),
			downloadBytes: this.downloadBytes,
			tokenizerBytes: this.tokenizerBytes,
			weightBytes: this.downloadBytes - this.tokenizerBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.kh.device.destroy();
	}
};
function $(e) {
	return {
		logits: e.logits.slice(),
		probabilities: e.probabilities.slice()
	};
}
function Bn(e, t, n) {
	let r = new Float32Array(e.length), i = /* @__PURE__ */ new Map();
	for (let r = 0; r < e.length; r += 1) {
		if (n[r] <= .5) continue;
		let e = t[r];
		i.has(e) || i.set(e, []), i.get(e).push(r);
	}
	for (let t of i.values()) {
		let n = -Infinity;
		for (let r of t) n = Math.max(n, e[r]);
		let i = 0;
		for (let r of t) i += Math.exp(e[r] - n);
		for (let a of t) r[a] = Math.exp(e[a] - n) / i;
	}
	return r;
}
function Vn(e, t, n) {
	let r = 0;
	for (let i = 0; i < e.length; i += 1) if (e[i] === t) {
		if (r === n) return i;
		r += 1;
	}
	return -1;
}
//#endregion
export { _ as BucketOverflowError, Nn as EncoderModel, On as JsonTokenizer, zn as Kleinhirn, Bn as groupSoftmax, Pn as loadEngine };
