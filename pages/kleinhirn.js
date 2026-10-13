//#region src/words.ts
var e = class {
	view;
	dtype;
	width;
	stride;
	source;
	rows;
	constructor(e, t, n, r, i) {
		this.view = e, this.dtype = t, this.width = n, this.stride = r, this.source = i, this.rows = Math.floor(e.length / r);
	}
	ensure() {
		return null;
	}
	copyRow(e, t, n) {
		if (!Number.isInteger(e) || e < 0 || e >= this.rows) throw Error(`token id ${e} is outside the vocabulary of ${this.rows} rows (ids are integers from 0)`);
		let r = e * this.stride;
		t.set(this.view.subarray(r, r + this.width), n);
	}
	alloc(e) {
		return this.dtype === "f16" ? new Uint16Array(e) : new Float32Array(e);
	}
	info() {
		return {
			mode: "full",
			source: this.source,
			rows: this.rows,
			width: this.width,
			dtype: this.dtype
		};
	}
};
function t(t, n) {
	if (t.byteLength !== n.rows * n.rowBytes) throw Error(`words.bin is ${t.byteLength} bytes, ${n.rows} rows of ${n.rowBytes} need ${n.rows * n.rowBytes}`);
	let r = n.dtype === "f16" ? new Uint16Array(t) : new Float32Array(t);
	return new e(r, n.dtype, n.width, n.rowBytes / r.BYTES_PER_ELEMENT, "words.bin");
}
function n(t, n, r) {
	return new e(n === "f16" ? new Uint16Array(t) : new Float32Array(t), n, r, r, "legacy");
}
function r(e, t) {
	let n = e.info();
	return n.source === "legacy" && t !== "full" ? {
		...n,
		requested: t,
		note: "legacy manifest has no words.bin"
	} : {
		...n,
		requested: t
	};
}
async function i(e, n) {
	if (n.rowBytes !== n.width * (n.dtype === "f16" ? 2 : 4) + 4) throw Error(`words: rowBytes ${n.rowBytes} is not ${n.width} ${n.dtype} values and a checksum`);
	let r = performance.now(), i = await fetch(e);
	if (!i.ok) throw Error(`fetch ${e}: ${i.status}`);
	let a = await i.arrayBuffer(), o = performance.now() - r;
	if (a.byteLength !== n.bytes) throw Error(`words.bin is ${a.byteLength} bytes, manifest says ${n.bytes}`);
	let s = performance.now(), c = [...new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", a))].map((e) => e.toString(16).padStart(2, "0")).join(""), l = performance.now() - s;
	if (c !== n.sha256) throw Error("sha256 mismatch on words.bin");
	return {
		table: t(a, n),
		bytes: a.byteLength,
		fetchMs: o,
		sha256Ms: l
	};
}
function a(e, t) {
	let n;
	for (let r = 0; r < t.length; r += 1) {
		let { inputIds: i, seqLen: a } = t[r], o = e.ensure(i, a ?? i.length);
		o && (n ??= []).push(o);
	}
	return n ? Promise.all(n).then(() => void 0) : null;
}
//#endregion
//#region src/weights.ts
async function o(e, t, r, a) {
	if (t.words) {
		let n = await (a ?? i(e + t.words.file, t.words));
		return {
			table: n.table,
			bytes: Number(n.table.info().bytes ?? n.bytes),
			timing: {
				embedJoinMs: 0,
				wordsFetchMs: n.fetchMs,
				wordsSha256Ms: n.sha256Ms
			}
		};
	}
	let o = performance.now(), s = t.tensors.filter((e) => e.keepOnCpu).sort((e, t) => (e.rowStart ?? 0) - (t.rowStart ?? 0));
	if (!s.length) throw Error("manifest has neither words.bin nor a keepOnCpu word table");
	if (s.some((e) => e.name !== "embeddings.word.weight")) throw Error("only the word table stays on the CPU");
	let c = new Uint8Array(s.reduce((e, t) => e + t.byteLength, 0)), l = 0;
	for (let e of s) c.set(new Uint8Array(r[e.shard], e.offset, e.byteLength), l), l += e.byteLength;
	return {
		table: n(c.buffer, s[0].dtype === "f16" ? "f16" : "f32", s[0].shape[1]),
		bytes: 0,
		timing: {
			embedJoinMs: performance.now() - o,
			wordsFetchMs: 0,
			wordsSha256Ms: 0
		}
	};
}
function s(e) {
	return [...new Uint8Array(e)].map((e) => e.toString(16).padStart(2, "0")).join("");
}
async function c(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	let n = performance.now();
	return {
		buf: await t.arrayBuffer(),
		bodyMs: performance.now() - n
	};
}
async function l(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	return t.json();
}
async function u(e, t, n, r = l) {
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
async function d(e, t) {
	let n = e.slice(0, e.lastIndexOf("/") + 1), r = t ?? await l(e), i = new TextEncoder().encode(JSON.stringify(r)).byteLength, a = 0, o = 0, u = performance.now(), d = await Promise.all(r.shards.map(async (e) => {
		let { buf: t, bodyMs: r } = await c(n + e.file);
		o += r;
		let l = performance.now(), u = s(await crypto.subtle.digest("SHA-256", t));
		if (a += performance.now() - l, u !== e.sha256) throw Error(`sha256 mismatch on ${e.file}`);
		return i += t.byteLength, t;
	})), f = performance.now() - u;
	return {
		manifest: r,
		shardBytes: d,
		downloadBytes: i,
		fetchMs: f,
		bodyMs: o,
		sha256Ms: a
	};
}
//#endregion
//#region src/backend.ts
var f = (e) => e.replace(/\/f16\//, "/f32/");
function p(e, n, r) {
	if (r === "full") return i(e, n);
	let a = performance.now();
	return import("./words-rows-DsElhaea.js").then((i) => i.RemoteRows.open(e, n, {
		progressive: r === "progressive",
		view: t
	})).then((e) => ({
		table: e,
		bytes: 0,
		fetchMs: performance.now() - a,
		sha256Ms: 0
	}));
}
var m = () => import("./webgpu-backend-C9w9O5jV.js");
async function h(e, t, n, r, i, a = "full") {
	let s = e.slice(0, e.lastIndexOf("/") + 1), c = t.words ? p(s + t.words.file, t.words, a) : void 0;
	if (c?.catch(() => void 0), n) return (await m()).loadGpu(n, e, t, c);
	let [{ loadWasm: l, workerTransport: u }, f] = await Promise.all([import("./wasm-backend-DlqXkx4-.js"), d(e, t)]), h = await o(s, t, f.shardBytes, c);
	return l(f, t, (r ?? u)(), i, h);
}
//#endregion
//#region src/cache.ts
var g = [
	1,
	4,
	8,
	16
], _ = g[g.length - 1];
function v(e) {
	for (let t of g) if (e <= t) return t;
	throw Error(`batch of ${e} exceeds the maximum of ${_}`);
}
var y = class {
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
function b(e) {
	return `${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markerIndices.join(",")}|${e.markerMask.join(",")}|${e.markerGroups.join(",")}`;
}
function x(e) {
	return `${e.qtype}|${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markers.join(",")}`;
}
function S(e) {
	return Math.ceil(e / 64) * 64;
}
function C(e, t, n) {
	let r = e.map((e, t) => t).sort((t, r) => n(e[t]) - n(e[r])), i = Array(e.length);
	return r.forEach((e, t) => {
		i[e] = t;
	}), {
		unique: r.map((t) => e[t]),
		slot: t.map((e) => i[e])
	};
}
function w(e, t) {
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
//#region src/device.ts
var T = {
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
async function E(e, t) {
	if (!navigator.gpu) throw Error("WebGPU unavailable: navigator.gpu missing");
	let n = await navigator.gpu.requestAdapter();
	if (!n) throw Error("WebGPU unavailable: no adapter");
	let r = n.features.has("shader-f16"), i = n.features.has("timestamp-query"), a = [];
	e && r && a.push("shader-f16"), i && a.push("timestamp-query");
	let o = await n.requestDevice({
		requiredFeatures: a,
		...t ? { requiredLimits: T } : {}
	}), s = n.info ?? {}, c = {
		adapter: n,
		device: o,
		hasF16: a.includes("shader-f16"),
		hasTimestamps: a.includes("timestamp-query"),
		limitsMode: t ? "minimum" : "default",
		adapterInfo: {
			vendor: s.vendor ?? "",
			architecture: s.architecture ?? "",
			device: s.device ?? "",
			description: s.description ?? ""
		},
		failure: null
	};
	return o.addEventListener("uncapturederror", (e) => {
		c.failure ??= `WebGPU error: ${e.error.message}`;
	}), o.lost.then((e) => {
		c.failure ??= `WebGPU device lost (${e.reason}): ${e.message}`;
	}), c;
}
function D(e) {
	if (e.failure) throw Error(e.failure);
}
async function O(e, t, n) {
	D(e);
	let { device: r } = e;
	r.pushErrorScope("validation");
	let i, a;
	try {
		i = t();
	} finally {
		a = r.popErrorScope(), a.catch(() => {});
	}
	let [o, s] = await Promise.allSettled([n(i), a]);
	if (s.status === "rejected") throw s.reason;
	if (s.value) throw Error(`WebGPU validation error: ${s.value.message}`);
	if (D(e), o.status === "rejected") throw o.reason;
	return o.value;
}
function k(e, t) {
	return O(e, t, async (e) => e);
}
//#endregion
//#region src/plan/layers.ts
var A = (e, t = "zero") => e === void 0 ? t : `w:${e}`, j = (e, t) => {
	if (e === void 0) throw Error(`layer tensor ${t} missing`);
	return `w:${e}`;
}, M = (e) => Math.ceil(e / 16), N = 65535, P = (e, t) => Math.min(Math.ceil(e * t / 64), N);
function F(e, t, n, r, i, a) {
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
		dispatch: [M(n), "rows16"]
	};
}
var I = 134217728, L = (e) => e.headDim % 32 == 0 && e.length % 32 == 0 && e.rows * e.heads * e.length * 4 <= I;
function R(e, t, n, r) {
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
				j(r.posKey, "posKey"),
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
				j(r.posQuery, "posQuery"),
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
function z(e, t, n) {
	let r = {
		L: e.length,
		H: e.heads,
		D: e.headDim
	};
	if (L(e)) return [
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
function B(e, t, n, r) {
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
function V(e, t, n) {
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
		dispatch: [P(t.rows, t.hidden), 1]
	};
}
function ee(e, t, n) {
	let r = e.hidden, i = e.intermediate;
	return [
		F("qkv", e, 3 * r, r, 0, [
			"x",
			j(t.qkvW, "qkvW"),
			A(t.qkvB),
			"qkv"
		]),
		...n.standard ? z(e, n.attnScale, 0) : n.rel && L(e) ? R(e, n.attnScale, n.rel, t) : [{
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
				j(t.posKey, "posKey"),
				j(t.posQuery, "posQuery"),
				"relidx",
				"mask",
				"ctx"
			],
			dispatch: [e.heads, "rows"]
		}],
		F("attnOut", e, r, r, 0, [
			"ctx",
			j(t.attnOutW, "attnOutW"),
			A(t.attnOutB),
			"attnOut"
		]),
		B("lnA", e, 2, [
			"x",
			"attnOut",
			j(t.attnNormW, "attnNormW"),
			A(t.attnNormB),
			"mask",
			"tmp"
		]),
		F("ffn1", e, i, r, n.act, [
			"tmp",
			j(t.ffnInW, "ffnInW"),
			A(t.ffnInB),
			"mid"
		]),
		F("ffn2", e, r, i, 0, [
			"mid",
			j(t.ffnOutW, "ffnOutW"),
			A(t.ffnOutB),
			"ffnOut"
		]),
		B("lnF", e, 2, [
			"tmp",
			"ffnOut",
			j(t.ffnNormW, "ffnNormW"),
			A(t.ffnNormB),
			"mask",
			"x"
		])
	];
}
function te(e, t, n) {
	let r = e.hidden, i = e.intermediate, a = [];
	return n.firstNorm && a.push(B("lnA", e, 0, [
		n.stream,
		"dummy",
		j(t.attnNormW, "attnNormW"),
		A(t.attnNormB),
		"mask",
		"normed"
	])), a.push(F("qkv", e, 3 * r, r, 0, [
		n.firstNorm ? "normed" : n.stream,
		j(t.qkvW, "qkvW"),
		A(t.qkvB),
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
	}), a.push(...z(e, n.attnScale, n.window)), a.push(F("attnOut", e, r, r, 0, [
		"ctx",
		j(t.attnOutW, "attnOutW"),
		A(t.attnOutB),
		"attnOut"
	])), a.push(V("addA", e, [n.stream, "attnOut"])), a.push(B("lnF", e, 0, [
		n.stream,
		"dummy",
		j(t.ffnNormW, "ffnNormW"),
		A(t.ffnNormB),
		"mask",
		"normed"
	])), n.ffn.kind === "geglu" ? (a.push(F("ffnIn", e, 2 * i, r, 0, [
		"normed",
		j(t.ffnInW, "ffnInW"),
		A(t.ffnInB),
		"mid"
	])), a.push({
		name: "geglu",
		kernel: "geglu",
		constants: { I: i },
		bind: ["mid", "gate"],
		dispatch: ["rows", 1]
	}), a.push(F("ffnOut", e, r, i, 0, [
		"gate",
		j(t.ffnOutW, "ffnOutW"),
		A(t.ffnOutB),
		"ffnOut"
	]))) : (a.push(F("lin1", e, n.ffn.width, r, n.ffn.act, [
		"normed",
		j(t.ffnInW, "ffnInW"),
		A(t.ffnInB),
		"mid"
	])), a.push(F("lin2", e, r, n.ffn.width, 0, [
		"mid",
		j(t.ffnOutW, "ffnOutW"),
		A(t.ffnOutB),
		"ffnOut"
	]))), a.push(V("addF", e, [n.stream, "ffnOut"])), a;
}
//#endregion
//#region src/plan/spec.ts
function H(e, t) {
	let n = 0;
	return Array.from(t, (t) => t === e ? e : e + ++n);
}
var U = {
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
}, W = {
	w: "conv.weight",
	b: "conv.bias",
	normW: "conv.ln.weight",
	normB: "conv.ln.bias"
}, G = {
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
function ne(e) {
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
var re = {
	...U,
	layer: (e) => {
		let { posKey: t, posQuery: n, ...r } = U.layer(e);
		return r;
	}
};
function ie(e, t) {
	if (e === "bert" || e === "electra" || e === "roberta" || e === "xlm-roberta" || e === "distilbert") return re;
	if (e === "deberta-v2") return t?.conv ? {
		...U,
		conv: W
	} : U;
	if (e === "modernbert") return t ? ne({
		attention: t.attention.bias,
		mlp: t.ffn.bias,
		norm: t.block.norm.bias
	}) : G;
	throw Error(`no tensor names for family ${e}`);
}
function ae(e, t, n = 16) {
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
function oe(e) {
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
function K(e, t, n, r) {
	let i = e - t, a = n >> 1, o = Math.abs(i), s = i;
	return o > a && (s = (Math.ceil(Math.log(o / a) / Math.log((r - 1) / a) * (a - 1)) + a) * Math.sign(i)), Math.min(Math.max(s + n, 0), 2 * n - 1);
}
function se(e, t, n) {
	let r = new Uint32Array(e * e);
	for (let i = 0; i < e; i += 1) for (let a = 0; a < e; a += 1) r[i * e + a] = K(i, a, t, n);
	return r;
}
function q(e, t, n) {
	let r = n / 2, i = new Float32Array(e * 2 * n);
	for (let a = 0; a < e; a += 1) for (let e = 0; e < r; e += 1) {
		let r = a * t ** (-(2 * e) / n);
		i[a * 2 * n + e] = Math.cos(r), i[a * 2 * n + n + e] = Math.sin(r);
	}
	return i;
}
function J(e, t) {
	return e === "add" || e === "rope" ? [0] : [t - 1];
}
function ce(e, t) {
	for (let n of e) for (let e of [...n.captureOps ?? [], ...n.ops]) {
		let r = `${n.prefix}${e.name}`;
		for (let n of e.bind) if (!n.startsWith("w:") && !t.has(n)) throw Error(`plan op ${r}: unknown buffer ${n}`);
		for (let t of J(e.kernel, e.bind.length)) {
			let n = e.bind[t];
			if (e.bind.some((e, r) => r !== t && e === n)) throw Error(`plan op ${r}: writable buffer ${n} is bound twice`);
		}
	}
}
var le = {
	relu: 1,
	gelu: 2,
	tanh: 3,
	silu: 4
};
function Y(e) {
	let t = le[e];
	if (t === void 0) throw Error(`activation ${e} has no kernel yet`);
	return t;
}
var ue = (e) => {
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
function de(e, t, n) {
	let { length: r, batch: i, f16: a } = n, o = n.tensorNames ?? ie(e.family, e), s = e.hidden, c = e.embeddingSize, l = (e) => Math.ceil(e * (a ? 2 : 4) / 4) * 4, u = r * i, d = t.type === "classify" || t.type === "token" || t.type === "embed", f = t.type === "julia" ? t.options : d ? 0 : n.markers, p = f * i, m = e.block.norm.eps, h = {
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
	})), S = `w:${o.embedNormW}`, C = o.embedNormB ? `w:${o.embedNormB}` : "zero", w = e.embed.positions === "absolute";
	if (w || e.embed.typeVocab > 0) {
		let t = e.embed.typeVocab > 0 ? "w:embeddings.type.weight" : "zero", n = e.embed.project ? "embE" : "x", i = [{
			name: "embed",
			kernel: "embln",
			constants: {
				N: c,
				L: r,
				OFFSET: w ? e.embed.positionOffset : 0,
				MAXPOS: w ? e.embed.maxPositions : 1,
				EPS: e.embed.norm.eps,
				MASKMUL: +!!e.embed.maskMultiply,
				POSIDS: e.embed.padId === void 0 ? 0 : 1
			},
			bind: [
				"emb",
				w ? "w:embeddings.position.weight" : "zero",
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
	let T = e.conv;
	if (T) {
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
					KS: T.kernel
				},
				bind: [
					"x",
					"mask",
					"cols"
				],
				dispatch: ["rows", 1]
			}, _("conv", u, s, T.kernel * s, Y(T.act), [
				"cols",
				`w:${o.conv.w}`,
				`w:${o.conv.b}`,
				"convOut"
			], "rows16")]
		});
	}
	let E = e.attention.kind === "deberta-relative" ? e.attention.rel : void 0, D;
	if (e.attention.kind === "deberta-relative") {
		if (!E) throw Error("deberta-relative attention needs attention.rel");
		let t = Math.sqrt((1 + E.types.length) * (s / e.heads));
		D = () => ({
			scale: t,
			window: 0
		});
	} else {
		let t = e.attention.window, n = (s / e.heads) ** -.5;
		D = (e) => ({
			scale: n,
			window: !t || e % t.globalEvery === 0 ? 0 : t.half
		});
	}
	let O = e.attention.rope, k = !!O && O.thetaGlobal !== O.thetaLocal;
	for (let t = 0; t < e.layers; t += 1) {
		let n = o.layer(t), i = D(t), a;
		if (e.block.order === "post") a = ee(h, n, {
			act: Y(e.ffn.act),
			attnScale: i.scale,
			standard: e.attention.kind === "standard",
			rel: E && {
				moff: K(0, r - 1, E.buckets, E.maxPositions),
				nm: K(r - 1, 0, E.buckets, E.maxPositions) - K(0, r - 1, E.buckets, E.maxPositions) + 1
			}
		});
		else {
			let r = e.block.firstNormIdentity && t === 0, o = e.ffn.kind === "geglu" ? { kind: "geglu" } : {
				kind: "mlp",
				width: e.intermediate,
				act: Y(e.ffn.act)
			};
			a = te(h, r ? {
				...n,
				attnNormW: void 0
			} : n, {
				stream: "x",
				firstNorm: !r,
				attnScale: i.scale,
				window: i.window,
				rope: !!O,
				ffn: o,
				ropeTable: k ? i.window === 0 ? "cossinG" : "cossinL" : "cossin"
			});
		}
		let c = t === 0 && T;
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
				dispatch: [P(u, s), 1]
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
	let A = "x";
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
		}), A = "tmp";
	}
	let j, M = [], N = p, F = 1, I = "logits", L = !1;
	if (t.type === "classify" || t.type === "token" || t.type === "embed") {
		let e = t.type === "token", n = e ? u : i, a = e ? "rows16" : Math.ceil(i / 16), o = A, c = s, l = [];
		if (!e) {
			let e = t.pool === "cls" ? "first" : t.pool;
			e === "first" ? (L = !0, l.push({
				name: "gather",
				kernel: "gather",
				constants: {
					K: 1,
					D: s,
					L: r
				},
				bind: [
					"first",
					A,
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
					A,
					"mask",
					"states"
				],
				dispatch: [Math.ceil(s / 64), i]
			}), M.push({
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
				}), M.push({
					id: u,
					elements: n * c,
					final: a
				}), o = u;
				return;
			}
			if (r.in !== c) throw Error(`head step ${r.name}: input ${r.in}, previous width ${c}`);
			let u = s === t.steps.length - 1, d = u ? "out" : `hd${s}`;
			l.push(_(`step${s}`, n, r.out, r.in, r.act === "none" ? 0 : Y(r.act), [
				o,
				`w:${r.name}.weight`,
				r.bias ? `w:${r.name}.bias` : "zero",
				d
			], a)), M.push({
				id: d,
				elements: n * r.out,
				final: u
			}), o = d, c = r.out;
		}), !t.steps.length) {
			if (e) throw Error("a token head needs at least one dense step");
			M.find((e) => e.id === "states").final = !0;
		}
		I = o, N = n, F = c, y.push({
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
						A,
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
		if (A !== "tmp") throw Error("julia head expects a final norm");
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
				dispatch: [P(u, s), 1]
			}],
			capture: x("tmp")
		}), j = {
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
				ops: te(h, i, {
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
	let R = y.flatMap((e) => [...e.captureOps ?? [], ...e.ops]), z = /* @__PURE__ */ new Map();
	for (let e of R) {
		let t = e.bind[J(e.kernel, e.bind.length)[0]], n = ue(e);
		n > 0 && z.set(t, Math.max(z.get(t) ?? 0, n));
	}
	let B = /* @__PURE__ */ new Set([
		"states",
		"kNormed",
		"h1",
		"raw"
	]), V = [], H = (e, t, n = "rw", r) => {
		V.push(r ? {
			id: e,
			bytes: t,
			usage: n,
			init: r
		} : {
			id: e,
			bytes: t,
			usage: n
		});
	}, U = (e, t = "rw") => {
		let n = z.get(e);
		n !== void 0 && H(e, l((B.has(e) ? p : u) * n), t);
	};
	H("emb", l(u * c)), e.embed.project && H("embE", l(u * c)), (e.embed.positions === "absolute" || e.embed.typeVocab > 0) && H("typeIds", u * 4);
	for (let e of ["x", "tmp"]) z.has(e) && H(e, l(u * z.get(e)), "rwSrc");
	T && (H("cols", l(u * T.kernel * s)), H("convOut", l(u * s)));
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
	]) d && B.has(e) || U(e);
	H("mask", u * 4), R.some((e) => e.bind.includes("kinfo")) && H("kinfo", 8 * i);
	let W = R.filter((e) => e.kernel === "attscore");
	for (let e of ["c2p", "p2c"]) {
		let t = R.filter((t) => t.kernel === "attrel" && t.bind[4] === e);
		t.length && H(e, Math.max(...t.map((e) => e.constants.ROWS * e.constants.H * e.constants.NM * 4)));
	}
	if (W.length && H("scores", Math.max(...W.map((e) => e.constants.ROWS * e.constants.H * e.constants.L * 4))), R.some((e) => e.bind.includes("dummy")) && H("dummy", 4, "storage"), d) {
		for (let e of M) H(e.id, l(e.elements), e.final ? "logits" : "rw");
		L && H("first", 3 * i * 4, "rw", new Uint32Array(3 * i));
	} else H("packed", 3 * p * 4), H("logits", p * 4, "logits");
	if (E) {
		let e = se(r, E.buckets, E.maxPositions);
		H("relidx", e.byteLength, "rw", e);
	}
	if (O && !k) {
		let t = q(r, O.thetaGlobal, e.headDim);
		H("cossin", t.byteLength, "rw", t);
	} else if (O) {
		let t = q(r, O.thetaGlobal, e.headDim), n = q(r, O.thetaLocal, e.headDim);
		H("cossinG", t.byteLength, "rw", t), H("cossinL", n.byteLength, "rw", n);
	}
	j && H("typeRow", l(i * s));
	let G = R.filter((e) => e.bind.includes("zero"));
	if (G.length) {
		let e = Math.max(...G.map((e) => e.constants.N));
		H("zero", l(e), "rw", new Uint8Array(l(e)));
	}
	return ce(y, new Set(V.map((e) => e.id))), {
		f16: a,
		length: r,
		batch: i,
		markers: f,
		embeddingSize: c,
		buffers: V,
		segments: y,
		inputs: d ? {
			embeddings: "emb",
			mask: "mask",
			...e.embed.positions === "absolute" || e.embed.typeVocab > 0 ? { typeIds: "typeIds" } : {}
		} : {
			embeddings: "emb",
			mask: "mask",
			markers: "packed"
		},
		rowSelect: j,
		output: d ? {
			buffer: I,
			bytes: l(N * F),
			dtype: "storage",
			rows: N,
			cols: F
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
//#region src/errors.ts
var X = class extends Error {
	constructor(e) {
		super(e), this.name = "BucketOverflowError";
	}
};
//#endregion
//#region src/tasks.ts
function Z(e, t) {
	let n = 0;
	for (let t = 0; t < e.length; t += 1) Number.isFinite(e[t]) || (n += 1);
	if (n) throw Error(`${t}: ${n} of ${e.length} values are not finite`);
}
function Q(e) {
	let t = new Float32Array(e.length), n = -Infinity;
	for (let t = 0; t < e.length; t += 1) n = Math.max(n, e[t]);
	let r = 0;
	for (let i = 0; i < e.length; i += 1) t[i] = Math.exp(e[i] - n), r += t[i];
	for (let e = 0; e < t.length; e += 1) t[e] /= r;
	return t;
}
function fe(e) {
	let t = 0;
	for (let n = 0; n < e.length; n += 1) t += e[n] * e[n];
	let n = Math.max(Math.sqrt(t), 1e-12), r = new Float32Array(e.length);
	for (let t = 0; t < e.length; t += 1) r[t] = e[t] / n;
	return r;
}
function pe(e, t) {
	let n = Array.from({ length: t }, (e, t) => `LABEL_${t}`);
	for (let [r, i] of Object.entries(e ?? {})) {
		let e = Number(r);
		if (!Number.isInteger(e) || String(e) !== r || e < 0 || e >= t) throw Error(`label id '${r}' is not an integer from 0 to ${t - 1}`);
		n[e] = i;
	}
	return n;
}
function $(e, t) {
	if (!Number.isInteger(t) || t <= 0) throw Error(`argmaxRows needs a positive integer column count, got ${t}`);
	let n = Math.floor(e.length / t), r = [];
	for (let i = 0; i < n; i += 1) {
		let n = 0;
		for (let r = 1; r < t; r += 1) e[i * t + r] > e[i * t + n] && (n = r);
		r.push(n);
	}
	return r;
}
function me(e) {
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
function he(e, t, n, r, i = ["O"]) {
	let a = t.length, o = [], s = $(e, a);
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
			let t = me(e.label), n = me(l[l.length - 1].label);
			(t.tag !== n.tag || t.begin) && (u(), l = []);
		}
		l.push(e);
	}
	return u(), c.filter((e) => !i.includes(e.group));
}
function ge(e, t, n) {
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
//#endregion
//#region src/encoder.ts
var _e = 16, ve = [128, 512], ye = 8, be = () => import("./tokenizer.js");
async function xe(e) {
	let t, n;
	try {
		let [r, i] = await Promise.all([be(), fetch(e)]);
		if (n = r.JsonTokenizer, !i.ok) throw Error(`fetch ${e}: ${i.status}`);
		t = await i.arrayBuffer();
	} catch (e) {
		return {
			bytes: 0,
			error: String(e instanceof Error ? e.message : e)
		};
	}
	try {
		return {
			bytes: t.byteLength,
			tokenizer: await n.fromBytes(new Uint8Array(t))
		};
	} catch (e) {
		return {
			bytes: t.byteLength,
			error: String(e instanceof Error ? e.message : e)
		};
	}
}
var Se = class e {
	backend;
	words;
	spec;
	head;
	task;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	batchFit = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	loadTiming = {};
	tokenizer = null;
	tokenizerError = null;
	precisionNote;
	wordsMode = "full";
	recommendedPrecision;
	labels = [];
	textMaxLength = 0;
	constructor(e, t, n, r, i, a) {
		this.backend = e, this.words = t, this.spec = n, this.head = r, this.task = i, this.precision = a;
	}
	static async load(t, n = {}) {
		let r = t.backend === "wasm";
		if (r && t.precision === "f16") throw Error("the WASM path runs the f32 manifest");
		r || m().catch(() => void 0);
		let i = t.precision !== "f32", a = performance.now(), o = r ? f(t.manifestUrl) : t.manifestUrl, s = l(o), c = r ? void 0 : await E(i, t.limits !== "default"), d = await u(o, r ? "f32" : t.precision, await s), p = d.manifest, g = d.url, _ = performance.now() - a;
		if (p.format !== "kleinhirn-weights-2" || !p.spec) throw Error(`EncoderModel needs kleinhirn-weights-2, manifest is ${p.format}`);
		let v = p.words?.dtype ?? p.tensors[0]?.dtype ?? "f32";
		if (c && v === "f16" && !c.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (r && v !== "f32") throw Error(`the WASM path needs an f32 manifest, ${g} holds ${v}`);
		if (t.precision && t.precision !== "auto" && t.precision !== v) throw Error(`precision ${t.precision} requested but manifest is ${v}`);
		let y = g.slice(0, g.lastIndexOf("/") + 1), b = p.tokenizer ? xe(y + p.tokenizer) : void 0, x = await h(g, p, c, n.wasmTransport, {
			build: t.wasmBuild,
			threads: t.threads,
			split: t.wasmSplit,
			batchPlans: t.wasmBatchPlans
		}, t.words), { backend: S, words: C } = x, w = p.spec, T = new e(S, C, w, p.head, p.task ?? "", v);
		T.precisionNote = d.note, T.wordsMode = t.words ?? "full", T.recommendedPrecision = p.recommendedPrecision;
		let D = await b;
		D?.error === void 0 ? D && (T.tokenizer = D.tokenizer) : T.tokenizerError = D.error;
		let O = T.head;
		T.labels = O.type === "classify" || O.type === "token" ? pe(p.labels, O.classes) : Object.entries(p.labels ?? {}).sort((e, t) => Number(e[0]) - Number(t[0])).map(([, e]) => e);
		let k = p.sentenceTransformers, A = Math.max(...t.buckets ?? ve);
		T.textMaxLength = t.maxLength ?? Math.min(p.maxLength ?? Math.min(w.embed.maxPositions - w.embed.positionOffset, k?.maxSeqLength ?? Infinity), A), T.downloadBytes = x.downloadBytes + (D?.bytes ?? 0), T.loadTiming = {
			...x.timing,
			manifestMs: _,
			planMs: 0
		};
		let j = performance.now();
		return await S.prepare(() => {
			for (let e of t.buckets ?? ve) {
				let t = T.planFor(e, 1);
				S.check(t), T.plans.set(e, S.runner(t));
			}
		}), T.loadTiming.planMs = performance.now() - j, T;
	}
	planFor(e, t) {
		return de(this.spec, this.head, {
			length: e,
			batch: t,
			markers: 0,
			f16: this.precision === "f16"
		});
	}
	makePlan(e, t) {
		return this.backend.runner(this.planFor(e, t));
	}
	pickBucket(e) {
		let t = [...this.plans.values()].filter((t) => e <= t.length).sort((e, t) => e.length - t.length);
		if (!t.length) throw new X(`seqLen ${e} exceeds the loaded buckets`);
		return t[0];
	}
	enqueue(e) {
		let t = this.queue.then(e);
		return this.queue = t.catch(() => {}), t;
	}
	wordRows(e, t) {
		let n = this.spec.embeddingSize, r = this.words.alloc(t * e.length * n);
		for (let [i, a] of e.entries()) for (let e = 0; e < a.inputIds.length; e += 1) this.words.copyRow(a.inputIds[e], r, (i * t + e) * n);
		return r;
	}
	maskAndTypes(e, t) {
		let n = new Float32Array(t * e.length), r = new Uint32Array(t * e.length);
		for (let [i, a] of e.entries()) {
			if (n.fill(1, i * t, i * t + a.inputIds.length), a.typeIds && this.spec.embed.typeVocab > 0) for (let e = 0; e < a.inputIds.length; e += 1) {
				let n = a.typeIds[e] ?? 0;
				if (!Number.isInteger(n) || n < 0 || n >= this.spec.embed.typeVocab) throw Error(`token type ${n} is not an integer inside the ${this.spec.embed.typeVocab} type rows`);
				r[i * t + e] = n;
			}
			let { padId: e, maxPositions: o } = this.spec.embed;
			if (e !== void 0) {
				let n = H(e, a.inputIds);
				for (let a = 0; a < t; a += 1) {
					let s = a < n.length ? n[a] : e;
					if (s >= o || s > 65535) throw Error(`position row ${s} of id ${a} is outside the ${o} position rows`);
					r[i * t + a] |= s << 16;
				}
			}
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
	async runIds(e, t = {}) {
		let n = e.inputIds.length;
		if (n === 0) throw Error("empty input");
		this.checkPositions(n);
		let r = t.bucket === void 0 ? this.pickBucket(n) : this.plans.get(t.bucket);
		if (!r) throw Error(`bucket ${t.bucket} not loaded`);
		if (n > r.length) throw new X(`seqLen ${n} exceeds bucket ${r.length}`);
		t.capture && r.assertCapturable();
		let i = a(this.words, [e]);
		return i && await i, this.enqueue(() => this.backend.call(() => {
			let { mask: i, typeIds: a } = this.maskAndTypes([e], r.length);
			r.upload({
				embeddings: this.wordRows([e], r.length),
				mask: i,
				typeIds: a
			}), r.run(n, !!t.capture);
		}, async () => {
			let e = await r.readOutput(), i = this.shape(e, 0, n, r);
			return Z(i.data, "engine output"), t.capture && (i.capture = await r.readCapture(), i.captureSlotElements = r.length * this.spec.hidden), i;
		}));
	}
	batchPlan(e, t) {
		let n = `${e}:${t}`, r = this.batchFit.get(n);
		if (r === 1) return;
		let i = this.batchPlans.get(`${e}:${r ?? t}`);
		if (i) return i;
		let a = r === void 0 ? this.backend.fitBatch((t) => this.planFor(e, t), t) : this.planFor(e, r);
		if (this.batchFit.set(n, a ? a.batch : 1), !a) return;
		if (this.batchPlans.size >= ye) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let o = this.backend.runner(a);
		return this.batchPlans.set(`${e}:${a.batch}`, o), o;
	}
	async runChunk(e, t) {
		let n = Math.max(...e.map((e) => e.inputIds.length));
		if (n === 0) throw Error("empty input");
		this.checkPositions(n);
		let r = t === void 0 ? this.pickBucket(n) : this.plans.get(t);
		if (!r) throw Error(`bucket ${t} not loaded`);
		if (n > r.length) throw new X(`batch exceeds bucket ${r.length}`);
		let i = v(e.length), o = Math.min(r.length, S(n)), s = a(this.words, e);
		return s && await s, this.enqueue(async () => {
			let t = [], n;
			for (; t.length < e.length;) {
				let a = await this.backend.call(() => {
					n ??= (i === 1 ? void 0 : this.batchPlan(o, i)) ?? r;
					let a = e.slice(t.length, t.length + n.batch);
					return this.submitPiece(a, n), {
						rows: a,
						plan: n
					};
				}, (e) => this.readPiece(e.rows, e.plan));
				t.push(...a);
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
		}), t.run(r);
	}
	async readPiece(e, t) {
		let n = await t.readOutput();
		return e.map((e, r) => {
			let i = this.shape(n, r, e.inputIds.length, t);
			return Z(i.data, "engine output"), i;
		});
	}
	async runIdsBatch(e, t = {}) {
		for (let [t, n] of e.entries()) if (n.inputIds.length === 0) throw Error(`empty input (input ${t})`);
		let n = e.map((e, t) => t).sort((t, n) => e[t].inputIds.length - e[n].inputIds.length || t - n), r = Array(e.length);
		for (let i = 0; i < n.length; i += _) {
			let a = n.slice(i, i + _), o = await this.runChunk(a.map((t) => e[t]), t.bucket);
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
			let e = $(n, n.length)[0];
			return {
				label: this.labels[e] ?? String(e),
				index: e,
				score: n[e],
				scores: n,
				logits: n
			};
		}
		let a;
		a = i === "multi" ? Float32Array.from(n, (e) => 1 / (1 + Math.exp(-e))) : Q(n);
		let o = $(i === "multi" ? n : a, a.length)[0];
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
		let i = n.template ?? "This example is {}.", a = await this.runIdsBatch(t.map((t) => this.textInput(e, i.replace("{}", () => t), "only_first"))), o = Float32Array.from(a, (e) => e.data[r]), s = Q(o), c = $(s, s.length)[0];
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
		for (let r = 0; r < t.length; r += _e) {
			let i = await this.runIdsBatch(t.slice(r, r + _e).map((t) => this.textInput(e, t)));
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
		return r.map((e) => i ? fe(e.data) : e.data);
	}
	async tokenClassify(e) {
		this.requireHead("token");
		let t = this.requireTokenizer().encode(e, null, { maxLength: this.textMaxLength }), n = await this.runIds({
			inputIds: t.ids,
			typeIds: t.typeIds
		}), r = n.cols, i = new Float32Array(n.rows * r);
		for (let e = 0; e < n.rows; e += 1) i.set(Q(n.data.subarray(e * r, (e + 1) * r)), e * r);
		return he(i, this.labels, t.offsets, t.specialTokensMask).map((t) => ({
			...t,
			text: e.slice(t.start, t.end)
		}));
	}
	liveGpuBytes() {
		let e = this.backend.weightBytes;
		for (let t of this.plans.values()) e += t.bytes;
		for (let t of this.batchPlans.values()) e += t.bytes;
		return e;
	}
	info() {
		return {
			precision: this.precision,
			...this.recommendedPrecision ? { recommendedPrecision: this.recommendedPrecision } : {},
			...this.precisionNote ? { precisionNote: this.precisionNote } : {},
			buildId: "mv690uvg",
			...this.backend.info(),
			family: this.spec.family,
			task: this.task,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			textMaxLength: this.textMaxLength,
			hasTokenizer: this.tokenizer !== null,
			words: r(this.words, this.wordsMode),
			batchSizes: [...g],
			gpuBytes: this.backend.kind === "webgpu" ? this.liveGpuBytes() : null,
			downloadBytes: this.downloadBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.backend.dispose();
	}
}, Ce = () => import("./gliner-BsX8tg3h.js"), we = () => import("./julia-ELHMs-3J.js"), Te = { load: async (e, t) => (e.backend !== "wasm" && m().catch(() => void 0), (await Ce()).Kleinhirn.load(e, t)) };
async function Ee(e) {
	let t = e.backend ?? "auto", n = await l(e.manifestUrl), r = async (e) => Oe(n) ? (await we()).JuliaEngine.load(e, n) : ke(n) ? Se.load(De(e)) : Te.load(e), i = {
		...e,
		backend: "wasm",
		precision: "f32"
	};
	if (t === "wasm") return r(i);
	if (t === "webgpu") return r(e);
	if (typeof navigator < "u" && navigator.gpu) try {
		return await r({
			...e,
			backend: "webgpu"
		});
	} catch {}
	return r(i);
}
function De(e) {
	return {
		manifestUrl: e.manifestUrl,
		limits: e.limits,
		precision: e.precision,
		backend: e.backend === "wasm" ? "wasm" : "webgpu",
		buckets: e.buckets?.map((e) => typeof e == "number" ? e : e.length)
	};
}
function Oe(e) {
	return e.encoder?.arch === "modernbert-julia";
}
function ke(e) {
	return e.format === "kleinhirn-weights-2";
}
//#endregion
export { X as BucketOverflowError, o as C, Se as EncoderModel, Te as Kleinhirn, d as S, r as T, C as _, T as a, m as b, k as c, _ as d, S as f, b as g, ge as groupSoftmax, v as h, oe as i, g as l, Ee as loadEngine, x as m, de as n, E as o, w as p, ae as r, O as s, Z as t, y as u, f as v, a as w, l as x, h as y };
