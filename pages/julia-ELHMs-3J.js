import { BucketOverflowError as e, T as t, _ as n, b as r, d as i, f as a, h as o, i as s, m as c, n as l, o as u, p as d, t as f, u as p, v as m, w as h, x as g, y as _ } from "./kleinhirn.js";
import { i as v, n as y, o as b, r as x, s as S, t as C } from "./pairs-BUwAo4kH.js";
//#region src/tokenizer/bpe.ts
var w = "▁", T = -1, E = -2;
function D(e) {
	let t = [];
	for (let n of new TextEncoder().encode(e)) t.push(`<0x${n.toString(16).toUpperCase().padStart(2, "0")}>`);
	return t;
}
var O = class {
	vocab;
	ranks;
	pairs;
	mergedIds;
	added;
	addedList;
	unkId;
	padTokenId = 0;
	bosTokenId = 2;
	eosTokenId = 1;
	maskTokenId = 4;
	maskToken = "<mask>";
	constructor(e) {
		let t = x(e.model.vocab);
		this.vocab = new y(t), this.pairs = v(t, e.model.merges);
		let n = this.pairs.length / 2;
		this.ranks = new C(n), this.mergedIds = new Int32Array(n).fill(E);
		for (let e = 0; e < n; e += 1) this.ranks.set(b(t, this.pairs[2 * e]), b(t, this.pairs[2 * e + 1]), e);
		this.unkId = this.vocab.id(e.model.unk_token ?? "<unk>") ?? 3, this.added = /* @__PURE__ */ new Map(), this.addedList = [];
		for (let t of e.added_tokens) this.added.set(t.content, t.id), this.addedList.push(t.content), t.content === "<pad>" ? this.padTokenId = t.id : t.content === "<bos>" ? this.bosTokenId = t.id : t.content === "<eos>" ? this.eosTokenId = t.id : t.content === "<mask>" && (this.maskTokenId = t.id);
		this.addedList.sort((e, t) => t.length - e.length);
	}
	mergedOf(e) {
		if (this.mergedIds[e] === E) {
			let t = this.vocab.idOfConcat(this.pairs[2 * e], this.pairs[2 * e + 1], 0);
			this.mergedIds[e] = t === void 0 ? T : t;
		}
		return this.mergedIds[e];
	}
	bpe(e) {
		let t = [];
		for (let n of e) {
			let e = this.vocab.idOfChar(n);
			if (e !== void 0) t.push(e);
			else for (let e of D(n)) t.push(this.vocab.id(e) ?? T);
		}
		for (;;) {
			let e = -1, n = T, r = T;
			for (let i = 0; i + 1 < t.length; i += 1) {
				if (t[i] === T || t[i + 1] === T) continue;
				let a = this.ranks.get(t[i], t[i + 1]);
				a !== -1 && (e === -1 || a < e) && (e = a, n = t[i], r = t[i + 1]);
			}
			if (e === -1) break;
			let i = this.mergedOf(e), a = [];
			for (let e = 0; e < t.length; e += 1) e + 1 < t.length && t[e] === n && t[e + 1] === r ? (a.push(i), e += 1) : a.push(t[e]);
			t = a;
		}
		return t;
	}
	encodePlain(e) {
		let t = e.replaceAll(" ", w);
		t.startsWith(w) || (t = w + t);
		let n = [];
		for (let e of t.matchAll(/▁[^▁]*/g)) for (let t of this.bpe(e[0])) n.push(t === T ? this.unkId : t);
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
		return this.vocab.tokenOfId(e);
	}
}, k = {
	choice: 0,
	score: 1,
	noul: 2
};
function A(e, t, n = 1024, r = 256, i = !1) {
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
		qtype: k[s] ?? 0,
		seqLen: f.length,
		truncated: g
	};
}
//#endregion
//#region src/julia.ts
function j(e) {
	return {
		logits: e.logits.slice(),
		probabilities: e.probabilities.slice()
	};
}
function M(e, t) {
	let n = new Float32Array(t), r = -Infinity;
	for (let n = 0; n < t; n += 1) r = Math.max(r, e[n]);
	let i = 0;
	for (let n = 0; n < t; n += 1) i += Math.exp(e[n] - r);
	for (let a = 0; a < t; a += 1) n[a] = Math.exp(e[a] - r) / i;
	return n;
}
var N = 8, P = class v {
	backend;
	words;
	tokenizer;
	spec;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	batchFit = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	tokenizerBytes = 0;
	wordsMode = "full";
	loadTiming = {};
	cache = new p(0);
	constructor(e, t, n, r, i) {
		this.backend = e, this.words = t, this.tokenizer = n, this.spec = r, this.precision = i;
	}
	static async load(e, t, n = {}) {
		let i = e.backend === "wasm";
		if (i && e.precision === "f16") throw Error("the WASM path runs the f32 manifest");
		i || r().catch(() => void 0);
		let a = e.precision !== "f32", o = i ? m(e.manifestUrl) : e.manifestUrl, s = o.slice(0, o.lastIndexOf("/") + 1), c = performance.now(), l = t && o === e.manifestUrl ? Promise.resolve(t) : g(o), d = i ? void 0 : await u(a, e.limits !== "default"), f = await l, h = performance.now() - c, y = (async () => (await fetch(s + f.tokenizer)).arrayBuffer())(), b = f.tensors.find((e) => e.name.endsWith("wqkv.weight"))?.dtype ?? "f32";
		if (d && b === "f16" && !d.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (i && b !== "f32") throw Error(`the WASM path needs an f32 manifest, ${o} holds ${b}`);
		if (e.precision && e.precision !== "auto" && e.precision !== b) throw Error(`precision ${e.precision} requested but manifest is ${b}`);
		let x = await _(o, f, d, n.wasmTransport, void 0, e.words), { backend: C } = x, w = b, T = performance.now(), E = await y, D = new O(await S(new Uint8Array(E))), k = performance.now() - T, A = f.encoder, j = new v(C, x.words, D, A, w);
		j.downloadBytes = x.downloadBytes + E.byteLength, j.tokenizerBytes = E.byteLength, j.wordsMode = e.words ?? "full", j.loadTiming = {
			...x.timing,
			manifestMs: h,
			tokenizerMs: k,
			planMs: 0
		};
		let M = performance.now();
		return await C.prepare(() => {
			for (let t of e.buckets ?? [512]) {
				let e = typeof t == "number" ? t : t.length, n = j.planFor(e, 1);
				C.check(n), j.plans.set(e, C.runner(n));
			}
		}), j.cache = new p(e.cacheSize ?? 256), j.loadTiming.planMs = performance.now() - M, j;
	}
	planFor(e, t) {
		let { spec: n, head: r } = s(this.spec);
		return l(n, r, {
			length: e,
			batch: t,
			markers: this.spec.options,
			f16: this.precision === "f16"
		});
	}
	pickBucket(t) {
		let n = [...this.plans.keys()].sort((e, t) => e - t).find((e) => t <= e);
		if (n === void 0) throw new e(`seqLen ${t} exceeds largest bucket`);
		return this.plans.get(n);
	}
	embeddingRows(e, t) {
		let n = this.spec.hiddenSize, r = this.words.alloc(t.length * n);
		for (let t = 0; t < e.seqLen; t += 1) this.words.copyRow(e.inputIds[t], r, t * n);
		return r;
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
	async runPrepared(t, n = !1, r) {
		let i = r === void 0 ? this.pickBucket(t.seqLen) : this.plans.get(r);
		if (!i) throw Error(`bucket ${r} not loaded`);
		if (t.seqLen > i.length) throw new e(`seqLen ${t.seqLen} exceeds bucket ${i.length}`);
		n && i.assertCapturable();
		let a = c(t), o = n ? void 0 : this.cache.get(a);
		if (o) return j(o);
		let s = h(this.words, [t]);
		return s && await s, this.enqueue(() => this.backend.call(() => {
			i.upload({
				embeddings: this.embeddingRows(t, i),
				mask: this.maskOf(t, i.length),
				packedMarkers: this.packedMarkers(t),
				qtype: t.qtype
			}), i.run(t.seqLen, n);
		}, async () => {
			let e = await i.readOutput();
			f(e, "logits");
			let r = M(e, Math.min(t.markers.length, this.spec.options));
			return n || this.cache.set(a, j({
				logits: e,
				probabilities: r
			})), {
				logits: e,
				probabilities: r,
				captureData: n ? await i.readCapture() : void 0
			};
		}));
	}
	batchPlan(e, t) {
		let n = `${e}:${t}`, r = this.batchFit.get(n);
		if (r === 1) return;
		let i = this.batchPlans.get(`${e}:${r ?? t}`);
		if (i) return i;
		let a = r === void 0 ? this.backend.fitBatch((t) => this.planFor(e, t), t) : this.planFor(e, r);
		if (this.batchFit.set(n, a ? a.batch : 1), !a) return;
		if (this.batchPlans.size >= N) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let o = this.backend.runner(a);
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
		let n = this.spec.hiddenSize, r = this.words.alloc(t.length * t.batch * n);
		for (let [i, a] of e.entries()) {
			let e = i * t.length * n;
			for (let t = 0; t < a.seqLen; t += 1) this.words.copyRow(a.inputIds[t], r, e + t * n);
		}
		return r;
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
	async runBatchChunk(t, n) {
		let r = Math.max(...t.map((e) => e.seqLen)), i = n === void 0 ? this.pickBucket(r) : this.plans.get(n);
		if (!i) throw Error(`bucket ${n} not loaded`);
		if (r > i.length) throw new e(`batch seqLen ${r} exceeds bucket ${i.length}`);
		let s = o(t.length), c = Math.min(i.length, a(r)), l = h(this.words, t);
		return l && await l, this.enqueue(async () => {
			let e = [], n;
			for (; e.length < t.length;) {
				let r = await this.backend.call(() => {
					n ??= (s === 1 ? void 0 : this.batchPlan(c, s)) ?? i;
					let r = t.slice(e.length, e.length + n.batch);
					return this.submitPiece(r, n), {
						rows: r,
						plan: n
					};
				}, (e) => this.readPiece(e.rows, e.plan));
				e.push(...r);
			}
			return e;
		});
	}
	submitPiece(e, t) {
		let n = e.length === t.batch ? e : [...e, ...Array.from({ length: t.batch - e.length }, () => this.padInput())], r = t.batch === 1 ? Math.max(...e.map((e) => e.seqLen)) : t.length * t.batch;
		t.upload({
			embeddings: this.batchEmbeddingRows(n, t),
			mask: this.batchMask(n, t),
			packedMarkers: this.batchPackedMarkers(n),
			qtype: n.map((e) => e.qtype)
		}), t.run(r);
	}
	async readPiece(e, t) {
		let n = await t.readOutput(), r = this.spec.options;
		return e.map((e, t) => {
			let i = n.slice(t * r, (t + 1) * r);
			f(i, "logits");
			let a = {
				logits: i,
				probabilities: M(i, Math.min(e.markers.length, r))
			};
			return this.cache.set(c(e), j(a)), a;
		});
	}
	async runPreparedBatch(e, t) {
		let r = Array(e.length), a = [], o = [];
		for (let [t, n] of e.entries()) {
			let e = this.cache.get(c(n));
			e ? r[t] = j(e) : (a.push(n), o.push(t));
		}
		let { unique: s, slot: l } = d(a, c), u = n(s, l, (e) => e.seqLen), f = [];
		for (let e = 0; e < u.unique.length; e += i) {
			let n = await this.runBatchChunk(u.unique.slice(e, e + i), t);
			f.push(...n);
		}
		let p = /* @__PURE__ */ new Set();
		for (let [e] of a.entries()) {
			let t = f[u.slot[e]];
			r[o[e]] = p.has(u.slot[e]) ? j(t) : t, p.add(u.slot[e]);
		}
		return r;
	}
	prepare(e, t = 1024, n = 512, r = !1) {
		return A(this.tokenizer, e, t, n, r);
	}
	async decide(e) {
		let t = performance.now(), n = A(this.tokenizer, e, this.maxBucket(), 256), r = performance.now() - t, i = performance.now(), { logits: a, probabilities: o } = await this.runPrepared({
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
			let t = A(this.tokenizer, e, this.maxBucket(), 256);
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
		let n = this.pickBucket(e.seqLen), { PlanExecutor: i } = await r();
		if (!(n instanceof i)) throw Error("GPU profiling needs the WebGPU path");
		let a = h(this.words, [e]);
		return a && await a, this.enqueue(() => n.profileForward({
			embeddings: this.embeddingRows(e, n),
			mask: this.maskOf(e, n.length),
			packedMarkers: this.packedMarkers(e),
			qtype: e.qtype
		}, t.granularity ?? "pass", e.seqLen));
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
			...this.backend.info(),
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			gpuBytes: this.backend.kind === "webgpu" ? this.liveGpuBytes() : null,
			downloadBytes: this.downloadBytes,
			tokenizerBytes: this.tokenizerBytes,
			words: t(this.words, this.wordsMode),
			weightBytes: this.downloadBytes - this.tokenizerBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.backend.dispose();
	}
};
//#endregion
export { P as JuliaEngine };
