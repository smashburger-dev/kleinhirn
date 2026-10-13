import { BucketOverflowError as e, T as t, _ as n, b as r, d as i, f as a, g as o, groupSoftmax as s, h as c, n as l, o as u, p as d, r as f, t as p, u as m, v as h, w as g, x as _, y as v } from "./kleinhirn.js";
import { a as y, s as b, t as x } from "./pairs-BUwAo4kH.js";
//#region src/tokenizer/unigram.ts
var S = 10;
function C(e, t) {
	let n = Float64Array.from(e.scores), { blob: r, offs: i } = e.pieces, a = e.pieces.length, o = 1;
	for (let e = 0; e < a; e += 1) o += i[e + 1] - i[e];
	let s = new Int32Array(o).fill(-1), c = new x(0), l = 1, u = Infinity;
	for (let e = 0; e < a; e += 1) {
		let t = 0;
		for (let n = i[e]; n < i[e + 1];) {
			let e = r.codePointAt(n);
			n += e > 65535 ? 2 : 1;
			let i = c.get(t, e);
			i === -1 && (i = l, l += 1, c.set(t, e, i)), t = i;
		}
		s[t] = e, n[e] < u && (u = n[e]);
	}
	return {
		edges: c,
		valueOf: s,
		scores: n,
		unkScore: u - S,
		unkId: t
	};
}
function w(e, t) {
	let n = [...t], r = n.length;
	if (r === 0) return [];
	let i = Array(r + 1).fill(-Infinity), a = Array(r + 1).fill(-1), o = Array(r + 1).fill(0);
	i[0] = 0;
	for (let t = 0; t < r; t += 1) {
		if (i[t] === -Infinity) continue;
		let s = 0, c = !1;
		for (let l = t; l < r; l += 1) {
			let r = e.edges.get(s, n[l].codePointAt(0));
			if (r === -1) break;
			s = r;
			let u = e.valueOf[s];
			if (u >= 0) {
				l === t && (c = !0);
				let n = i[t] + e.scores[u];
				n > i[l + 1] && (i[l + 1] = n, a[l + 1] = u, o[l + 1] = t);
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
var T = /\s{2,}|[\n\r\t]/g, E = "▁", D = class {
	unigram;
	added;
	addedList;
	padTokenId = 0;
	constructor(e) {
		this.unigram = C(y(e.model.vocab), e.model.unk_id ?? 3), this.added = /* @__PURE__ */ new Map(), this.addedList = [];
		for (let t of e.added_tokens) this.added.set(t.content, t.id), this.addedList.push(t.content), t.content === "[PAD]" && (this.padTokenId = t.id);
		this.addedList.sort((e, t) => t.length - e.length);
	}
	normalize(e) {
		return e.replace(T, " ").normalize("NFC").replace(/\s+$/, "");
	}
	encodePretoken(e) {
		if (this.added.has(e)) return [this.added.get(e)];
		let t = [], n = 0;
		for (let r = 0; r <= e.length; r += 1) (r < e.length && e[r] === " " || r === e.length) && (r > n && t.push(...w(this.unigram, E + e.slice(n, r))), n = r + 1);
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
}, O = "[SEP_STRUCT]", k = "[SEP_TEXT]", A = "[P]", j = "[L]", M = /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]/giu;
function N(e) {
	let t = [];
	M.lastIndex = 0;
	for (let n of e.matchAll(M)) t.push(n[0].toLowerCase());
	return t;
}
function P(t, n, r, i, a) {
	let o = r.reduce((e, [, t]) => e + t.length, 0);
	if (o < 1 || o > a) throw new e(`expected 1..${a} labels in total, got ${o}`);
	let s = n;
	s && !/[.!?]$/.test(s) && (s += "."), s ||= ".";
	let c = N(s), l = r.map(([e, t]) => [
		"(",
		A,
		e,
		"(",
		...t.flatMap((e) => [j, e]),
		")",
		")"
	]), u = [];
	for (let e of l) u.push(...e, O);
	u.pop(), u.push(k, ...c);
	let d = /* @__PURE__ */ new Set(), f = /* @__PURE__ */ new Set(), p = 0;
	for (let e of l) {
		e.length > 1 && (d.add(p + 1), f.add(p + 1));
		for (let t = 4; t < e.length - 2; t += 2) d.add(p + t);
		p += e.length + 1;
	}
	let m = [], h = [], g = 0, _ = !1;
	for (let e = 0; e < u.length; e += 1) {
		let n = u[e], r = !_;
		n === k ? _ = !0 : n === O && (g += 1);
		let i = m.length;
		m.push(...t.encodeIds(n)), r && !f.has(e) && d.has(e) && h.push({
			pos: i,
			group: g
		});
	}
	let v = m.length;
	if (v > i || h.length > a) throw new e(`input exceeds bucket: seqLen ${v} > ${i} or markers > ${a}`);
	return F(h, m, v, i, a, t.padTokenId);
}
function F(e, t, n, r, i, a) {
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
//#region src/gliner.ts
var I = 16, L = 8, R = class y {
	backend;
	words;
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
	wordsMode = "full";
	loadTiming = {};
	cache = new m(0);
	headHidden = 768;
	constructor(e, t, n, r, i, a) {
		this.backend = e, this.words = t, this.tokenizer = n, this.spec = r, this.temperature = i, this.precision = a;
	}
	static async load(e, t = {}) {
		let n = e.backend === "wasm";
		if (n && e.precision === "f16") throw Error("the WASM path runs the f32 manifest");
		n || r().catch(() => void 0);
		let i = e.precision !== "f32", a = n ? h(e.manifestUrl) : e.manifestUrl, o = a.slice(0, a.lastIndexOf("/") + 1), s = performance.now(), c = _(a), l = n ? void 0 : await u(i, e.limits !== "default"), d = await c, f = performance.now() - s, p = (async () => (await fetch(o + d.tokenizer)).arrayBuffer())(), g = d.tensors.find((e) => e.name.endsWith("qkv.weight"))?.dtype ?? "f32";
		if (l && g === "f16" && !l.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n && g !== "f32") throw Error(`the WASM path needs an f32 manifest, ${a} holds ${g}`);
		if (e.precision && e.precision !== "auto" && e.precision !== g) throw Error(`precision ${e.precision} requested but manifest is ${g}`);
		let x = await v(a, d, l, t.wasmTransport, void 0, e.words), { backend: S } = x, C = g, w = performance.now(), T = await p, E = new D(await b(new Uint8Array(T))), O = performance.now() - w, k = d.encoder, A = new y(S, x.words, E, k, d.head.temperature, C);
		A.downloadBytes = x.downloadBytes + T.byteLength, A.tokenizerBytes = T.byteLength, A.wordsMode = e.words ?? "full", A.loadTiming = {
			...x.timing,
			manifestMs: f,
			tokenizerMs: O,
			planMs: 0
		}, A.headHidden = Number(d.head.hiddenSize) || 768, A.cache = new m(e.cacheSize ?? 256);
		let j = performance.now();
		return await S.prepare(() => {
			for (let t of e.buckets ?? [128]) {
				let e = typeof t == "number" ? {
					length: t,
					markers: I
				} : {
					length: t.length,
					markers: t.markers ?? I
				}, n = A.planFor(e.length, e.markers, 1);
				S.check(n), A.plans.set(e.length, S.runner(n));
			}
		}), A.loadTiming.planMs = performance.now() - j, A;
	}
	planFor(e, t, n) {
		let { spec: r, head: i } = f(this.spec, {
			temperature: this.temperature,
			hiddenSize: this.headHidden
		}, t);
		return l(r, i, {
			length: e,
			batch: n,
			markers: t,
			f16: this.precision === "f16"
		});
	}
	pickBucket(t, n = 1) {
		let r = [...this.plans.values()].filter((e) => t <= e.length && n <= e.markers).sort((e, t) => e.length - t.length || e.markers - t.markers);
		if (!r.length) throw new e(`seqLen ${t} or ${n} markers exceed loaded buckets`);
		return r[0];
	}
	embeddingRows(e, t) {
		let n = this.spec.hiddenSize, r = this.words.alloc(t.length * n);
		for (let t = 0; t < e.seqLen; t += 1) this.words.copyRow(e.inputIds[t], r, t * n);
		return r;
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
	async runPrepared(t, n = !1, r) {
		let i = t.markerMask.reduce((e, t) => e + +(t > .5), 0), a = r === void 0 ? this.pickBucket(t.seqLen, i) : this.plans.get(r);
		if (!a) throw Error(`bucket ${r} not loaded`);
		if (t.seqLen > a.length || i > a.markers) throw new e(`seqLen ${t.seqLen}/${i} markers exceeds bucket ${a.length}/${a.markers}`);
		n && a.assertCapturable();
		let c = o(t), l = n ? void 0 : this.cache.get(c);
		if (l) return z(l);
		let u = g(this.words, [t]);
		return u && await u, this.enqueue(() => this.backend.call(() => {
			a.upload({
				embeddings: this.embeddingRows(t, a),
				mask: this.maskOf(t, a.length),
				packedMarkers: this.packedMarkers(t, a.markers)
			}), a.run(t.seqLen, n);
		}, async () => {
			let e = await a.readOutput();
			p(e, "logits");
			let r = s(e, t.markerGroups, t.markerMask);
			return n || this.cache.set(c, z({
				logits: e,
				probabilities: r
			})), {
				logits: e,
				probabilities: r,
				captureData: n ? await a.readCapture() : void 0
			};
		}));
	}
	batchPlan(e, t, n) {
		let r = `${e}:${t}:${n}`, i = this.batchFit.get(r);
		if (i === 1) return;
		let a = this.batchPlans.get(`${e}:${t}:${i ?? n}`);
		if (a) return a;
		let o = i === void 0 ? this.backend.fitBatch((n) => this.planFor(e, t, n), n) : this.planFor(e, t, i);
		if (this.batchFit.set(r, o ? o.batch : 1), !o) return;
		if (this.batchPlans.size >= L) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		let s = this.backend.runner(o);
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
	batchPackedMarkers(e, t) {
		let n = new Uint32Array(e.length * 3 * t.markers);
		for (let [r, i] of e.entries()) n.set(this.packedMarkers(i, t.markers), r * 3 * t.markers);
		return n;
	}
	async runBatchChunk(t, n) {
		let r = Math.max(...t.map((e) => e.seqLen)), i = Math.max(...t.map((e) => e.markerMask.reduce((e, t) => e + +(t > .5), 0))), o = n === void 0 ? this.pickBucket(r, i) : this.plans.get(n);
		if (!o) throw Error(`bucket ${n} not loaded`);
		if (r > o.length || i > o.markers) throw new e(`batch exceeds bucket ${o.length}/${o.markers}`);
		let s = c(t.length), l = Math.min(o.length, a(r)), u = g(this.words, t);
		return u && await u, this.enqueue(async () => {
			let e = [], n;
			for (; e.length < t.length;) {
				let r = await this.backend.call(() => {
					n ??= (s === 1 ? void 0 : this.batchPlan(l, o.markers, s)) ?? o;
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
			packedMarkers: this.batchPackedMarkers(n, t)
		}), t.run(r);
	}
	async readPiece(e, t) {
		let n = await t.readOutput();
		return e.map((e, r) => {
			let i = n.slice(r * t.markers, (r + 1) * t.markers);
			p(i, "logits");
			let a = {
				logits: i,
				probabilities: s(i, e.markerGroups, e.markerMask)
			};
			return this.cache.set(o(e), z(a)), a;
		});
	}
	async runPreparedBatch(e, t) {
		let r = Array(e.length), a = [], s = [];
		for (let [t, n] of e.entries()) {
			let e = this.cache.get(o(n));
			e ? r[t] = z(e) : (a.push(n), s.push(t));
		}
		let { unique: c, slot: l } = d(a, o), u = n(c, l, (e) => e.seqLen), f = [];
		for (let e = 0; e < u.unique.length; e += i) {
			let n = await this.runBatchChunk(u.unique.slice(e, e + i), t);
			f.push(...n);
		}
		let p = /* @__PURE__ */ new Set();
		for (let [e] of a.entries()) {
			let t = f[u.slot[e]];
			r[s[e]] = p.has(u.slot[e]) ? z(t) : t, p.add(u.slot[e]);
		}
		return r;
	}
	async classify(e, t) {
		let n = performance.now(), r = t.map((e) => [e.task, e.labels]), i = r.reduce((e, [, t]) => e + t.length, 0), a = P(this.tokenizer, e, r, this.maxBucket(i), i), o = performance.now() - n, s = performance.now(), { logits: c, probabilities: l } = await this.runPrepared(a), u = performance.now() - s;
		return this.toResult(t, a, c, l, {
			tokenizeMs: o,
			gpuMs: u,
			totalMs: performance.now() - n
		});
	}
	async classifyBatch(e) {
		let t = performance.now(), n = e.map((e) => {
			let t = e.tasks.map((e) => [e.task, e.labels]), n = t.reduce((e, [, t]) => e + t.length, 0);
			return P(this.tokenizer, e.text, t, this.maxBucket(n), n);
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
					let o = B(t.markerGroups, a, i);
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
	maxBucket(t = 1) {
		let n = [...this.plans.values()].filter((e) => e.markers >= t).map((e) => e.length);
		if (!n.length) throw new e(`no bucket routes ${t} markers`);
		return Math.max(...n);
	}
	async profile(e, t = {}) {
		let n = await this.profileDetailed(e, t);
		return n && n.times;
	}
	async profileDetailed(e, t = {}) {
		let n = e.markerMask.reduce((e, t) => e + +(t > .5), 0), i = this.pickBucket(e.seqLen, n), { PlanExecutor: a } = await r();
		if (!(i instanceof a)) throw Error("GPU profiling needs the WebGPU path");
		let o = g(this.words, [e]);
		return o && await o, this.enqueue(() => i.profileForward({
			embeddings: this.embeddingRows(e, i),
			mask: this.maskOf(e, i.length),
			packedMarkers: this.packedMarkers(e, i.markers)
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
			buildId: "mv690uvg",
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
function z(e) {
	return {
		logits: e.logits.slice(),
		probabilities: e.probabilities.slice()
	};
}
function B(e, t, n) {
	let r = 0;
	for (let i = 0; i < e.length; i += 1) if (e[i] === t) {
		if (r === n) return i;
		r += 1;
	}
	return -1;
}
//#endregion
export { R as Kleinhirn };
