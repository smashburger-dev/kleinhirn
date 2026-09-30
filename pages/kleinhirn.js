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
	}), c = r.info ?? {};
	return {
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
		}
	};
}
//#endregion
//#region src/weights.ts
function n(e) {
	return [...new Uint8Array(e)].map((e) => e.toString(16).padStart(2, "0")).join("");
}
async function r(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	let n = performance.now();
	return {
		buf: await t.arrayBuffer(),
		bodyMs: performance.now() - n
	};
}
async function i(e) {
	let t = await fetch(e);
	if (!t.ok) throw Error(`fetch ${e}: ${t.status}`);
	return t.json();
}
async function a(e, t) {
	let a = e.slice(0, e.lastIndexOf("/") + 1), o = t ?? await i(e), s = new TextEncoder().encode(JSON.stringify(o)).byteLength, c = 0, l = 0, u = performance.now(), d = await Promise.all(o.shards.map(async (e) => {
		let { buf: t, bodyMs: i } = await r(a + e.file);
		l += i;
		let o = performance.now(), u = n(await crypto.subtle.digest("SHA-256", t));
		if (c += performance.now() - o, u !== e.sha256) throw Error(`sha256 mismatch on ${e.file}`);
		return s += t.byteLength, t;
	})), f = performance.now() - u;
	return {
		manifest: o,
		shardBytes: d,
		downloadBytes: s,
		fetchMs: f,
		bodyMs: l,
		sha256Ms: c
	};
}
async function o(e, t, n) {
	let { manifest: r, shardBytes: i, downloadBytes: o, fetchMs: s, bodyMs: c, sha256Ms: l } = await a(t, n), u = /* @__PURE__ */ new Map(), d = /* @__PURE__ */ new Map(), f = 0, p = performance.now();
	for (let t of r.tensors) {
		let n = new Uint8Array(i[t.shard], t.offset, t.byteLength);
		if (t.keepOnCpu) {
			let e = d.get(t.name);
			e || (e = {
				parts: [],
				dtype: t.dtype
			}, d.set(t.name, e)), e.parts.push(n.slice(0));
			continue;
		}
		let r = Math.ceil(t.byteLength / 4) * 4, a = e.createBuffer({
			size: r,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		if (t.byteLength % 4 == 0) e.queue.writeBuffer(a, 0, n);
		else {
			let t = new Uint8Array(r);
			t.set(n), e.queue.writeBuffer(a, 0, t);
		}
		f += r, u.set(t.name, a);
	}
	let m = performance.now() - p, h = d.get("embeddings.word.weight"), g = performance.now(), _ = /* @__PURE__ */ new Float32Array();
	if (h) {
		let e = h.parts.reduce((e, t) => e + t.byteLength, 0), t = new Uint8Array(e), n = 0;
		for (let e of h.parts) t.set(e, n), n += e.byteLength;
		_ = h.dtype === "f16" ? new Uint16Array(t.buffer) : new Float32Array(t.buffer);
	}
	let v = performance.now() - g;
	return {
		manifest: r,
		tensors: u,
		embeddings: _,
		downloadBytes: o,
		gpuBytes: f,
		timing: {
			fetchMs: s,
			bodyMs: c,
			sha256Ms: l,
			uploadMs: m,
			embedJoinMs: v
		}
	};
}
//#endregion
//#region src/tokenizer/unigram.ts
var s = 10;
function c(e, t) {
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
		unkScore: a - s,
		unkId: t
	};
}
function l(e, t) {
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
var u = /\s{2,}|[\n\r\t]/g, d = "▁", f = class {
	unigram;
	added;
	addedList;
	padTokenId = 0;
	constructor(e) {
		this.unigram = c(e.model.vocab, e.model.unk_id ?? 3), this.added = /* @__PURE__ */ new Map(), this.addedList = [];
		for (let t of e.added_tokens) this.added.set(t.content, t.id), this.addedList.push(t.content), t.content === "[PAD]" && (this.padTokenId = t.id);
		this.addedList.sort((e, t) => t.length - e.length);
	}
	normalize(e) {
		return e.replace(u, " ").normalize("NFC").replace(/\s+$/, "");
	}
	encodePretoken(e) {
		if (this.added.has(e)) return [this.added.get(e)];
		let t = [], n = 0;
		for (let r = 0; r <= e.length; r += 1) (r < e.length && e[r] === " " || r === e.length) && (r > n && t.push(...l(this.unigram, d + e.slice(n, r))), n = r + 1);
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
}, p = class extends Error {
	constructor(e) {
		super(e), this.name = "BucketOverflowError";
	}
}, m = "[SEP_STRUCT]", h = "[SEP_TEXT]", g = "[P]", _ = "[L]", v = /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]/giu;
function y(e) {
	let t = [];
	v.lastIndex = 0;
	for (let n of e.matchAll(v)) t.push(n[0].toLowerCase());
	return t;
}
function b(e, t, n, r, i) {
	let a = n.reduce((e, [, t]) => e + t.length, 0);
	if (a < 1 || a > i) throw new p(`expected 1..${i} labels in total, got ${a}`);
	let o = t;
	o && !/[.!?]$/.test(o) && (o += "."), o ||= ".";
	let s = y(o), c = n.map(([e, t]) => [
		"(",
		g,
		e,
		"(",
		...t.flatMap((e) => [_, e]),
		")",
		")"
	]), l = [];
	for (let e of c) l.push(...e, m);
	l.pop(), l.push(h, ...s);
	let u = /* @__PURE__ */ new Set(), d = /* @__PURE__ */ new Set(), f = 0;
	for (let e of c) {
		e.length > 1 && (u.add(f + 1), d.add(f + 1));
		for (let t = 4; t < e.length - 2; t += 2) u.add(f + t);
		f += e.length + 1;
	}
	let v = [], b = [], S = 0, C = !1;
	for (let t = 0; t < l.length; t += 1) {
		let n = l[t], r = !C;
		n === h ? C = !0 : n === m && (S += 1);
		let i = v.length;
		v.push(...e.encodeIds(n)), r && !d.has(t) && u.has(t) && b.push({
			pos: i,
			group: S
		});
	}
	let w = v.length;
	if (w > r || b.length > i) throw new p(`input exceeds bucket: seqLen ${w} > ${r} or markers > ${i}`);
	return x(b, v, w, r, i, e.padTokenId);
}
function x(e, t, n, r, i, a) {
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
//#region src/kernels/add.wgsl?raw
var S = "{{ENABLE}}// Elementwise residual accumulation. MODE 0: dst[i] += src[i].\n// MODE 1: dst[i] += src[b * N + i % N] where b = row / L is the batch\n// sequence (per-row type embeddings; at B = 1 this is src[i % N]).\n// One workgroup of 64 threads per 64 elements.\n\noverride TOTAL: u32 = 1u;\noverride N: u32 = 384u;\noverride MODE: u32 = 0u;\noverride L: u32 = 128u;\n\n@group(0) @binding(0) var<storage, read_write> dst: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> src: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= TOTAL) { return; }\n  let s = select(src[i], src[(i / (N * L)) * N + i % N], MODE == 1u);\n  dst[i] = {{F}}(f32(dst[i]) + f32(s));\n}\n", C = "{{ENABLE}}// DeBERTa-v2 relative attention, one workgroup per (head, query row).\n// scores[i,j] = (q_i.k_j + c2p[i,j] + p2c[i,j]) / SCALE over key positions,\n// masked softmax (masked pairs -> -1e4, fully masked rows give a uniform\n// distribution like the fp32 reference, never NaN), then . v.\n//   c2p[i,j] = q_i . pos_key[idx[i,j]]   (content -> position)\n//   p2c[i,j] = k_j . pos_query[idx[i,j]] (position -> content)\n// pos_key/pos_query are stored [2*SPAN, H*D] row-major (m-major).\n// The q row is hoisted into registers once (it would otherwise be re-read\n// for every key); fully masked query rows write zeros and skip the loop.\n// Batch (K16): B sequences are packed as B*L global rows; row r = b*L + i\n// belongs to sequence b, whose keys/values live at global rows b*L + j\n// while relative positions and the sliding table stay local (i, j).\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 13.856406; // sqrt(64 * 3)\n\n@group(0) @binding(0) var<storage, read> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> posKey: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> posQuery: array<{{F}}>;\n@group(0) @binding(3) var<storage, read> relidx: array<u32>;\n@group(0) @binding(4) var<storage, read> mask: array<f32>;\n@group(0) @binding(5) var<storage, read_write> ctx: array<{{F}}>;\n\n// One score per bucket position; the array follows the L override, so a\n// L1024 pipeline takes 4 KiB (well under the 16 KiB minimum limit).\nvar<workgroup> scores: array<f32, L>;\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let b = i / L;\n  let il = i - b * L;\n  let kbase = b * L;\n  let hd = H * D;\n  let qb = i * 3u * hd + h * D;\n  var qreg: array<f32, 64>;\n  for (var d = 0u; d < D; d += 1u) {\n    qreg[d] = f32(qkv[qb + d]);\n  }\n  // Masked query rows contribute nothing downstream (their ctx feeds only\n  // their own row), so they can be zeroed and skipped.\n  if (mask[i] <= 0.5) {\n    for (var d = lid.x; d < D; d += 64u) {\n      ctx[i * hd + h * D + d] = {{F}}(0.0);\n    }\n    return;\n  }\n  for (var j = lid.x; j < L; j += 64u) {\n    // Masked keys land at -1e4 regardless of the dot products; skip them.\n    if (mask[kbase + j] <= 0.5) {\n      scores[j] = -1e4;\n      continue;\n    }\n    var s = 0.0;\n    let p = relidx[il * L + j] * hd;\n    let kb = (kbase + j) * 3u * hd + (H + h) * D;\n    let pk = p + h * D;\n    let pq = p + h * D;\n    for (var d = 0u; d < D; d += 4u) {\n      let q0 = qreg[d];\n      let q1 = qreg[d + 1u];\n      let q2 = qreg[d + 2u];\n      let q3 = qreg[d + 3u];\n      let k0 = f32(qkv[kb + d]);\n      let k1 = f32(qkv[kb + d + 1u]);\n      let k2 = f32(qkv[kb + d + 2u]);\n      let k3 = f32(qkv[kb + d + 3u]);\n      s += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;\n      s += q0 * f32(posKey[pk + d]) + q1 * f32(posKey[pk + d + 1u])\n        + q2 * f32(posKey[pk + d + 2u]) + q3 * f32(posKey[pk + d + 3u]);\n      s += k0 * f32(posQuery[pq + d]) + k1 * f32(posQuery[pq + d + 1u])\n        + k2 * f32(posQuery[pq + d + 2u]) + k3 * f32(posQuery[pq + d + 3u]);\n    }\n    scores[j] = s / SCALE;\n  }\n  workgroupBarrier();\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) { m = max(m, scores[j]); }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    let e = exp(scores[j] - mx);\n    scores[j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let invTotal = 1.0 / red[0];\n  workgroupBarrier();\n  for (var d = lid.x; d < D; d += 64u) {\n    let vb = (2u * H + h) * D + d;\n    // The softmax normalization folds into the epilogue (ctx = acc / total):\n    // exp() underflows to exactly 0 for masked keys, so the sv != 0 branch\n    // still keeps stale rows out, and four independent accumulators shorten\n    // the serial FMA dependency.\n    var acc0 = 0.0;\n    var acc1 = 0.0;\n    var acc2 = 0.0;\n    var acc3 = 0.0;\n    for (var j = 0u; j + 3u < L; j += 4u) {\n      let s0 = scores[j];\n      let s1 = scores[j + 1u];\n      let s2 = scores[j + 2u];\n      let s3 = scores[j + 3u];\n      if (s0 != 0.0) { acc0 += s0 * f32(qkv[(kbase + j + 0u) * 3u * hd + vb]); }\n      if (s1 != 0.0) { acc1 += s1 * f32(qkv[(kbase + j + 1u) * 3u * hd + vb]); }\n      if (s2 != 0.0) { acc2 += s2 * f32(qkv[(kbase + j + 2u) * 3u * hd + vb]); }\n      if (s3 != 0.0) { acc3 += s3 * f32(qkv[(kbase + j + 3u) * 3u * hd + vb]); }\n    }\n    var acc = (acc0 + acc1 + acc2 + acc3) * invTotal;\n    ctx[i * hd + h * D + d] = {{F}}(acc);\n  }\n}\n", w = "{{ENABLE}}// states[g, :] = x[markers[g], :] — collect hidden states at the\n// classification marker positions. One workgroup, threads stride over rows*D.\n// Batch (K16): markers hold B blocks of 3*K (indices, mask bits, groups);\n// output row g belongs to sequence b = g / K and reads x at the\n// sequence-local marker index plus b * L.\n\noverride K: u32 = 16u;\noverride D: u32 = 384u;\noverride L: u32 = 128u;\n\n@group(0) @binding(0) var<storage, read> markers: array<u32>;\n@group(0) @binding(1) var<storage, read> x: array<{{F}}>;\n@group(0) @binding(2) var<storage, read_write> states: array<{{F}}>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  // The bound size encodes the batch: 3*K u32 per sequence.\n  let total = arrayLength(&markers) / 3u;\n  for (var i = lid.x; i < total * D; i += 64u) {\n    let g = i / D;\n    let b = g / K;\n    let d = i - g * D;\n    states[i] = x[(b * L + markers[b * 3u * K + (g - b * K)]) * D + d];\n  }\n}\n", T = "{{ENABLE}}// GeGLU epilogue for the ModernBERT FFN: mid is the raw [L, 2*I]\n// projection; out[row, j] = gelu(mid[row, j]) * mid[row, I + j] for j < I.\n// gelu is the exact erf form via Abramowitz-Stegun 7.1.26 (same constants\n// as matmul.wgsl ACT=2). One workgroup per row.\n\noverride I: u32 = 1152u;\n\n@group(0) @binding(0) var<storage, read> mid: array<{{F}}>;\n@group(0) @binding(1) var<storage, read_write> gate: array<{{F}}>;\n\nfn gelu(v: f32) -> f32 {\n  let u = v * 0.7071067811865476;\n  let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n  let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n    - 0.284496736) * t + 0.254829592) * t;\n  let e = 1.0 - p * exp(-u * u);\n  return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n}\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let base = row * 2u * I;\n  for (var j = lid.x; j < I; j += 64u) {\n    gate[row * I + j] = {{F}}(\n      gelu(f32(mid[base + j])) * f32(mid[base + I + j]));\n  }\n}\n", E = "{{ENABLE}}// LayerNorm per row, one workgroup per row, f32 accumulation.\n// MODE 0: out = LN(a)\n// MODE 1: out = LN(a) * mask[row]   (embedding masking)\n// MODE 2: out = LN(a + b)          (residual add fused)\n// mean/var over the last dimension with manifest epsilon (1e-7).\n\noverride N: u32 = 384u;\noverride MODE: u32 = 0u;\noverride EPS: f32 = 1e-7;\n\n@group(0) @binding(0) var<storage, read> a: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> b: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> weight: array<{{F}}>;\n@group(0) @binding(3) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(4) var<storage, read> mask: array<f32>;\n@group(0) @binding(5) var<storage, read_write> out: array<{{F}}>;\n\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.x;\n  let base = row * N;\n  var s = 0.0;\n  var sq = 0.0;\n  for (var i = lid.x; i < N; i += 64u) {\n    var v = f32(a[base + i]);\n    if (MODE == 2u) { v += f32(b[base + i]); }\n    s += v;\n    sq += v * v;\n  }\n  red[lid.x] = s;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let mean = red[0] / f32(N);\n  workgroupBarrier();\n  red[lid.x] = sq;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let variance = red[0] / f32(N) - mean * mean;\n  let inv = 1.0 / sqrt(variance + EPS);\n  for (var i = lid.x; i < N; i += 64u) {\n    var v = f32(a[base + i]);\n    if (MODE == 2u) { v += f32(b[base + i]); }\n    var y = (v - mean) * inv * f32(weight[i]) + f32(bias[i]);\n    if (MODE == 1u) { y *= mask[row]; }\n    out[base + i] = {{F}}(y);\n  }\n}\n", D = "{{ENABLE}}// logits[k] = raw[k] / TEMP where markerMask[k] > 0.5 else -1e4.\n// Marker payloads are packed u32: [0..K) marker_indices, [K..2K)\n// marker_mask as f32 bits, [2K..3K) marker_groups (resolved on the CPU).\n// Batch (K16): packed holds B blocks of 3*K; output k maps to sequence\n// b = k / K and mask element b*3K + K + (k mod K).\n\noverride K: u32 = 16u;\noverride TEMP: f32 = 1.0;\n\n@group(0) @binding(0) var<storage, read> raw: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> packed: array<u32>;\n@group(0) @binding(2) var<storage, read_write> logits: array<f32>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let total = arrayLength(&packed) / 3u;\n  for (var k = lid.x; k < total; k += 64u) {\n    let b = k / K;\n    let m = bitcast<f32>(packed[b * 3u * K + K + (k - b * K)]);\n    logits[k] = select(-1e4, f32(raw[k]) / TEMP, m > 0.5);\n  }\n}\n", O = "{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N]. Row-major storage, 16x16 tiles,\n// f32 accumulation. ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26).\n\noverride M: u32 = 1u;\noverride N: u32 = 1u;\noverride K: u32 = 1u;\noverride ACT: u32 = 0u;\n\n@group(0) @binding(0) var<storage, read> a: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> w: array<{{F}}>;\n@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;\n@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;\n\nvar<workgroup> ta: array<f32, 256>;\nvar<workgroup> tw: array<f32, 256>;\n\nfn activate(v: f32) -> f32 {\n  if (ACT == 1u) { return max(v, 0.0); }\n  if (ACT == 2u) {\n    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26\n    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.\n    let u = v * 0.7071067811865476;\n    let t = 1.0 / (1.0 + 0.3275911 * abs(u));\n    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t\n      - 0.284496736) * t + 0.254829592) * t;\n    let e = 1.0 - p * exp(-u * u);\n    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));\n  }\n  return v;\n}\n\n@compute @workgroup_size(16, 16)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let row = wid.y * 16u + lid.y;\n  let col = wid.x * 16u + lid.x;\n  var acc = 0.0;\n  for (var t = 0u; t < K; t += 16u) {\n    let ai = t + lid.x;\n    let wi = t + lid.y;\n    ta[lid.y * 16u + lid.x] = select(0.0, f32(a[row * K + ai]), row < M && ai < K);\n    tw[lid.y * 16u + lid.x] = select(0.0, f32(w[col * K + wi]), col < N && wi < K);\n    workgroupBarrier();\n    for (var k = 0u; k < 16u; k += 1u) {\n      acc += ta[lid.y * 16u + k] * tw[k * 16u + lid.x];\n    }\n    workgroupBarrier();\n  }\n  if (row < M && col < N) {\n    c[row * N + col] = {{F}}(activate(acc + f32(bias[col])));\n  }\n}\n", ee = "{{ENABLE}}// ModernBERT multi-head attention, one workgroup per (head, query\n// row). scores[i,j] = (q_i . k_j) / sqrt(D) over key positions, masked by the\n// key mask and (WINDOW > 0) the sliding window |i - j| <= WINDOW; softmax in\n// f32 (fully masked rows give a uniform distribution, never NaN), then . v.\n// RoPE is applied upstream to q and k inside qkv; layout matches the DeBERTa\n// kernel: row i holds [q | k | v] of H * D each.\n// The q row is hoisted into registers once; masked query rows write zeros\n// and skip the loop.\n// Batch (K16): B sequences are packed as B*L global rows; row r = b*L + i\n// attends to keys b*L + j and the sliding window compares local i to j.\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\noverride SCALE: f32 = 0.125; // 64^-0.5\noverride WINDOW: u32 = 0u;   // 0 = global attention\n\n@group(0) @binding(0) var<storage, read> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> ctx: array<{{F}}>;\n\n// One score per bucket position; the array follows the L override, so a\n// L1024 pipeline takes 4 KiB (well under the 16 KiB minimum limit).\nvar<workgroup> scores: array<f32, L>;\nvar<workgroup> red: array<f32, 64>;\n\n@compute @workgroup_size(64)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let h = wid.x;\n  let i = wid.y;\n  let b = i / L;\n  let il = i - b * L;\n  let kbase = b * L;\n  let hd = H * D;\n  let qb = i * 3u * hd + h * D;\n  var qreg: array<f32, 64>;\n  for (var d = 0u; d < D; d += 1u) {\n    qreg[d] = f32(qkv[qb + d]);\n  }\n  // Masked query rows contribute nothing downstream (their ctx feeds only\n  // their own row), so they can be zeroed and skipped.\n  if (mask[i] <= 0.5) {\n    for (var d = lid.x; d < D; d += 64u) {\n      ctx[i * hd + h * D + d] = {{F}}(0.0);\n    }\n    return;\n  }\n  for (var j = lid.x; j < L; j += 64u) {\n    var outside = mask[kbase + j] <= 0.5;\n    if (WINDOW > 0u) {\n      let dist = select(il - j, j - il, j > il);\n      outside = outside || dist > WINDOW;\n    }\n    // Masked keys land at -1e4 regardless of the dot product; skip them.\n    if (outside) {\n      scores[j] = -1e4;\n      continue;\n    }\n    var s = 0.0;\n    let kb = (kbase + j) * 3u * hd + (H + h) * D;\n    for (var d = 0u; d < D; d += 4u) {\n      s += qreg[d] * f32(qkv[kb + d])\n        + qreg[d + 1u] * f32(qkv[kb + d + 1u])\n        + qreg[d + 2u] * f32(qkv[kb + d + 2u])\n        + qreg[d + 3u] * f32(qkv[kb + d + 3u]);\n    }\n    scores[j] = s * SCALE;\n  }\n  workgroupBarrier();\n  var m = -1e30;\n  for (var j = lid.x; j < L; j += 64u) { m = max(m, scores[j]); }\n  red[lid.x] = m;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }\n    workgroupBarrier();\n  }\n  let mx = red[0];\n  workgroupBarrier();\n  var sum = 0.0;\n  for (var j = lid.x; j < L; j += 64u) {\n    let e = exp(scores[j] - mx);\n    scores[j] = e;\n    sum += e;\n  }\n  red[lid.x] = sum;\n  workgroupBarrier();\n  for (var o = 32u; o > 0u; o >>= 1u) {\n    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }\n    workgroupBarrier();\n  }\n  let invTotal = 1.0 / red[0];\n  workgroupBarrier();\n  for (var d = lid.x; d < D; d += 64u) {\n    let vb = (2u * H + h) * D + d;\n    // Softmax normalization folds into the epilogue (ctx = acc / total):\n    // exp() underflows to exactly 0 for masked keys, so the sv != 0 branch\n    // keeps stale rows out, and four accumulators shorten the serial chain.\n    var acc0 = 0.0;\n    var acc1 = 0.0;\n    var acc2 = 0.0;\n    var acc3 = 0.0;\n    for (var j = 0u; j + 3u < L; j += 4u) {\n      let s0 = scores[j];\n      let s1 = scores[j + 1u];\n      let s2 = scores[j + 2u];\n      let s3 = scores[j + 3u];\n      if (s0 != 0.0) { acc0 += s0 * f32(qkv[(kbase + j + 0u) * 3u * hd + vb]); }\n      if (s1 != 0.0) { acc1 += s1 * f32(qkv[(kbase + j + 1u) * 3u * hd + vb]); }\n      if (s2 != 0.0) { acc2 += s2 * f32(qkv[(kbase + j + 2u) * 3u * hd + vb]); }\n      if (s3 != 0.0) { acc3 += s3 * f32(qkv[(kbase + j + 3u) * 3u * hd + vb]); }\n    }\n    var acc = (acc0 + acc1 + acc2 + acc3) * invTotal;\n    ctx[i * hd + h * D + d] = {{F}}(acc);\n  }\n}\n", k = "{{ENABLE}}// RoPE (rotate-half convention) applied in place to the q and k\n// sections of the qkv buffer. One workgroup of 32 threads per (position,\n// head, section): thread d handles the pair (d, d + D/2), so no shared\n// state is needed. cossin holds per-position tables [L, 2 * D]: cos at\n// i * 2 * D + d and sin at i * 2 * D + D + d.\n// out[d]      = x[d] * cos_d - x[d + half] * sin_d\n// out[d + half] = x[d + half] * cos_d + x[d] * sin_d\n// (cos/sin are indexed by d mod half, which is why both halves use the\n// same table entries.)\n// Batch (K16): i is a global row in B*L packed space; the rotary table is\n// indexed by the sequence-local position i mod L.\n\noverride L: u32 = 128u;\noverride H: u32 = 6u;\noverride D: u32 = 64u;\n\n@group(0) @binding(0) var<storage, read_write> qkv: array<{{F}}>;\n@group(0) @binding(1) var<storage, read> cossin: array<f32>;\n\n@compute @workgroup_size(32)\nfn main(\n  @builtin(workgroup_id) wid: vec3<u32>,\n  @builtin(local_invocation_id) lid: vec3<u32>,\n) {\n  let i = wid.x;\n  let il = i - (i / L) * L;\n  let h = wid.y / 2u;\n  let seg = wid.y % 2u; // 0 = q, 1 = k (v keeps absolute position, no RoPE)\n  let hd = H * D;\n  let base = i * 3u * hd + (seg * H + h) * D;\n  let half = D / 2u;\n  let d = lid.x;\n  let a = f32(qkv[base + d]);\n  let b = f32(qkv[base + d + half]);\n  let c = cossin[il * 2u * D + d];\n  let s = cossin[il * 2u * D + D + d];\n  qkv[base + d] = {{F}}(a * c - b * s);\n  qkv[base + d + half] = {{F}}(b * c + a * s);\n}\n";
//#endregion
//#region src/kernels/index.ts
function A(e, t) {
	return e.replace("{{ENABLE}}", t ? "enable f16;\n" : "").replaceAll("{{F}}", t ? "f16" : "f32");
}
var j = {
	add: S,
	attention: C,
	gather: w,
	geglu: T,
	layernorm: E,
	masklogits: D,
	matmul: O,
	mbattention: ee,
	rope: k
};
//#endregion
//#region src/graph/deberta.ts
function M(e, t, n, r) {
	let i = e - t, a = n >> 1, o = Math.abs(i), s = i;
	return o > a && (s = (Math.ceil(Math.log(o / a) / Math.log((r - 1) / a) * (a - 1)) + a) * Math.sign(i)), Math.min(Math.max(s + n, 0), 2 * n - 1);
}
function N(e, t, n) {
	let r = new Uint32Array(e * e);
	for (let i = 0; i < e; i += 1) for (let a = 0; a < e; a += 1) r[i * e + a] = M(i, a, t, n);
	return r;
}
var P = class {
	spec;
	tensors;
	embeddingsLN;
	head;
	headTemperature;
	length;
	markers;
	headHidden;
	batch;
	device;
	f16;
	pipeLn = [];
	pipeMmQkv;
	pipeMmAttn;
	pipeMmFfn1;
	pipeMmFfn2;
	pipeMmFc1;
	pipeMmFc2;
	pipeAttn;
	pipeGather;
	pipeMaskL;
	emb;
	x;
	tmp;
	qkv;
	ctx;
	attnOut;
	mid;
	ffnOut;
	maskBuf;
	packedBuf;
	states;
	h1;
	raw;
	logitsBuf;
	staging;
	capBuf;
	ownedBufs = [];
	gpuBytes = 0;
	bgEmb;
	bgEmbPlain;
	layerBgs = [];
	bgGather;
	bgFc1;
	bgFc2;
	bgMaskL;
	constructor(e, t, n, r, i, a, o, s, c = 16, l = 768, u = 1) {
		this.spec = t, this.tensors = n, this.embeddingsLN = r, this.head = i, this.headTemperature = a, this.length = o, this.markers = c, this.headHidden = l, this.batch = u, this.device = e, this.f16 = s, this.buildPipelines(), this.buildBuffers(), this.buildBindGroups();
	}
	bytesOf(e) {
		return e * (this.f16 ? 2 : 4);
	}
	buf(e, t) {
		let n = this.device.createBuffer({
			size: e,
			usage: t
		});
		return this.ownedBufs.push(n), this.gpuBytes += e, n;
	}
	captureBuffer() {
		return this.capBuf || (this.capBuf = this.device.createBuffer({
			size: (this.spec.layers + 2) * this.length * this.spec.hiddenSize * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		}), this.ownedBufs.push(this.capBuf), this.gpuBytes += this.capBuf.size), this.capBuf;
	}
	destroy() {
		for (let e of this.ownedBufs) e.destroy();
		this.ownedBufs.length = 0;
	}
	pipe(e, t) {
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.device.createShaderModule({ code: A(e, this.f16) }),
				entryPoint: "main",
				constants: t
			}
		});
	}
	buildPipelines() {
		let e = this.length, t = e * this.batch, n = this.markers * this.batch, r = this.spec.hiddenSize, i = this.spec.intermediateSize, a = this.spec.layerNormEps;
		this.pipeLn = [
			0,
			1,
			2
		].map((e) => this.pipe(j.layernorm, {
			N: r,
			MODE: e,
			EPS: a
		}));
		let o = j.matmul;
		this.pipeMmQkv = this.pipe(o, {
			M: t,
			N: 3 * r,
			K: r,
			ACT: 0
		}), this.pipeMmAttn = this.pipe(o, {
			M: t,
			N: r,
			K: r,
			ACT: 0
		}), this.pipeMmFfn1 = this.pipe(o, {
			M: t,
			N: i,
			K: r,
			ACT: 2
		}), this.pipeMmFfn2 = this.pipe(o, {
			M: t,
			N: r,
			K: i,
			ACT: 0
		}), this.pipeMmFc1 = this.pipe(o, {
			M: n,
			N: this.headHidden,
			K: r,
			ACT: 1
		}), this.pipeMmFc2 = this.pipe(o, {
			M: n,
			N: 1,
			K: this.headHidden,
			ACT: 0
		}), this.pipeAttn = this.pipe(j.attention, {
			L: e,
			H: this.spec.heads,
			D: r / this.spec.heads,
			SCALE: Math.sqrt(3 * (r / this.spec.heads))
		}), this.pipeGather = this.pipe(j.gather, {
			K: this.markers,
			D: r,
			L: e
		}), this.pipeMaskL = this.pipe(j.masklogits, {
			K: this.markers,
			TEMP: this.headTemperature
		});
	}
	buildBuffers() {
		let e = this.length * this.batch, t = this.markers * this.batch, n = this.spec.hiddenSize, r = this.spec.intermediateSize, i = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, a = (e, t) => this.buf(this.bytesOf(e * t), i), o = i | GPUBufferUsage.COPY_SRC;
		this.emb = a(e, n), this.x = this.buf(this.bytesOf(e * n), o), this.tmp = this.buf(this.bytesOf(e * n), o), this.qkv = a(e, 3 * n), this.ctx = a(e, n), this.attnOut = a(e, n), this.mid = a(e, r), this.ffnOut = a(e, n), this.states = a(t, n), this.h1 = a(t, this.headHidden), this.raw = a(t, 1), this.maskBuf = this.buf(e * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.packedBuf = this.buf(3 * t * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.dummyBuf = this.buf(4, GPUBufferUsage.STORAGE), this.logitsBuf = this.buf(t * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST), this.staging = this.device.createBuffer({
			size: t * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		}), this.ownedBufs.push(this.staging), this.gpuBytes += t * 4;
		let s = N(this.length, this.spec.positionBuckets, this.spec.maxRelativePositions);
		this.relidxBuf = this.buf(s.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.device.queue.writeBuffer(this.relidxBuf, 0, s);
	}
	relidxBuf;
	dummyBuf;
	buildBindGroups() {
		let e = this.device, t = this.tensors, n = this.spec.layers, r = (t, n) => e.createBindGroup({
			layout: t,
			entries: n.map((e, t) => ({
				binding: t,
				resource: { buffer: e }
			}))
		}), i = (e) => (t) => r(e.getBindGroupLayout(0), t), a = i(this.pipeLn[1]);
		this.bgEmb = a([
			this.emb,
			this.dummyBuf,
			this.embeddingsLN.weight,
			this.embeddingsLN.bias,
			this.maskBuf,
			this.x
		]);
		let o = i(this.pipeLn[0]);
		this.bgEmbPlain = o([
			this.emb,
			this.dummyBuf,
			this.embeddingsLN.weight,
			this.embeddingsLN.bias,
			this.maskBuf,
			this.tmp
		]);
		let s = i(this.pipeMmQkv), c = i(this.pipeAttn), l = i(this.pipeMmAttn), u = i(this.pipeLn[2]), d = i(this.pipeMmFfn1), f = i(this.pipeMmFfn2), p = i(this.pipeLn[2]);
		for (let e = 0; e < n; e += 1) this.layerBgs.push([
			s([
				this.x,
				t.get(`layers.${e}.qkv.weight`),
				t.get(`layers.${e}.qkv.bias`),
				this.qkv
			]),
			c([
				this.qkv,
				t.get(`layers.${e}.pos_key`),
				t.get(`layers.${e}.pos_query`),
				this.relidxBuf,
				this.maskBuf,
				this.ctx
			]),
			l([
				this.ctx,
				t.get(`layers.${e}.attn_out.weight`),
				t.get(`layers.${e}.attn_out.bias`),
				this.attnOut
			]),
			u([
				this.x,
				this.attnOut,
				t.get(`layers.${e}.attn_ln.weight`),
				t.get(`layers.${e}.attn_ln.bias`),
				this.maskBuf,
				this.tmp
			]),
			d([
				this.tmp,
				t.get(`layers.${e}.ffn_in.weight`),
				t.get(`layers.${e}.ffn_in.bias`),
				this.mid
			]),
			f([
				this.mid,
				t.get(`layers.${e}.ffn_out.weight`),
				t.get(`layers.${e}.ffn_out.bias`),
				this.ffnOut
			]),
			p([
				this.tmp,
				this.ffnOut,
				t.get(`layers.${e}.ffn_ln.weight`),
				t.get(`layers.${e}.ffn_ln.bias`),
				this.maskBuf,
				this.x
			])
		]);
		this.bgGather = i(this.pipeGather)([
			this.packedBuf,
			this.x,
			this.states
		]), this.bgFc1 = i(this.pipeMmFc1)([
			this.states,
			this.head.fc1w,
			this.head.fc1b,
			this.h1
		]), this.bgFc2 = i(this.pipeMmFc2)([
			this.h1,
			this.head.fc2w,
			this.head.fc2b,
			this.raw
		]), this.bgMaskL = i(this.pipeMaskL)([
			this.raw,
			this.packedBuf,
			this.logitsBuf
		]);
	}
	layerSeq(e) {
		let t = Math.ceil(this.spec.hiddenSize / 16), n = Math.ceil(e / 16), r = Math.ceil(this.spec.intermediateSize / 16);
		return [
			[
				this.pipeMmQkv,
				0,
				Math.ceil(3 * this.spec.hiddenSize / 16),
				n,
				"qkv"
			],
			[
				this.pipeAttn,
				1,
				this.spec.heads,
				e,
				"attn"
			],
			[
				this.pipeMmAttn,
				2,
				t,
				n,
				"attnOut"
			],
			[
				this.pipeLn[2],
				3,
				e,
				1,
				"lnA"
			],
			[
				this.pipeMmFfn1,
				4,
				r,
				n,
				"ffn1"
			],
			[
				this.pipeMmFfn2,
				5,
				t,
				n,
				"ffn2"
			],
			[
				this.pipeLn[2],
				6,
				e,
				1,
				"lnF"
			]
		];
	}
	headSteps() {
		let e = Math.ceil(this.markers * this.batch / 16);
		return [
			[
				this.pipeGather,
				this.bgGather,
				1,
				1,
				"gather"
			],
			[
				this.pipeMmFc1,
				this.bgFc1,
				Math.ceil(this.headHidden / 16),
				e,
				"fc1"
			],
			[
				this.pipeMmFc2,
				this.bgFc2,
				1,
				e,
				"fc2"
			],
			[
				this.pipeMaskL,
				this.bgMaskL,
				1,
				1,
				"maskl"
			]
		];
	}
	encodeDispatchProfile(e, t) {
		let n = this.device.createCommandEncoder(), r = [], i = (e, i, a, o, s) => {
			let c = r.length;
			r.push(s);
			let l = n.beginComputePass({ timestampWrites: {
				querySet: t.querySet,
				beginningOfPassWriteIndex: c * 2,
				endOfPassWriteIndex: c * 2 + 1
			} });
			l.setPipeline(e), l.setBindGroup(0, i), l.dispatchWorkgroups(a, o), l.end();
		};
		i(this.pipeLn[1], this.bgEmb, e, 1, "embed");
		let a = this.layerSeq(e);
		for (let e = 0; e < this.spec.layers; e += 1) for (let [t, n, r, o, s] of a) i(t, this.layerBgs[e][n], r, o, `L${e}.${s}`);
		for (let [e, t, n, r, a] of this.headSteps()) i(e, t, n, r, `head.${a}`);
		return n.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.markers * this.batch * 4), n.resolveQuerySet(t.querySet, 0, r.length * 2, t.resolve, 0), n.copyBufferToBuffer(t.resolve, 0, t.staging, 0, r.length * 16), {
			commands: n.finish(),
			names: r
		};
	}
	encode(e, t, n = this.length, r) {
		let i = this.device.createCommandEncoder(), a = 0, o = () => {
			let e = r ? {
				querySet: r.querySet,
				beginningOfPassWriteIndex: a * 2,
				endOfPassWriteIndex: a * 2 + 1
			} : void 0;
			return a += 1, i.beginComputePass(e ? { timestampWrites: e } : void 0);
		}, s = e ? this.length : n, c = this.length * this.spec.hiddenSize * 4, l = this.layerSeq(s);
		if (!e && !r) {
			let e = o();
			e.setPipeline(this.pipeLn[1]), e.setBindGroup(0, this.bgEmb), e.dispatchWorkgroups(s);
			for (let n = 0; n < this.spec.layers; n += 1) {
				let r = this.layerBgs[n];
				for (let [n, i, a, o] of l) t?.has(i) || (e.setPipeline(n), e.setBindGroup(0, r[i]), e.dispatchWorkgroups(a, o));
			}
			for (let [t, n, r, i] of this.headSteps()) e.setPipeline(t), e.setBindGroup(0, n), e.dispatchWorkgroups(r, i);
			return e.end(), i.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.markers * this.batch * 4), i.finish();
		}
		let u = o();
		e ? (u.setPipeline(this.pipeLn[0]), u.setBindGroup(0, this.bgEmbPlain), u.dispatchWorkgroups(s), u.setPipeline(this.pipeLn[1])) : u.setPipeline(this.pipeLn[1]), u.setBindGroup(0, this.bgEmb), u.dispatchWorkgroups(s), u.end(), e && (i.copyBufferToBuffer(this.tmp, 0, this.captureBuffer(), 0, c), i.copyBufferToBuffer(this.x, 0, this.captureBuffer(), c, c));
		for (let n = 0; n < this.spec.layers; n += 1) {
			let r = o(), a = this.layerBgs[n];
			for (let [e, n, i, o] of l) t?.has(n) || (r.setPipeline(e), r.setBindGroup(0, a[n]), r.dispatchWorkgroups(i, o));
			r.end(), e && i.copyBufferToBuffer(this.x, 0, this.captureBuffer(), (n + 2) * c, c);
		}
		let d = o();
		for (let [e, t, n, r] of this.headSteps()) d.setPipeline(e), d.setBindGroup(0, t), d.dispatchWorkgroups(n, r);
		if (d.end(), i.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.markers * this.batch * 4), r) {
			let e = a * 16;
			i.resolveQuerySet(r.querySet, 0, a * 2, r.resolve, 0), i.copyBufferToBuffer(r.resolve, 0, r.staging, 0, e);
		}
		return i.finish();
	}
	submit(e, t, n = this.length) {
		this.device.queue.submit([this.encode(e, t, n)]);
	}
	async kernelTimesMs(e, t = "pass", n = this.length) {
		let r = await this.profileForward(e, t, n);
		return r && r.times;
	}
	async profileForward(e, t = "pass", n = this.length) {
		if (!this.device.features.has("timestamp-query")) return null;
		let r = 1 + this.spec.layers * 7 + 4, i = t === "dispatch" ? r : this.spec.layers + 2, a = this.timestampResources(i);
		this.upload(e);
		let o;
		if (t === "dispatch") {
			let e = this.encodeDispatchProfile(n, a);
			o = e.names, this.device.queue.submit([e.commands]);
		} else o = [
			"embed",
			...Array.from({ length: this.spec.layers }, (e, t) => `layer${t}`),
			"head"
		], this.device.queue.submit([this.encode(!1, void 0, this.length, a)]);
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
		this.device.queue.writeBuffer(this.emb, 0, e.embeddings), this.device.queue.writeBuffer(this.maskBuf, 0, e.mask), this.device.queue.writeBuffer(this.packedBuf, 0, e.packedMarkers);
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
}, F = "▁";
function te(e) {
	let t = [];
	for (let n of new TextEncoder().encode(e)) t.push(`<0x${n.toString(16).toUpperCase().padStart(2, "0")}>`);
	return t;
}
var ne = class {
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
		for (let n of e) this.vocab.has(n) ? t.push(n) : t.push(...te(n));
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
		let t = e.replaceAll(" ", F);
		t.startsWith(F) || (t = F + t);
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
}, re = {
	choice: 0,
	score: 1,
	noul: 2
};
function I(e, t, n = 1024, r = 256, i = !1) {
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
		qtype: re[s] ?? 0,
		seqLen: f.length,
		truncated: g
	};
}
//#endregion
//#region src/graph/julia.ts
function L(e, t, n) {
	let r = n / 2, i = new Float32Array(e * 2 * n);
	for (let a = 0; a < e; a += 1) for (let e = 0; e < r; e += 1) {
		let r = a * t ** (-(2 * e) / n);
		i[a * 2 * n + e] = Math.cos(r), i[a * 2 * n + n + e] = Math.sin(r);
	}
	return i;
}
var R = class {
	spec;
	tensors;
	length;
	batch;
	device;
	f16;
	pipeLn;
	pipeMmQkv;
	pipeMmAttn;
	pipeRope;
	pipeAttnG;
	pipeAttnW;
	pipeAdd0;
	pipeAdd1;
	pipeMmIn;
	pipeGeglu;
	pipeMmFfn;
	pipeMmLin1;
	pipeMmLin2;
	pipeGather;
	pipeMmFc1;
	pipeMmFc2;
	pipeMaskL;
	emb;
	x;
	normed;
	qkv;
	ctx;
	attnOut;
	mid;
	gate;
	ffnOut;
	tmp;
	maskBuf;
	packedBuf;
	cossinBuf;
	typeRow;
	zeroBuf;
	dummyBuf;
	states;
	kNormed;
	h1;
	raw;
	logitsBuf;
	staging;
	capBuf;
	ownedBufs = [];
	gpuBytes = 0;
	bgEmb;
	layerBgs = [];
	bgFinal;
	bgType;
	headBgs = [];
	bgGather;
	bgLnK;
	bgFc1;
	bgFc2;
	bgMaskL;
	constructor(e, t, n, r, i, a = 1) {
		this.spec = t, this.tensors = n, this.length = r, this.batch = a, this.device = e, this.f16 = i, this.buildPipelines(), this.buildBuffers(), this.buildBindGroups();
	}
	bytesOf(e) {
		return e * (this.f16 ? 2 : 4);
	}
	buf(e, t) {
		let n = this.device.createBuffer({
			size: e,
			usage: t
		});
		return this.ownedBufs.push(n), this.gpuBytes += e, n;
	}
	captureBuffer() {
		return this.capBuf || (this.capBuf = this.device.createBuffer({
			size: (this.spec.layers + 5) * this.length * this.spec.hiddenSize * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		}), this.ownedBufs.push(this.capBuf), this.gpuBytes += this.capBuf.size), this.capBuf;
	}
	destroy() {
		for (let e of this.ownedBufs) e.destroy();
		this.ownedBufs.length = 0;
	}
	pipe(e, t) {
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.device.createShaderModule({ code: A(e, this.f16) }),
				entryPoint: "main",
				constants: t
			}
		});
	}
	buildPipelines() {
		let e = this.length, t = e * this.batch, n = this.spec.options, r = n * this.batch, i = this.spec.hiddenSize, a = this.spec.intermediate, o = this.spec.normEps;
		this.pipeLn = this.pipe(j.layernorm, {
			N: i,
			MODE: 0,
			EPS: o
		});
		let s = j.matmul;
		this.pipeMmQkv = this.pipe(s, {
			M: t,
			N: 3 * i,
			K: i,
			ACT: 0
		}), this.pipeMmAttn = this.pipe(s, {
			M: t,
			N: i,
			K: i,
			ACT: 0
		}), this.pipeMmIn = this.pipe(s, {
			M: t,
			N: 2 * a,
			K: i,
			ACT: 0
		}), this.pipeMmFfn = this.pipe(s, {
			M: t,
			N: i,
			K: a,
			ACT: 0
		}), this.pipeMmLin1 = this.pipe(s, {
			M: t,
			N: this.spec.headFfn,
			K: i,
			ACT: 1
		}), this.pipeMmLin2 = this.pipe(s, {
			M: t,
			N: i,
			K: this.spec.headFfn,
			ACT: 0
		}), this.pipeMmFc1 = this.pipe(s, {
			M: r,
			N: i,
			K: i,
			ACT: 2
		}), this.pipeMmFc2 = this.pipe(s, {
			M: r,
			N: 1,
			K: i,
			ACT: 0
		}), this.pipeRope = this.pipe(j.rope, {
			L: e,
			H: this.spec.heads,
			D: i / this.spec.heads
		}), this.pipeAttnG = this.pipe(j.mbattention, {
			L: e,
			H: this.spec.heads,
			D: i / this.spec.heads,
			SCALE: (i / this.spec.heads) ** -.5,
			WINDOW: 0
		}), this.pipeAttnW = this.pipe(j.mbattention, {
			L: e,
			H: this.spec.heads,
			D: i / this.spec.heads,
			SCALE: (i / this.spec.heads) ** -.5,
			WINDOW: this.spec.localAttention
		}), this.pipeAdd0 = this.pipe(j.add, {
			TOTAL: t * i,
			N: i,
			MODE: 0,
			L: e
		}), this.pipeAdd1 = this.pipe(j.add, {
			TOTAL: t * i,
			N: i,
			MODE: 1,
			L: e
		}), this.pipeGeglu = this.pipe(j.geglu, { I: a }), this.pipeGather = this.pipe(j.gather, {
			K: n,
			D: i,
			L: e
		}), this.pipeMaskL = this.pipe(j.masklogits, {
			K: n,
			TEMP: 1
		});
	}
	buildBuffers() {
		let e = this.length * this.batch, t = this.spec.hiddenSize, n = this.spec.intermediate, r = this.spec.options * this.batch, i = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, a = (e, t) => this.buf(this.bytesOf(e * t), i), o = i | GPUBufferUsage.COPY_SRC;
		this.emb = a(e, t), this.x = this.buf(this.bytesOf(e * t), o), this.normed = a(e, t), this.qkv = a(e, 3 * t), this.ctx = a(e, t), this.attnOut = a(e, t), this.mid = a(e, 2 * n), this.gate = a(e, n), this.ffnOut = a(e, t), this.tmp = this.buf(this.bytesOf(e * t), o), this.maskBuf = this.buf(e * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.packedBuf = this.buf(3 * r * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		let s = L(this.length, this.spec.ropeTheta, t / this.spec.heads);
		this.cossinBuf = this.buf(s.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.device.queue.writeBuffer(this.cossinBuf, 0, s), this.typeRow = a(this.batch, t), this.zeroBuf = this.buf(this.bytesOf(2 * n), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this.device.queue.writeBuffer(this.zeroBuf, 0, new Uint8Array(this.bytesOf(2 * n))), this.dummyBuf = this.buf(4, GPUBufferUsage.STORAGE), this.states = a(r, t), this.kNormed = a(r, t), this.h1 = a(r, t), this.raw = a(r, 1), this.logitsBuf = this.buf(r * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST), this.staging = this.device.createBuffer({
			size: r * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		}), this.ownedBufs.push(this.staging), this.gpuBytes += r * 4;
	}
	typeEmbBuf;
	buildBindGroups() {
		let e = this.device, t = this.tensors, n = (t, n) => e.createBindGroup({
			layout: t,
			entries: n.map((e, t) => ({
				binding: t,
				resource: { buffer: e }
			}))
		}), r = (e) => (t) => n(e.getBindGroupLayout(0), t), i = r(this.pipeLn), a = r(this.pipeMmQkv), o = r(this.pipeMmAttn), s = r(this.pipeRope), c = r(this.pipeAttnG), l = r(this.pipeAttnW), u = r(this.pipeAdd0), d = r(this.pipeMmIn), f = r(this.pipeGeglu), p = r(this.pipeMmFfn), m = r(this.pipeMmLin1), h = r(this.pipeMmLin2);
		this.bgEmb = i([
			this.emb,
			this.dummyBuf,
			t.get("embeddings.norm.weight"),
			this.zeroBuf,
			this.maskBuf,
			this.x
		]);
		for (let e = 0; e < this.spec.layers; e += 1) {
			let n = `layers.${e}`, r = e % this.spec.globalEvery === 0, m = t.get(`${n}.attn_norm.weight`);
			this.layerBgs.push([
				i(m ? [
					this.x,
					this.dummyBuf,
					m,
					this.zeroBuf,
					this.maskBuf,
					this.normed
				] : [
					this.x,
					this.dummyBuf,
					m ?? this.zeroBuf,
					this.zeroBuf,
					this.maskBuf,
					this.normed
				]),
				a([
					m ? this.normed : this.x,
					t.get(`${n}.wqkv.weight`),
					this.zeroBuf,
					this.qkv
				]),
				s([this.qkv, this.cossinBuf]),
				(r ? c : l)([
					this.qkv,
					this.maskBuf,
					this.ctx
				]),
				o([
					this.ctx,
					t.get(`${n}.attn_out.weight`),
					this.zeroBuf,
					this.attnOut
				]),
				u([this.x, this.attnOut]),
				i([
					this.x,
					this.dummyBuf,
					t.get(`${n}.mlp_norm.weight`),
					this.zeroBuf,
					this.maskBuf,
					this.normed
				]),
				d([
					this.normed,
					t.get(`${n}.mlp_in.weight`),
					this.zeroBuf,
					this.mid
				]),
				f([this.mid, this.gate]),
				p([
					this.gate,
					t.get(`${n}.mlp_out.weight`),
					this.zeroBuf,
					this.ffnOut
				]),
				u([this.x, this.ffnOut])
			]);
		}
		this.bgFinal = i([
			this.x,
			this.dummyBuf,
			t.get("final_norm.weight"),
			this.zeroBuf,
			this.maskBuf,
			this.tmp
		]), this.bgType = r(this.pipeAdd1)([this.tmp, this.typeRow]);
		let g = t.get("type_emb.weight");
		this.typeEmbBuf = g;
		for (let e = 0; e < this.spec.headLayers; e += 1) {
			let n = `head.${e}`;
			this.headBgs.push([
				i([
					this.tmp,
					this.dummyBuf,
					t.get(`${n}.norm1.weight`),
					t.get(`${n}.norm1.bias`),
					this.maskBuf,
					this.normed
				]),
				a([
					this.normed,
					t.get(`${n}.in_proj.weight`),
					t.get(`${n}.in_proj.bias`),
					this.qkv
				]),
				c([
					this.qkv,
					this.maskBuf,
					this.ctx
				]),
				o([
					this.ctx,
					t.get(`${n}.out_proj.weight`),
					t.get(`${n}.out_proj.bias`),
					this.attnOut
				]),
				u([this.tmp, this.attnOut]),
				i([
					this.tmp,
					this.dummyBuf,
					t.get(`${n}.norm2.weight`),
					t.get(`${n}.norm2.bias`),
					this.maskBuf,
					this.normed
				]),
				m([
					this.normed,
					t.get(`${n}.linear1.weight`),
					t.get(`${n}.linear1.bias`),
					this.mid
				]),
				h([
					this.mid,
					t.get(`${n}.linear2.weight`),
					t.get(`${n}.linear2.bias`),
					this.ffnOut
				]),
				u([this.tmp, this.ffnOut])
			]);
		}
		this.bgGather = r(this.pipeGather)([
			this.packedBuf,
			this.tmp,
			this.states
		]), this.bgLnK = i([
			this.states,
			this.dummyBuf,
			t.get("scorer.norm.weight"),
			t.get("scorer.norm.bias"),
			this.maskBuf,
			this.kNormed
		]), this.bgFc1 = r(this.pipeMmFc1)([
			this.kNormed,
			t.get("scorer.fc1.weight"),
			t.get("scorer.fc1.bias"),
			this.h1
		]), this.bgFc2 = r(this.pipeMmFc2)([
			this.h1,
			t.get("scorer.fc2.weight"),
			t.get("scorer.fc2.bias"),
			this.raw
		]), this.bgMaskL = r(this.pipeMaskL)([
			this.raw,
			this.packedBuf,
			this.logitsBuf
		]);
	}
	encode(e, t, n, r, i) {
		let a = this.device.createCommandEncoder(), o = 0, s = () => {
			let e = r ? {
				querySet: r.querySet,
				beginningOfPassWriteIndex: o * 2,
				endOfPassWriteIndex: o * 2 + 1
			} : void 0;
			return o += 1, a.beginComputePass(e ? { timestampWrites: e } : void 0);
		}, c = e ? this.length : t, l = this.length * this.spec.hiddenSize * 4, u = Math.ceil(this.spec.hiddenSize / 16), d = Math.ceil(c / 16), f = Math.ceil(2 * this.spec.intermediate / 16), p = Math.ceil(this.spec.headFfn / 16), m = Math.ceil(this.length * this.batch * this.spec.hiddenSize / 64), h = this.spec.hiddenSize, g = this.f16 ? 2 : 4, _ = (e) => [
			[
				this.pipeLn,
				0,
				c,
				1,
				"lnA"
			],
			[
				this.pipeMmQkv,
				1,
				Math.ceil(3 * h / 16),
				d,
				"qkv"
			],
			[
				this.pipeRope,
				2,
				c,
				2 * this.spec.heads,
				"rope"
			],
			[
				e % this.spec.globalEvery === 0 ? this.pipeAttnG : this.pipeAttnW,
				3,
				this.spec.heads,
				c,
				"attn"
			],
			[
				this.pipeMmAttn,
				4,
				u,
				d,
				"attnOut"
			],
			[
				this.pipeAdd0,
				5,
				m,
				1,
				"addA"
			],
			[
				this.pipeLn,
				6,
				c,
				1,
				"lnF"
			],
			[
				this.pipeMmIn,
				7,
				f,
				d,
				"ffnIn"
			],
			[
				this.pipeGeglu,
				8,
				c,
				1,
				"geglu"
			],
			[
				this.pipeMmFfn,
				9,
				u,
				d,
				"ffnOut"
			],
			[
				this.pipeAdd0,
				10,
				m,
				1,
				"addF"
			]
		], v = (e) => {
			for (let t = 0; t < this.spec.layers; t += 1) {
				let n = this.layerBgs[t];
				for (let [r, i, a, o] of _(t)) (i !== 0 || t !== 0) && (e.setPipeline(r), e.setBindGroup(0, n[i]), e.dispatchWorkgroups(a, o));
			}
		}, y = (e) => [
			[
				this.pipeLn,
				0,
				c,
				1,
				"lnA"
			],
			[
				this.pipeMmQkv,
				1,
				Math.ceil(3 * h / 16),
				d,
				"qkv"
			],
			[
				this.pipeAttnG,
				2,
				this.spec.heads,
				c,
				"attn"
			],
			[
				this.pipeMmAttn,
				3,
				u,
				d,
				"attnOut"
			],
			[
				this.pipeAdd0,
				4,
				m,
				1,
				"addA"
			],
			[
				this.pipeLn,
				5,
				c,
				1,
				"lnF"
			],
			[
				this.pipeMmLin1,
				6,
				p,
				d,
				"lin1"
			],
			[
				this.pipeMmLin2,
				7,
				u,
				d,
				"lin2"
			],
			[
				this.pipeAdd0,
				8,
				m,
				1,
				"addF"
			]
		], b = (e) => {
			for (let t = 0; t < this.spec.headLayers; t += 1) {
				let n = this.headBgs[t];
				for (let [r, i, a, o] of y(t)) e.setPipeline(r), e.setBindGroup(0, n[i]), e.dispatchWorkgroups(a, o);
			}
		}, x = () => {
			let e = this.spec.options * this.batch, t = Math.ceil(e / 16);
			return [
				[
					this.pipeGather,
					this.bgGather,
					1,
					1,
					"gather"
				],
				[
					this.pipeLn,
					this.bgLnK,
					e,
					1,
					"ln"
				],
				[
					this.pipeMmFc1,
					this.bgFc1,
					Math.ceil(h / 16),
					t,
					"fc1"
				],
				[
					this.pipeMmFc2,
					this.bgFc2,
					1,
					t,
					"fc2"
				],
				[
					this.pipeMaskL,
					this.bgMaskL,
					1,
					1,
					"maskl"
				]
			];
		}, S = (e) => {
			for (let [t, n, r, i] of x()) e.setPipeline(t), e.setBindGroup(0, n), e.dispatchWorkgroups(r, i);
		}, C = Array.isArray(n) ? n : [n];
		for (let e = 0; e < this.batch; e += 1) a.copyBufferToBuffer(this.typeEmbBuf, (C[e] ?? 0) * h * g, this.typeRow, e * h * g, h * g);
		if (r && i) {
			let e = (e, t, n, r, a) => {
				i.push(a);
				let o = s();
				o.setPipeline(e), o.setBindGroup(0, t), o.dispatchWorkgroups(n, r), o.end();
			};
			e(this.pipeLn, this.bgEmb, c, 1, "embed");
			for (let t = 0; t < this.spec.layers; t += 1) for (let [n, r, i, a, o] of _(t)) (r !== 0 || t !== 0) && e(n, this.layerBgs[t][r], i, a, `L${t}.${o}`);
			e(this.pipeLn, this.bgFinal, c, 1, "final"), e(this.pipeAdd1, this.bgType, m, 1, "type");
			for (let t = 0; t < this.spec.headLayers; t += 1) for (let [n, r, i, a, o] of y(t)) e(n, this.headBgs[t][r], i, a, `head${t}.${o}`);
			for (let [t, n, r, i, a] of x()) e(t, n, r, i, `scorer.${a}`);
			return a.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4), a.resolveQuerySet(r.querySet, 0, o * 2, r.resolve, 0), a.copyBufferToBuffer(r.resolve, 0, r.staging, 0, o * 16), a.finish();
		}
		if (!e && !r) {
			let e = s();
			return e.setPipeline(this.pipeLn), e.setBindGroup(0, this.bgEmb), e.dispatchWorkgroups(c), v(e), e.setPipeline(this.pipeLn), e.setBindGroup(0, this.bgFinal), e.dispatchWorkgroups(c), e.setPipeline(this.pipeAdd1), e.setBindGroup(0, this.bgType), e.dispatchWorkgroups(m), b(e), S(e), e.end(), a.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4), a.finish();
		}
		let w = s();
		w.setPipeline(this.pipeLn), w.setBindGroup(0, this.bgEmb), w.dispatchWorkgroups(c), w.end(), e && a.copyBufferToBuffer(this.x, 0, this.captureBuffer(), 0, l);
		for (let t = 0; t < this.spec.layers; t += 1) {
			let n = s(), r = this.layerBgs[t];
			for (let [e, i, a, o] of _(t)) (i !== 0 || t !== 0) && (n.setPipeline(e), n.setBindGroup(0, r[i]), n.dispatchWorkgroups(a, o));
			n.end(), e && a.copyBufferToBuffer(this.x, 0, this.captureBuffer(), (t + 1) * l, l);
		}
		let T = s();
		T.setPipeline(this.pipeLn), T.setBindGroup(0, this.bgFinal), T.dispatchWorkgroups(c), T.end(), e && a.copyBufferToBuffer(this.tmp, 0, this.captureBuffer(), (this.spec.layers + 1) * l, l);
		let E = s();
		E.setPipeline(this.pipeAdd1), E.setBindGroup(0, this.bgType), E.dispatchWorkgroups(m), E.end(), e && a.copyBufferToBuffer(this.tmp, 0, this.captureBuffer(), (this.spec.layers + 2) * l, l);
		for (let t = 0; t < this.spec.headLayers; t += 1) {
			let n = s(), r = this.headBgs[t];
			for (let [e, i, a, o] of y(t)) n.setPipeline(e), n.setBindGroup(0, r[i]), n.dispatchWorkgroups(a, o);
			n.end(), e && a.copyBufferToBuffer(this.tmp, 0, this.captureBuffer(), (this.spec.layers + 3 + t) * l, l);
		}
		let D = s();
		if (S(D), D.end(), a.copyBufferToBuffer(this.logitsBuf, 0, this.staging, 0, this.spec.options * this.batch * 4), r) {
			let e = o * 16;
			a.resolveQuerySet(r.querySet, 0, o * 2, r.resolve, 0), a.copyBufferToBuffer(r.resolve, 0, r.staging, 0, e);
		}
		return a.finish();
	}
	submit(e, t, n) {
		this.device.queue.submit([this.encode(e, t, n)]);
	}
	async kernelTimesMs(e, t = "pass", n = this.length) {
		let r = await this.profileForward(e, t, n);
		return r && r.times;
	}
	async profileForward(e, t = "pass", n = this.length) {
		if (!this.device.features.has("timestamp-query")) return null;
		let r = t === "dispatch", i = 3 + this.spec.layers * 11 - 1 + this.spec.headLayers * 9 + 5, a = r ? i : this.spec.layers + this.spec.headLayers + 4, o = this.timestampResources(a);
		this.upload(e);
		let s = r ? [] : [
			"embed",
			...Array.from({ length: this.spec.layers }, (e, t) => `layer${t}`),
			"final",
			"typed",
			...Array.from({ length: this.spec.headLayers }, (e, t) => `head${t}`),
			"scorer"
		];
		if (this.device.queue.submit([this.encode(!1, r ? n : this.length, e.qtype, o, r ? s : void 0)]), s.length !== a) throw Error(`profile: ${s.length} timed passes, expected ${a}`);
		await o.staging.mapAsync(GPUMapMode.READ);
		let c = new BigUint64Array(o.staging.getMappedRange().slice(0));
		o.staging.unmap();
		let l = {};
		for (let e = 0; e < a; e += 1) l[s[e]] = Number(c[e * 2 + 1] - c[e * 2]) / 1e6;
		return {
			times: l,
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
		this.device.queue.writeBuffer(this.emb, 0, e.embeddings), this.device.queue.writeBuffer(this.maskBuf, 0, e.mask), this.device.queue.writeBuffer(this.packedBuf, 0, e.packedMarkers);
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
}, z = [
	1,
	4,
	8,
	16
], B = z[z.length - 1];
function V(e) {
	for (let t of z) if (e <= t) return t;
	throw Error(`batch of ${e} exceeds the maximum of ${B}`);
}
var H = class {
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
function U(e) {
	return `${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markerIndices.join(",")}|${e.markerMask.join(",")}|${e.markerGroups.join(",")}`;
}
function W(e) {
	return `${e.qtype}|${e.inputIds.subarray(0, e.seqLen).join(",")}|${e.markers.join(",")}`;
}
function G(e) {
	return Math.ceil(e / 64) * 64;
}
function K(e, t, n) {
	let r = e.map((e, t) => t).sort((t, r) => n(e[t]) - n(e[r])), i = Array(e.length);
	return r.forEach((e, t) => {
		i[e] = t;
	}), {
		unique: r.map((t) => e[t]),
		slot: t.map((e) => i[e])
	};
}
function q(e, t) {
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
//#region src/julia.ts
function J(e, t) {
	let n = new Float32Array(t), r = -Infinity;
	for (let n = 0; n < t; n += 1) r = Math.max(r, e[n]);
	let i = 0;
	for (let n = 0; n < t; n += 1) i += Math.exp(e[n] - r);
	for (let a = 0; a < t; a += 1) n[a] = Math.exp(e[a] - r) / i;
	return n;
}
var ie = 8, ae = class e {
	kh;
	weights;
	tokenizer;
	spec;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	tokenizerBytes = 0;
	gpuBytes = 0;
	loadTiming = {};
	cache = new H(0);
	constructor(e, t, n, r, i) {
		this.kh = e, this.weights = t, this.tokenizer = n, this.spec = r, this.precision = i;
	}
	static async load(n, r) {
		let a = n.precision !== "f32", s = n.manifestUrl.slice(0, n.manifestUrl.lastIndexOf("/") + 1), c = performance.now(), l = r ? Promise.resolve(r) : i(n.manifestUrl), u = await t(a, n.limits !== "default"), d = await l, f = performance.now() - c, p = (async () => (await fetch(s + d.tokenizer)).arrayBuffer())(), m = await o(u.device, n.manifestUrl, d), h = m.manifest.tensors.find((e) => e.name.endsWith("wqkv.weight"))?.dtype ?? "f32";
		if (h === "f16" && !u.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n.precision && n.precision !== "auto" && n.precision !== h) throw Error(`precision ${n.precision} requested but manifest is ${h}`);
		let g = h, _ = performance.now(), v = await p, y = new ne(JSON.parse(new TextDecoder().decode(v))), b = performance.now() - _, x = d.encoder, S = new e(u, m, y, x, g);
		S.downloadBytes = m.downloadBytes + v.byteLength, S.tokenizerBytes = v.byteLength, S.gpuBytes = m.gpuBytes, S.loadTiming = {
			...m.timing,
			manifestMs: f,
			tokenizerMs: b,
			planMs: 0
		};
		let C = performance.now();
		for (let e of n.buckets ?? [512]) {
			let t = typeof e == "number" ? e : e.length, n = new R(u.device, x, m.tensors, t, g === "f16");
			S.gpuBytes += n.gpuBytes, S.plans.set(t, n);
		}
		return S.cache = new H(n.cacheSize ?? 256), S.loadTiming.planMs = performance.now() - C, S;
	}
	pickBucket(e) {
		let t = [...this.plans.keys()].sort((e, t) => e - t).find((t) => e <= t);
		if (t === void 0) throw new p(`seqLen ${e} exceeds largest bucket`);
		return this.plans.get(t);
	}
	embeddingRows(e, t) {
		let n = t.length, r = this.spec.hiddenSize;
		if (this.weights.embeddings instanceof Float32Array) {
			let t = new Float32Array(n * r), i = this.weights.embeddings;
			for (let n = 0; n < e.seqLen; n += 1) {
				let a = e.inputIds[n] * r;
				t.set(i.subarray(a, a + r), n * r);
			}
			return t;
		}
		let i = new Uint16Array(n * r), a = this.weights.embeddings;
		for (let t = 0; t < e.seqLen; t += 1) {
			let n = e.inputIds[t] * r;
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
		let r = n === void 0 ? this.pickBucket(e.seqLen) : this.plans.get(n);
		if (!r) throw Error(`bucket ${n} not loaded`);
		if (e.seqLen > r.length) throw new p(`seqLen ${e.seqLen} exceeds bucket ${r.length}`);
		let i = W(e), a = t ? void 0 : this.cache.get(i);
		return a ? {
			logits: a.logits,
			probabilities: a.probabilities
		} : this.enqueue(async () => {
			r.upload({
				embeddings: this.embeddingRows(e, r),
				mask: this.maskOf(e, r.length),
				packedMarkers: this.packedMarkers(e),
				qtype: e.qtype
			}), r.submit(t, e.seqLen, e.qtype);
			let n = await r.readLogits(), a = J(n, Math.min(e.markers.length, this.spec.options));
			return t || this.cache.set(i, {
				logits: n,
				probabilities: a
			}), {
				logits: n,
				probabilities: a,
				captureData: t ? await r.readCapture() : void 0
			};
		});
	}
	batchPlan(e, t) {
		let n = `${e}:${t}`, r = this.batchPlans.get(n);
		if (r) return r;
		let i = this.precision === "f16" ? 2 : 4, a = this.spec.hiddenSize, o = this.spec.intermediate, s = this.kh.device.limits.maxStorageBufferBindingSize, c = t * e * Math.max(3 * a, 2 * o) * i;
		if (c > s) throw new p(`batch ${t} x stride ${e} needs ${c} B > ${s} B binding limit`);
		if (this.batchPlans.size >= ie) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		return r = new R(this.kh.device, this.spec, this.weights.tensors, e, this.precision === "f16", t), this.gpuBytes += r.gpuBytes, this.batchPlans.set(n, r), r;
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
				let a = o.inputIds[t] * n;
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
		let n = Math.max(...e.map((e) => e.seqLen)), r = t === void 0 ? this.pickBucket(n) : this.plans.get(t);
		if (!r) throw Error(`bucket ${t} not loaded`);
		if (n > r.length) throw new p(`batch seqLen ${n} exceeds bucket ${r.length}`);
		let i = V(e.length), a = i === 1 ? n : Math.min(r.length, G(n)), o;
		try {
			o = i === 1 ? r : this.batchPlan(a, i);
		} catch (n) {
			if (!(n instanceof p) || e.length < 2) throw n;
			let r = Math.ceil(e.length / 2), i = await this.runBatchChunk(e.slice(0, r), t), a = await this.runBatchChunk(e.slice(r), t);
			return [...i, ...a];
		}
		let s = e.length === o.batch ? e : [...e, ...Array.from({ length: o.batch - e.length }, () => this.padInput())];
		return this.enqueue(async () => {
			o.upload({
				embeddings: this.batchEmbeddingRows(s, o),
				mask: this.batchMask(s, o),
				packedMarkers: this.batchPackedMarkers(s),
				qtype: 0
			}), o.submit(!1, a * o.batch, s.map((e) => e.qtype));
			let t = await o.readLogits(), n = this.spec.options;
			return e.map((e, r) => {
				let i = t.slice(r * n, (r + 1) * n), a = {
					logits: i,
					probabilities: J(i, Math.min(e.markers.length, n))
				};
				return this.cache.set(W(e), a), a;
			});
		});
	}
	async runPreparedBatch(e, t) {
		let n = Array(e.length), r = [], i = [];
		for (let [t, a] of e.entries()) {
			let e = this.cache.get(W(a));
			e ? n[t] = {
				logits: e.logits,
				probabilities: e.probabilities
			} : (r.push(a), i.push(t));
		}
		let { unique: a, slot: o } = q(r, W), s = K(a, o, (e) => e.seqLen), c = [];
		for (let e = 0; e < s.unique.length; e += B) {
			let n = await this.runBatchChunk(s.unique.slice(e, e + B), t);
			c.push(...n);
		}
		for (let [e] of r.entries()) n[i[e]] = c[s.slot[e]];
		return n;
	}
	prepare(e, t = 1024, n = 512, r = !1) {
		return I(this.tokenizer, e, t, n, r);
	}
	async decide(e) {
		let t = performance.now(), n = I(this.tokenizer, e, this.maxBucket(), 256), r = performance.now() - t, i = performance.now(), { logits: a, probabilities: o } = await this.runPrepared({
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
			let t = I(this.tokenizer, e, this.maxBucket(), 256);
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
	info() {
		return {
			precision: this.precision,
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			gpuBytes: this.gpuBytes,
			downloadBytes: this.downloadBytes,
			tokenizerBytes: this.tokenizerBytes,
			weightBytes: this.downloadBytes - this.tokenizerBytes,
			loadTiming: this.loadTiming
		};
	}
	dispose() {
		this.kh.device.destroy();
	}
}, Y = class e {
	worker;
	seq = 0;
	pending = /* @__PURE__ */ new Map();
	infoData = {};
	cache = new H(0);
	constructor() {}
	static async load(t) {
		let n = new e();
		return n.cache = new H(t.cacheSize ?? 256), n.worker = new Worker(new URL(
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
		let { unique: a, slot: o } = q(r, t), s = [];
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
		let { unique: n, slot: r } = q(e, U), i = [];
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
//#region src/index.ts
async function oe(e) {
	let t = e.backend ?? "auto";
	if (t === "wasm") {
		if (X(await i(e.manifestUrl))) throw Error("julia-1 needs WebGPU; the wasm fallback covers DeBERTa only");
		return Y.load(e);
	}
	if (typeof navigator < "u" && navigator.gpu) {
		let n = await i(e.manifestUrl);
		if (X(n)) return ae.load(e, n);
		try {
			return await Q.load(e);
		} catch (e) {
			if (t === "webgpu") throw e;
		}
	} else if (X(await i(e.manifestUrl))) throw Error("julia-1 needs WebGPU; the wasm fallback covers DeBERTa only");
	return Y.load(e);
}
function X(e) {
	return e.encoder?.arch === "modernbert-julia";
}
var Z = 16, se = 8, Q = class e {
	kh;
	weights;
	tokenizer;
	spec;
	temperature;
	precision;
	plans = /* @__PURE__ */ new Map();
	batchPlans = /* @__PURE__ */ new Map();
	queue = Promise.resolve();
	downloadBytes = 0;
	tokenizerBytes = 0;
	gpuBytes = 0;
	loadTiming = {};
	cache = new H(0);
	embLN;
	head;
	headHidden = 768;
	constructor(e, t, n, r, i, a) {
		this.kh = e, this.weights = t, this.tokenizer = n, this.spec = r, this.temperature = i, this.precision = a;
	}
	static async load(n) {
		let r = n.precision !== "f32", a = n.manifestUrl.slice(0, n.manifestUrl.lastIndexOf("/") + 1), s = performance.now(), c = i(n.manifestUrl), l = await t(r, n.limits !== "default"), u = await c, d = performance.now() - s, p = (async () => (await fetch(a + u.tokenizer)).arrayBuffer())(), m = await o(l.device, n.manifestUrl, u), h = m.manifest.tensors.find((e) => e.name.endsWith("qkv.weight"))?.dtype ?? "f32";
		if (h === "f16" && !l.hasF16) throw Error("f16 manifest but adapter lacks shader-f16; use the f32 manifest");
		if (n.precision && n.precision !== "auto" && n.precision !== h) throw Error(`precision ${n.precision} requested but manifest is ${h}`);
		let g = h, _ = performance.now(), v = await p, y = new f(JSON.parse(new TextDecoder().decode(v))), b = performance.now() - _, x = m.manifest.encoder, S = new e(l, m, y, x, m.manifest.head.temperature, g);
		S.downloadBytes = m.downloadBytes + v.byteLength, S.tokenizerBytes = v.byteLength, S.gpuBytes = m.gpuBytes, S.loadTiming = {
			...m.timing,
			manifestMs: d,
			tokenizerMs: b,
			planMs: 0
		};
		let C = m.tensors, w = {
			weight: C.get("embeddings.LayerNorm.weight"),
			bias: C.get("embeddings.LayerNorm.bias")
		}, T = {
			fc1w: C.get("head.fc1.weight"),
			fc1b: C.get("head.fc1.bias"),
			fc2w: C.get("head.fc2.weight"),
			fc2b: C.get("head.fc2.bias")
		}, E = Number(m.manifest.head.hiddenSize) || 768;
		S.embLN = w, S.head = T, S.headHidden = E, S.cache = new H(n.cacheSize ?? 256);
		let D = performance.now();
		for (let e of n.buckets ?? [128]) {
			let t = typeof e == "number" ? {
				length: e,
				markers: Z
			} : {
				length: e.length,
				markers: e.markers ?? Z
			}, n = new P(l.device, x, C, w, T, S.temperature, t.length, g === "f16", t.markers, E);
			S.gpuBytes += n.gpuBytes, S.plans.set(t.length, n);
		}
		return S.loadTiming.planMs = performance.now() - D, S;
	}
	pickBucket(e, t = 1) {
		let n = [...this.plans.values()].filter((n) => e <= n.length && t <= n.markers).sort((e, t) => e.length - t.length || e.markers - t.markers);
		if (!n.length) throw new p(`seqLen ${e} or ${t} markers exceed loaded buckets`);
		return n[0];
	}
	embeddingRows(e, t) {
		let n = t.length, r = this.spec.hiddenSize;
		if (this.weights.embeddings instanceof Float32Array) {
			let t = new Float32Array(n * r), i = this.weights.embeddings;
			for (let n = 0; n < e.seqLen; n += 1) {
				let a = e.inputIds[n] * r;
				t.set(i.subarray(a, a + r), n * r);
			}
			return t;
		}
		let i = new Uint16Array(n * r), a = this.weights.embeddings;
		for (let t = 0; t < e.seqLen; t += 1) {
			let n = e.inputIds[t] * r;
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
		let r = e.markerMask.reduce((e, t) => e + +(t > .5), 0), i = n === void 0 ? this.pickBucket(e.seqLen, r) : this.plans.get(n);
		if (!i) throw Error(`bucket ${n} not loaded`);
		if (e.seqLen > i.length || r > i.markers) throw new p(`seqLen ${e.seqLen}/${r} markers exceeds bucket ${i.length}/${i.markers}`);
		let a = U(e), o = t ? void 0 : this.cache.get(a);
		return o ? {
			logits: o.logits,
			probabilities: o.probabilities
		} : this.enqueue(async () => {
			i.upload({
				embeddings: this.embeddingRows(e, i),
				mask: this.maskOf(e, i.length),
				packedMarkers: this.packedMarkers(e, i.markers)
			}), i.submit(t, void 0, e.seqLen);
			let n = await i.readLogits(), r = $(n, e.markerGroups, e.markerMask);
			return t || this.cache.set(a, {
				logits: n,
				probabilities: r
			}), {
				logits: n,
				probabilities: r,
				captureData: t ? await i.readCapture() : void 0
			};
		});
	}
	batchPlan(e, t, n) {
		let r = `${e}:${t}:${n}`, i = this.batchPlans.get(r);
		if (i) return i;
		let a = this.precision === "f16" ? 2 : 4, o = this.spec.hiddenSize, s = this.spec.intermediateSize, c = this.kh.device.limits.maxStorageBufferBindingSize, l = Math.max(n * e * Math.max(3 * o, s), n * t * Math.max(o, this.headHidden)) * a;
		if (l > c) throw new p(`batch ${n} x stride ${e} needs ${l} B > ${c} B binding limit`);
		if (this.batchPlans.size >= se) {
			let e = this.batchPlans.keys().next().value;
			this.batchPlans.get(e)?.destroy(), this.batchPlans.delete(e);
		}
		return i = new P(this.kh.device, this.spec, this.weights.tensors, this.embLN, this.head, this.temperature, e, this.precision === "f16", t, this.headHidden, n), this.gpuBytes += i.gpuBytes, this.batchPlans.set(r, i), i;
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
				let a = o.inputIds[t] * n;
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
		let n = Math.max(...e.map((e) => e.seqLen)), r = Math.max(...e.map((e) => e.markerMask.reduce((e, t) => e + +(t > .5), 0))), i = t === void 0 ? this.pickBucket(n, r) : this.plans.get(t);
		if (!i) throw Error(`bucket ${t} not loaded`);
		if (n > i.length || r > i.markers) throw new p(`batch exceeds bucket ${i.length}/${i.markers}`);
		let a = V(e.length), o = a === 1 ? n : Math.min(i.length, G(n)), s;
		try {
			s = a === 1 ? i : this.batchPlan(o, i.markers, a);
		} catch (n) {
			if (!(n instanceof p) || e.length < 2) throw n;
			let r = Math.ceil(e.length / 2), i = await this.runBatchChunk(e.slice(0, r), t), a = await this.runBatchChunk(e.slice(r), t);
			return [...i, ...a];
		}
		let c = e.length === s.batch ? e : [...e, ...Array.from({ length: s.batch - e.length }, () => this.padInput())];
		return this.enqueue(async () => {
			s.upload({
				embeddings: this.batchEmbeddingRows(c, s),
				mask: this.batchMask(c, s),
				packedMarkers: this.batchPackedMarkers(c, s)
			}), s.submit(!1, void 0, o * s.batch);
			let t = await s.readLogits();
			return e.map((e, n) => {
				let r = t.slice(n * s.markers, (n + 1) * s.markers), i = {
					logits: r,
					probabilities: $(r, e.markerGroups, e.markerMask)
				};
				return this.cache.set(U(e), i), i;
			});
		});
	}
	async runPreparedBatch(e, t) {
		let n = Array(e.length), r = [], i = [];
		for (let [t, a] of e.entries()) {
			let e = this.cache.get(U(a));
			e ? n[t] = {
				logits: e.logits,
				probabilities: e.probabilities
			} : (r.push(a), i.push(t));
		}
		let { unique: a, slot: o } = q(r, U), s = K(a, o, (e) => e.seqLen), c = [];
		for (let e = 0; e < s.unique.length; e += B) {
			let n = await this.runBatchChunk(s.unique.slice(e, e + B), t);
			c.push(...n);
		}
		for (let [e] of r.entries()) n[i[e]] = c[s.slot[e]];
		return n;
	}
	async classify(e, t) {
		let n = performance.now(), r = t.map((e) => [e.task, e.labels]), i = r.reduce((e, [, t]) => e + t.length, 0), a = b(this.tokenizer, e, r, this.maxBucket(i), i), o = performance.now() - n, s = performance.now(), { logits: c, probabilities: l } = await this.runPrepared(a), u = performance.now() - s;
		return this.toResult(t, a, c, l, {
			tokenizeMs: o,
			gpuMs: u,
			totalMs: performance.now() - n
		});
	}
	async classifyBatch(e) {
		let t = performance.now(), n = e.map((e) => {
			let t = e.tasks.map((e) => [e.task, e.labels]), n = t.reduce((e, [, t]) => e + t.length, 0);
			return b(this.tokenizer, e.text, t, this.maxBucket(n), n);
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
					let o = ce(t.markerGroups, a, i);
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
		if (!t.length) throw new p(`no bucket routes ${e} markers`);
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
	info() {
		return {
			precision: this.precision,
			buildId: "munat69a",
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps,
			buckets: [...this.plans.keys()].sort((e, t) => e - t),
			gpuBytes: this.gpuBytes,
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
function $(e, t, n) {
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
function ce(e, t, n) {
	let r = 0;
	for (let i = 0; i < e.length; i += 1) if (e[i] === t) {
		if (r === n) return i;
		r += 1;
	}
	return -1;
}
//#endregion
export { p as BucketOverflowError, Q as Kleinhirn, $ as groupSoftmax, oe as loadEngine };
