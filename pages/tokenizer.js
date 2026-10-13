import { a as e, c as t, i as n, l as r, n as i, o as a, r as o, s, t as c, u as l } from "./pairs-BUwAo4kH.js";
//#region src/tokenizer/hf/normalized.ts
function u(e, t = 0) {
	let n = [], r = [], i = [], a = t;
	for (let t of e) n.push(t), r.push(a), a += t.length, i.push(a);
	return {
		chars: n,
		starts: r,
		ends: i,
		origin: t
	};
}
function d(e) {
	let t = e.codePointAt(0);
	return t < 128 ? 1 : t < 2048 ? 2 : t < 65536 ? 3 : 4;
}
function f(e, t, n = 0) {
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
function p(e, t, n) {
	return {
		chars: e.chars.slice(t, n),
		starts: e.starts.slice(t, n),
		ends: e.ends.slice(t, n),
		origin: t < e.starts.length ? e.starts[t] : e.origin
	};
}
function m(e, t) {
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
function h(e) {
	let t = [], n = 0;
	for (let r = 0; r < e.length; r += 1) {
		for (; t.length < n;) t.push(r - 1);
		t.push(r), n += e[r].length;
	}
	for (; t.length <= n;) t.push(e.length);
	return t[n] = e.length, t;
}
function g(e, t) {
	if (e.length === 0) return [[
		0,
		0,
		!1
	]];
	let n = e.join(""), r = h(e), i = [], a = 0;
	t.lastIndex = 0;
	for (let e = t.exec(n); e !== null; e = t.exec(n)) {
		let o = r[e.index], s = r[e.index + e[0].length];
		if (e[0].length === 0) {
			t.lastIndex = e.index + ((n.codePointAt(e.index) ?? 0) > 65535 ? 2 : 1);
			let r = n.charCodeAt(e.index);
			if (r >= 56320 && r <= 57343 && e.index > 0 && (n.codePointAt(e.index - 1) ?? 0) > 65535) continue;
		}
		a !== o && i.push([
			a,
			o,
			!1
		]), i.push([
			o,
			s,
			!0
		]), a = s;
	}
	return a !== e.length && i.push([
		a,
		e.length,
		!1
	]), i;
}
function _(e, t, n) {
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
	for (let [t, n, a] of r) a || i.push(p(e, t, n));
	return i;
}
function v(e, t, n) {
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
function y(e, t) {
	if (e.chars.length === 0) return e;
	let n = [...t];
	return {
		chars: [...n, ...e.chars],
		starts: [...n.map(() => e.starts[0]), ...e.starts],
		ends: [...n.map(() => e.ends[0]), ...e.ends],
		origin: e.origin
	};
}
function b(e) {
	return e.chars.join("");
}
//#endregion
//#region src/tokenizer/hf/bpe.ts
var x = class e {
	vocab;
	base;
	ranks;
	mergedIds;
	opt;
	constructor(e, t, n) {
		this.vocab = new i(e), this.opt = n;
		let r = 0;
		for (let t = 0; t < e.tokens.length; t += 1) {
			let n = a(e, t);
			if (!Number.isInteger(n) || n < 0) throw Error(`BPE: vocab id ${n} of ${e.tokens.tokenAt(t)} is not a non-negative integer`);
			r = Math.max(r, n);
		}
		if (this.base = r + 1, this.base * this.base > 2 ** 53 - 1) throw Error(`BPE: largest vocab id ${r} is too large for a collision-free pair key`);
		let o = n.continuingSubwordPrefix === null ? 0 : n.continuingSubwordPrefix.length, s = t.length / 2;
		this.ranks = new c(s), this.mergedIds = new Int32Array(s);
		for (let n = 0, r = 0; n + 1 < t.length; n += 2, r += 1) {
			let i = a(e, t[n]), s = a(e, t[n + 1]), c = this.vocab.idOfConcat(t[n], t[n + 1], o);
			if (c === void 0) {
				let r = e.tokens.tokenAt(t[n]);
				throw Error(`BPE: merged token ${r + e.tokens.tokenAt(t[n + 1]).slice(o)} is not in the vocab`);
			}
			this.ranks.set(i, s, r), this.mergedIds[r] = c;
		}
		if (n.unkToken !== null && !this.vocab.has(n.unkToken)) throw Error(`BPE: unk token ${n.unkToken} is not in the vocab`);
	}
	static fromJson(t) {
		if (t.dropout !== null && t.dropout !== void 0) throw Error("BPE: dropout is not supported");
		let r = o(t.vocab);
		return new e(r, n(r, t.merges), {
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
			let n = this.vocab.id(t);
			if (n !== void 0) return [{
				id: n,
				start: 0,
				end: e.length,
				value: t
			}];
		}
		let n = [];
		e.forEach((e, t) => {
			for (let r = d(e); r > 0; --r) n.push(t);
		});
		let r = [], i = 0;
		for (let t of this.mergeWord(e)) {
			let e = n[Math.min(i, n.length - 1)], a = n[Math.min(i + t.len, n.length) - 1] + 1;
			r.push({
				id: t.id,
				start: e,
				end: Math.max(a, e + 1),
				value: this.vocab.tokenOfId(t.id)
			}), i += t.len;
		}
		return r;
	}
	mergeWord(e) {
		let t = this.opt, n = [], r = null;
		for (let i = 0; i < e.length; i += 1) {
			let a = e[i];
			i > 0 && t.continuingSubwordPrefix !== null && (a = t.continuingSubwordPrefix + a), i === e.length - 1 && t.endOfWordSuffix !== null && (a += t.endOfWordSuffix);
			let o = this.vocab.id(a);
			if (o !== void 0) {
				r &&= (n.push(r), null), n.push({
					id: o,
					len: d(e[i])
				});
				continue;
			}
			if (t.byteFallback) {
				let e = [], t = !0;
				for (let n of new TextEncoder().encode(a)) {
					let r = this.vocab.id(`<0x${n.toString(16).toUpperCase().padStart(2, "0")}>`);
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
				let a = this.vocab.id(t.unkToken);
				r && t.fuseUnk ? r = {
					id: r.id,
					len: r.len + d(e[i])
				} : (r && n.push(r), r = {
					id: a,
					len: d(e[i])
				});
			}
		}
		return r && n.push(r), this.mergeAll(n);
	}
	mergeAll(e) {
		let t = e;
		for (;;) {
			let e = Infinity, n = -1, r = 0;
			for (let i = 0; i + 1 < t.length; i += 1) {
				let a = this.ranks.get(t[i].id, t[i + 1].id);
				a !== -1 && a < e && (e = a, n = i, r = this.mergedIds[a]);
			}
			if (n < 0) return t;
			t = [
				...t.slice(0, n),
				{
					id: r,
					len: t[n].len + t[n + 1].len
				},
				...t.slice(n + 2)
			];
		}
	}
}, S = /^[\p{Cc}\p{Cf}\p{Cs}\p{Co}]$/u, C = /^\p{White_Space}$/u, w = /^\p{Mn}$/u;
function T(e) {
	let t = e.charCodeAt(0);
	return t < 128 ? t === 32 || t >= 9 && t <= 13 : C.test(e);
}
function ee(e) {
	return e === 2192 || e === 2193 || e === 2274 || e === 69837 || e >= 78896 && e <= 78911;
}
function te(e) {
	return e === "	" || e === "\n" || e === "\r" ? !1 : S.test(e) && !ee(e.codePointAt(0));
}
function E(e) {
	return e >= 19968 && e <= 40959 || e >= 13312 && e <= 19903 || e >= 131072 && e <= 173791 || e >= 173824 && e <= 177983 || e >= 177984 && e <= 178207 || e >= 178464 && e <= 183983 || e >= 63744 && e <= 64255 || e >= 194560 && e <= 195103;
}
function D(e) {
	return {
		cleanText: e.clean_text !== !1,
		handleChineseChars: e.handle_chinese_chars !== !1,
		stripAccents: e.strip_accents ?? null,
		lowercase: e.lowercase !== !1
	};
}
function O(e, t) {
	let n = [], r = [], i = [];
	for (let a = 0; a < e.chars.length; a += 1) {
		let o = e.chars[a];
		(!t.cleanText || o !== "\0" && o !== "�" && !te(o)) && (n.push(t.cleanText && T(o) ? " " : o), r.push(e.starts[a]), i.push(e.ends[a]));
	}
	if (t.handleChineseChars) {
		let e = [], t = [], a = [];
		for (let o = 0; o < n.length; o += 1) {
			let s = n[o];
			E(s.codePointAt(0)) ? (e.push(" ", s, " "), t.push(r[o], r[o], r[o]), a.push(i[o], i[o], i[o])) : (e.push(s), t.push(r[o]), a.push(i[o]));
		}
		n = e, r = t, i = a;
	}
	if (t.stripAccents ?? t.lowercase) {
		let e = [], t = [], a = [];
		for (let o = 0; o < n.length; o += 1) for (let s of n[o].normalize("NFD")) w.test(s) || (e.push(s), t.push(r[o]), a.push(i[o]));
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
var k = /* @__PURE__ */ new Map(), A = /* @__PURE__ */ new Map();
function j(e) {
	let t = k.get(e);
	return t === void 0 && (t = e === "ͅ" || `x\u0345${e}`.normalize("NFD") !== `x\u0345${e}`, k.set(e, t)), t;
}
function M(e, t) {
	let n = e + t, r = A.get(n);
	return r === void 0 && (r = `x${t}${e}`.normalize("NFD") !== `x${t}${e}`, A.set(n, r)), r;
}
var N = /* @__PURE__ */ new Map();
function ne(e, t) {
	let n = e + "\0" + t, r = N.get(n);
	if (r === void 0) {
		let i = (e + t).normalize("NFC");
		r = [...i].length === 1 ? i : null, N.set(n, r);
	}
	return r;
}
function re(e, t) {
	let n = t === "NFC" ? "NFD" : "NFKD", r = [];
	for (let t of e.chars) [...t.normalize(n)].forEach((e, t) => r.push([e, t === 0 ? 0 : 1]));
	for (let e = 0; e < r.length;) {
		if (!j(r[e][0])) {
			e += 1;
			continue;
		}
		let t = e;
		for (; t < r.length && j(r[t][0]);) t += 1;
		for (let n = e + 1; n < t; n += 1) {
			let t = r[n], i = n - 1;
			for (; i >= e && M(t[0], r[i][0]);) r[i + 1] = r[i], --i;
			r[i + 1] = t;
		}
		e = t;
	}
	let i = [], a = -1, o = null, s = !1;
	for (let [e, t] of r) {
		let n = j(e);
		if (a >= 0) {
			let r = ne(i[a][0], e), c = s || o !== null && n && M(o, e);
			if (r !== null && c) {
				i[a] = [r, i[a][1] + t - 1];
				continue;
			}
		}
		n || (a = i.length), o = e, s = !n, i.push([e, t]), i.length === 1 && n && (o = null, a = -1);
	}
	return f(e, i);
}
function ie(e, t, n) {
	let r = e.chars.length, i = 0;
	if (t) for (; i < r && T(e.chars[i]);) i += 1;
	let a = 0;
	if (n) for (; a < r && T(e.chars[r - 1 - a]);) a += 1;
	if (i === 0 && a === 0) return e;
	let o = Math.min(i, r), s = Math.max(o, r - a);
	return {
		chars: e.chars.slice(o, s),
		starts: e.starts.slice(o, s),
		ends: e.ends.slice(o, s),
		origin: e.origin
	};
}
var P = "\\p{L}\\p{M}\\p{N}\\p{Pc}", F = "\\p{White_Space}", ae = `(?:(?<=[${P}])(?![${P}])|(?<![${P}])(?=[${P}]))`, oe = `(?:(?<=[${P}])(?=[${P}])|(?<![${P}])(?![${P}]))`;
function se(e) {
	let t = "", n = !1;
	for (let r = 0; r < e.length; r += 1) {
		let i = e[r];
		if (i === "\\") {
			let a = e[r + 1];
			r += 1;
			let o = {
				w: P,
				s: F
			}, s = {
				W: P,
				S: F
			};
			if (a in o) t += n ? o[a] : `[${o[a]}]`;
			else if (a in s) {
				if (n) throw Error(`Replace: unsupported \\${a} inside a character class in ${e}`);
				t += `[^${s[a]}]`;
			} else if (a === "d" || a === "D") throw Error(`Replace: unsupported \\${a} in ${e}`);
			else t += (a === "b" || a === "B") && !n ? a === "b" ? ae : oe : i + a;
		} else i === "[" ? n = !0 : i === "]" && (n = !1), t += i;
	}
	return t;
}
function I(e) {
	return e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
var L = class {
	trie;
	blob;
	cache = /* @__PURE__ */ new Map();
	encoder = new TextEncoder();
	decoder = new TextDecoder();
	constructor(e) {
		let t = typeof e == "string" ? Uint8Array.from(atob(e), (e) => e.charCodeAt(0)) : e, n = new DataView(t.buffer, t.byteOffset, t.byteLength), r = n.getUint32(0, !0);
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
			if (R(e) < 6) {
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
		return r ? f(e, n) : e;
	}
};
function R(e) {
	let t = 0;
	for (let n of e) t += d(n);
	return t;
}
function z(e) {
	if (e == null) return null;
	let t = String(e.type);
	switch (t) {
		case "Sequence": {
			let t = e.normalizers.map((e) => z(e));
			return (e) => t.reduce((e, t) => t(e), e);
		}
		case "BertNormalizer": {
			let t = D(e);
			return (e) => O(e, t);
		}
		case "NFC":
		case "NFKC": return (e) => re(e, t);
		case "Strip": {
			let t = e.strip_left !== !1, n = e.strip_right !== !1;
			return (e) => ie(e, t, n);
		}
		case "Replace": {
			let t = e.pattern, n = t.String === void 0 ? se(t.Regex) : I(t.String);
			if (n === "") throw Error("Replace: empty pattern");
			let r = new RegExp(n, "gu"), i = e.content;
			return (e) => v(e, g(e.chars, r), i);
		}
		case "Precompiled": {
			let t = new L(e.precompiled_charsmap), n = new Intl.Segmenter("en", { granularity: "grapheme" });
			return (e) => t.normalize(e, n);
		}
		default: throw Error(`tokenizer.json: unsupported normalizer type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/postprocessors.ts
var B = class e {
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
}, V = "Ġ";
function H(e, t) {
	let n = e.offsets.slice();
	for (let r = 0; r < e.ids.length; r += 1) {
		let i = [...e.tokens[r]], a = 0;
		for (; a < i.length && (i[a] === V || T(i[a]));) a += 1;
		let o = 0;
		for (; o < i.length && (i[i.length - 1 - o] === V || T(i[i.length - 1 - o]));) o += 1;
		if (a === 0 && o === 0) continue;
		let s = n[2 * r], c = n[2 * r + 1];
		a > 0 && ((r === 0 || s === 0) && t && a === 1 && (a = 0), s = Math.min(s + a, c)), o > 0 && c >= o && (c = Math.max(c - o, s)), n[2 * r] = s, n[2 * r + 1] = c;
	}
	return n;
}
function U(e, t) {
	e.ids.push(t), e.typeIds.push(0), e.attentionMask.push(1), e.offsets.push([0, 0]), e.wordIds.push(-1), e.sequenceIds.push(-1), e.specialTokensMask.push(1);
}
function W(e, t, n, r, i) {
	for (let a = 0; a < t.ids.length; a += 1) e.ids.push(t.ids[a]), e.typeIds.push(i), e.attentionMask.push(1), e.offsets.push([n[2 * a], n[2 * a + 1]]), e.wordIds.push(t.wordIds[a]), e.sequenceIds.push(r), e.specialTokensMask.push(0);
}
function G() {
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
var K = class e {
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
		let n = G();
		return U(n, this.cls), W(n, e, this.trim ? H(e, this.addPrefixSpace) : e.offsets, 0, 0), U(n, this.sep), t && (U(n, this.sep), W(n, t, this.trim ? H(t, this.addPrefixSpace) : t.offsets, 1, 0), U(n, this.sep)), n;
	}
}, ce = class e {
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
		let n = G();
		return W(n, e, this.trim ? H(e, this.addPrefixSpace) : e.offsets, 0, 0), t && W(n, t, this.trim ? H(t, this.addPrefixSpace) : t.offsets, 1, 1), n;
	}
};
function le(e) {
	let t = String(e?.type);
	switch (t) {
		case "TemplateProcessing": return B.fromJson(e);
		case "RobertaProcessing": return K.fromJson(e);
		case "ByteLevel": return ce.fromJson(e);
		default: throw Error(`tokenizer.json: unsupported post_processor type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/pretokenizers.ts
var ue = /^\p{P}$/u, de = /^[!-/:-@[-`{-~]$/;
function fe(e) {
	return e.charCodeAt(0) < 128 ? de.test(e) : ue.test(e);
}
function pe(e) {
	let t = [], n = -1;
	for (let r = 0; r < e.chars.length; r += 1) {
		let i = e.chars[r];
		T(i) ? (n >= 0 && t.push({
			start: n,
			end: r
		}), n = -1) : fe(i) ? (n >= 0 && t.push({
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
function me(e) {
	return pe(e).map((t) => p(e, t.start, t.end));
}
var he = /* @__PURE__ */ RegExp("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+|\\p{White_Space}+(?!\\P{White_Space})|\\p{White_Space}+", "gu"), q = (() => {
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
q[32];
var ge = new TextEncoder();
function _e(e) {
	let t = [], n = [], r = [];
	for (let i = 0; i < e.chars.length; i += 1) {
		let a = d(e.chars[i]) === 1 ? [e.chars[i].charCodeAt(0)] : ge.encode(e.chars[i]);
		for (let o of a) t.push(q[o]), n.push(e.starts[i]), r.push(e.ends[i]);
	}
	return {
		chars: t,
		starts: n,
		ends: r,
		origin: e.origin
	};
}
function ve(e, t) {
	return (n) => {
		let r = n;
		return e && r.chars[0] !== " " && (r = y(r, " ")), (t ? _(r, g(r.chars, he), "isolated") : [r]).map(_e);
	};
}
function ye(e, t, n) {
	let r = (e) => e === " ";
	return (i) => {
		let a = v(i, m(i.chars, r), e);
		return (t === "always" && a.chars[0] !== e || t === "first" && a.chars[0] !== e && (a.origin ?? 0) === 0) && (a = y(a, e)), n ? _(a, m(a.chars, (t) => t === e), "merged_with_next") : [a];
	};
}
function be(e) {
	return _(e, m(e.chars, T), "removed");
}
function J(e) {
	if (e == null) return null;
	let t = String(e.type);
	switch (t) {
		case "Sequence": {
			let t = e.pretokenizers.map((e) => J(e));
			return (e) => t.reduce((e, t) => e.flatMap((e) => t(e)).filter((e) => e.chars.length > 0), [e]);
		}
		case "BertPreTokenizer": return me;
		case "WhitespaceSplit": return be;
		case "ByteLevel": return ve(e.add_prefix_space !== !1, e.use_regex !== !1);
		case "Metaspace": {
			let t = e.replacement;
			if (e.add_prefix_space === !1 && e.prepend_scheme !== void 0 && e.prepend_scheme !== "never") throw Error("Metaspace: add_prefix_space does not match declared prepend_scheme");
			return ye(t, e.add_prefix_space === !1 ? "never" : e.prepend_scheme ?? "always", e.split !== !1);
		}
		default: throw Error(`tokenizer.json: unsupported pre_tokenizer type ${t}`);
	}
}
//#endregion
//#region src/tokenizer/hf/truncation.ts
var Y = "Truncation error: Sequence to truncate too short to respect the provided max_length";
function xe(e, t, n, r) {
	let i = t ?? 0, a = e + i;
	if (n <= 0) return [0, 0];
	if (a <= n) return [e, i];
	let o = a - n;
	if (r === "longest_first") {
		let t = Math.abs(e - i), n = Math.min(t, o), r = o - n, a = e <= i, s = Math.ceil(r / 2), c = n + Math.floor(r / 2);
		return [e - (a ? s : c), i - (a ? c : s)];
	}
	if (r === "only_first") {
		if (e <= o) throw Error(Y);
		return [e - o, i];
	}
	if (t === null) throw Error("Truncation error: Second sequence not provided");
	if (t <= o) throw Error(Y);
	return [e, t - o];
}
//#endregion
//#region src/tokenizer/hf/unigram.ts
var Se = 10;
function Ce(e, t, n) {
	let r = 0;
	for (let i = t; i < n; i += 1) {
		let t = e.charCodeAt(i);
		if (t >= 55296 && t < 56320 && i + 1 < n) {
			let t = e.charCodeAt(i + 1);
			t >= 56320 && t < 57344 && (i += 1);
		}
		r += 1;
	}
	return r;
}
var we = class n {
	index;
	minScore = Infinity;
	maxChars = 0;
	unkId;
	byteFallback;
	byId;
	constructor(e, t, n) {
		this.unkId = t, this.byteFallback = n;
		let i = e.pieces.length;
		if (t !== null && (i === 0 || t >= i)) throw Error("Unigram: unk_id is not in the vocab");
		this.byId = Float64Array.from(e.scores);
		let { blob: a, offs: o } = e.pieces;
		for (let e = 0; e < i; e += 1) {
			let t = this.byId[e];
			t < this.minScore && (this.minScore = t), this.maxChars = Math.max(this.maxChars, Ce(a, o[e], o[e + 1]));
		}
		this.index = new r(e.pieces);
	}
	static fromJson(t) {
		return new n(e(t.vocab), t.unk_id ?? null, t.byte_fallback === !0);
	}
	idOf(e) {
		let t = this.index.indexOf(e);
		return t < 0 ? void 0 : t;
	}
	tokenize(e) {
		if (e.length === 0) return [];
		let n = e.length, r = this.minScore - Se, i = Array(n + 1).fill(0), a = Array(n + 1).fill(-1), o = Array(n + 1).fill(0);
		for (let s = 0; s < n; s += 1) {
			let c = i[s], u = !1, d = t;
			for (let t = 1; t <= this.maxChars && s + t <= n; t += 1) {
				let n = s + t;
				d = l(d, e[n - 1]);
				let r = this.index.indexOfChars(e, s, n, d);
				if (r < 0) continue;
				let f = this.byId[r] + c;
				(a[n] < 0 || f > i[n]) && (i[n] = f, a[n] = s, o[n] = r), t === 1 && (u = !0);
			}
			if (!u) {
				if (this.unkId === null) throw Error("Unigram: the vocab has no unk id");
				let e = r + c;
				(a[s + 1] < 0 || e > i[s + 1]) && (i[s + 1] = e, a[s + 1] = s, o[s + 1] = this.unkId);
			}
		}
		let s = [], c = n, u = -1, d = -1;
		for (; c > 0;) {
			let e = a[c];
			this.unkId !== null && o[c] === this.unkId ? (u < 0 && (u = c), d = e) : (u >= 0 && (s.push([d, u]), u = -1), s.push([e, c])), c = e;
		}
		u >= 0 && s.push([d, u]), s.reverse();
		let f = [];
		for (let [t, n] of s) {
			let r = e.slice(t, n).join(""), i = this.idOf(r);
			if (i !== void 0) {
				f.push({
					id: i,
					start: t,
					end: n
				});
				continue;
			}
			if (this.byteFallback) {
				let e = new TextEncoder().encode(r), i = [], a = !0;
				for (let t of e) {
					let e = this.idOf(`<0x${t.toString(16).toUpperCase().padStart(2, "0")}>`);
					if (e === void 0) {
						a = !1;
						break;
					}
					i.push(e);
				}
				if (a) {
					for (let e of i) f.push({
						id: e,
						start: t,
						end: n
					});
					continue;
				}
			}
			if (this.unkId === null) throw Error("Unigram: the vocab has no unk id");
			f.push({
				id: this.unkId,
				start: t,
				end: n
			});
		}
		return f;
	}
}, Te = class e {
	vocab;
	unkToken;
	prefix;
	maxInputCharsPerWord;
	constructor(e, t, n, r) {
		if (this.vocab = e, this.unkToken = t, this.prefix = n, this.maxInputCharsPerWord = r, !e.has(t)) throw Error(`WordPiece: unk token ${t} is not in the vocab`);
	}
	static fromJson(t) {
		let n = new i(o(t.vocab));
		return new e(n, t.unk_token ?? "[UNK]", t.continuing_subword_prefix ?? "##", t.max_input_chars_per_word ?? 100);
	}
	id(e) {
		return this.vocab.id(e);
	}
	tokenize(e) {
		let t = {
			id: this.vocab.id(this.unkToken),
			start: 0,
			end: e.length
		};
		if (e.length > this.maxInputCharsPerWord) return [t];
		let n = [], r = 0;
		for (; r < e.length;) {
			let i = e.length, a = null;
			for (; r < i;) {
				let t = (r > 0 ? this.prefix : "") + e.slice(r, i).join(""), n = this.vocab.id(t);
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
}, X = /^[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]$/u;
function Ee(e) {
	return e == null ? null : String(e.type);
}
function De(e) {
	return e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function Z(e) {
	return new TextEncoder().encode(e).length;
}
function Oe(e) {
	let t = Ee(e);
	if (t === "WordPiece") return Te.fromJson(e);
	if (t === "BPE") return x.fromJson(e);
	if (t === "Unigram") return we.fromJson(e);
	throw Error(`tokenizer.json: unsupported model type ${t}`);
}
function Q(e, t) {
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
				let t = [...e.slice(0, a)].pop(), n = [...e.slice(o)][0], r = a === 0 || !X.test(t);
				if (o !== e.length && X.test(n) || !r) continue;
			}
			if (s.lstrip) {
				let t = a;
				for (; t > 0;) {
					let n = [...e.slice(Math.max(0, t - 2), t)].pop();
					if (!T(n)) break;
					t -= n.length;
				}
				a = Math.max(t, r);
			}
			if (s.rstrip) {
				let t = o;
				for (; t < e.length;) {
					let n = String.fromCodePoint(e.codePointAt(t));
					if (!T(n)) break;
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
function $(e, t) {
	if (e.length === 0) return {
		tokens: e,
		regex: null
	};
	let n = e.map((e, t) => t).sort((e, n) => Z(t[n]) - Z(t[e]) || e - n);
	return {
		tokens: n.map((t) => e[t]),
		regex: new RegExp(n.map((e) => `(${De(t[e])})`).join("|"), "g")
	};
}
var ke = class e {
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
		let n = z(t.normalizer), r = J(t.pre_tokenizer), i = Oe(t.model), a = le(t.post_processor), o = (t.added_tokens ?? []).map((e) => ({
			id: e.id,
			content: e.content,
			lstrip: e.lstrip === !0,
			rstrip: e.rstrip === !0,
			singleWord: e.single_word === !0,
			normalized: e.normalized === !0,
			special: e.special === !0
		})).filter((e) => e.content !== ""), s = [...o.filter((e) => e.special), ...o.filter((e) => !e.special)], c = s.filter((e) => !e.normalized), l = s.filter((e) => e.normalized), d = l.map((e) => {
			let t = u(e.content);
			return b(n ? n(t) : t);
		});
		return d.forEach((e, t) => {
			if (e === "") throw Error(`tokenizer.json: added token ${l[t].id} is empty after normalization`);
		}), new e(n, r, i, a, $(c, c.map((e) => e.content)), $(l, d));
	}
	static fromString(t) {
		return e.fromJson(JSON.parse(t));
	}
	static async fromBytes(t) {
		return e.fromJson(await s(t));
	}
	split(e) {
		let t = [];
		for (let n of Q(e, this.plain)) {
			let r = u(e.slice(n.start, n.end), n.start);
			if (n.token) {
				t.push({
					token: n.token,
					span: r
				});
				continue;
			}
			let i = this.normalizer ? this.normalizer(r) : r;
			if (i.chars.length === 0) continue;
			let a = h(i.chars);
			for (let e of Q(b(i), this.normalizedSet)) {
				let n = a[e.start], r = a[e.end];
				n !== r && t.push(e.token ? {
					token: e.token,
					span: p(i, n, r)
				} : { text: p(i, n, r) });
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
				t.ids.push(r.token.id), t.offsets.push(e.starts[0], e.ends[e.chars.length - 1]), t.wordIds.push(n), t.tokens.push(b(e)), n += 1;
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
			let e = n.maxLength - this.post.addedTokens(i !== null), [t, o] = xe(r.ids.length, i ? i.ids.length : null, e, a), s = (e, t) => t >= e.ids.length ? e : {
				ids: e.ids.slice(0, t),
				offsets: e.offsets.slice(0, 2 * t),
				wordIds: e.wordIds.slice(0, t),
				tokens: e.tokens.slice(0, t)
			};
			r = s(r, t), i &&= s(i, o);
		}
		return this.post.apply(r, i);
	}
};
//#endregion
export { ke as JsonTokenizer };
