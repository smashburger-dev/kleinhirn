//#region src/wasm-backend.ts
function e() {
	let e = new Worker(new URL(
		/* @vite-ignore */
		"" + new URL("assets/wasm-plan-worker-DZA_w1EI.js", import.meta.url).href,
		"" + import.meta.url
	), { type: "module" }), t = /* @__PURE__ */ new Map(), n = 0, r = (e) => {
		for (let n of t.values()) n.reject(e);
		t.clear();
	};
	return e.onmessage = (e) => {
		let n = t.get(e.data.id);
		t.delete(e.data.id), n?.resolve(e.data);
	}, e.onerror = (e) => r(/* @__PURE__ */ Error(`wasm worker failed: ${e.message ?? "script error"}`)), {
		request(r, i = []) {
			let a = n += 1;
			return new Promise((n, o) => {
				t.set(a, {
					resolve: n,
					reject: o
				}), e.postMessage({
					id: a,
					msg: r
				}, i);
			});
		},
		close() {
			e.terminate(), r(/* @__PURE__ */ Error("wasm worker closed"));
		}
	};
}
var t = (e) => {
	if (e.error !== void 0) throw Error(e.error);
	return e;
}, n = class {
	backend;
	pid;
	length;
	batch;
	markers;
	bytes = 0;
	input;
	out;
	constructor(e, t, n) {
		this.backend = e, this.pid = n, this.length = t.length, this.batch = t.batch, this.markers = t.markers;
	}
	upload(e) {
		this.input = e;
	}
	run(e, t = !1) {
		t && this.assertCapturable();
		let n = this.input;
		if (!n) throw Error("run before upload");
		this.input = void 0;
		let r = [
			n.embeddings,
			n.mask,
			n.typeIds,
			n.packedMarkers
		].filter((e) => e !== void 0).map((e) => e.buffer);
		this.out = this.backend.request({
			type: "run",
			pid: this.pid,
			input: n,
			rows: e
		}, r).then((e) => e.out);
	}
	readOutput() {
		let e = this.out;
		return e ? (this.out = void 0, e) : Promise.reject(/* @__PURE__ */ Error("readOutput before run"));
	}
	readCapture() {
		return Promise.reject(/* @__PURE__ */ Error("layer capture is a WebGPU parity tool"));
	}
	assertCapturable() {
		throw Error("layer capture is a WebGPU parity tool; the WASM path has none");
	}
	destroy() {
		this.backend.request({
			type: "destroy",
			pid: this.pid
		}).catch(() => {});
	}
}, r = class {
	t;
	kind = "wasm";
	weightBytes = 0;
	build;
	threads = 1;
	threadNote;
	batchPlans = !1;
	memoryBytes = 0;
	pid = 0;
	constructor(e) {
		this.t = e;
	}
	async request(e, n) {
		let r = t(await this.t.request(e, n));
		return r.memoryBytes !== void 0 && (this.memoryBytes = r.memoryBytes), r.weightBytes !== void 0 && (this.weightBytes = r.weightBytes), r;
	}
	check(e) {
		if (e.f16) throw Error("the WASM path runs f32 plans");
	}
	runner(e) {
		let t = this.pid += 1;
		return this.t.request({
			type: "plan",
			pid: t,
			plan: e
		}).catch(() => {}), new n(this, e, t);
	}
	fitBatch(e, t) {
		if (this.threads <= 1 || !this.batchPlans) return;
		let n = [
			16,
			8,
			4
		].find((e) => e <= t);
		return n ? e(n) : void 0;
	}
	async prepare(e) {
		let t = e();
		return await this.request({ type: "sync" }), t;
	}
	async call(e, t) {
		return t(e());
	}
	info() {
		return {
			backend: "wasm",
			threads: this.threads,
			...this.threadNote ? { threadNote: this.threadNote } : {},
			wasmBuild: this.build,
			wasmMemoryBytes: this.memoryBytes
		};
	}
	dispose() {
		this.t.request({ type: "close" }).catch(() => {}).finally(() => this.t.close());
	}
};
async function i(e, t, n, i = {}, a) {
	let o = performance.now(), s = new r(n), c = await s.request({
		type: "init",
		shards: e.shardBytes,
		tensors: t.tensors.filter((e) => !e.keepOnCpu),
		build: i.build,
		threads: i.threads,
		split: i.split
	}, e.shardBytes);
	return s.build = c.build, s.threads = c.threads ?? 1, s.threadNote = c.threadNote, s.batchPlans = i.batchPlans === !0, {
		backend: s,
		words: a.table,
		downloadBytes: e.downloadBytes + a.bytes,
		timing: {
			fetchMs: e.fetchMs,
			bodyMs: e.bodyMs,
			sha256Ms: e.sha256Ms,
			uploadMs: performance.now() - o,
			...a.timing
		}
	};
}
//#endregion
export { i as loadWasm, e as workerTransport };
