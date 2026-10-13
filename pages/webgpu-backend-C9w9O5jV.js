import { C as e, S as t, a as n, c as r, l as i, s as a } from "./kleinhirn.js";
//#region src/plan/check.ts
var o = {
	add: () => 0,
	attention: (e) => 4 * e + 256,
	attpv: () => 8200,
	attrel: () => 12292,
	attscore: () => 12292,
	attsoftmax: () => 256,
	attsoftrel: () => 256,
	embln: (e, t) => 4 * t.N + 256,
	gather: () => 0,
	geglu: () => 0,
	im2col: () => 0,
	layernorm: (e, t) => 4 * t.N + 256,
	masklogits: () => 0,
	matmul: () => 2048,
	mbattention: (e) => 4 * e + 256,
	mbflash: () => 16384,
	mmtile: () => 12288,
	mmtile16: () => 6144,
	mmtile8: () => 5120,
	pool: () => 0,
	rope: () => 0
}, s = (e, t) => e === "rows" ? t : e === "rows8" ? Math.ceil(t / 8) : e === "rows16" ? Math.ceil(t / 16) : e === "rows32" ? Math.ceil(t / 32) : e;
function c(e, t, r, i = {}) {
	let a = t.maxStorageBufferBindingSize ?? n.maxStorageBufferBindingSize, c = t.maxBufferSize ?? n.maxBufferSize, l = t.maxComputeWorkgroupsPerDimension ?? n.maxComputeWorkgroupsPerDimension, u = t.maxComputeWorkgroupStorageSize ?? n.maxComputeWorkgroupStorageSize, d = t.maxStorageBuffersPerShaderStage ?? n.maxStorageBuffersPerShaderStage, f = `bucket L${e.length} B${e.batch}`, p = [], m = (e, t, n, r) => {
		t > r && p.push(`${f}: ${e} needs ${t}, ${n} is ${r}`);
	}, h = e.segments.flatMap((e) => [...e.captureOps ?? [], ...e.ops].map((t) => ({
		op: t,
		at: `${e.prefix}${t.name}`
	})));
	!i.batchOnly && (!Number.isInteger(e.length) || e.length <= 0) && p.push(`${f}: the length must be a positive integer`), (!Number.isInteger(e.batch) || e.batch <= 0) && p.push(`${f}: the batch must be a positive integer`);
	let g = e.f16 ? 2 : 4, _ = e.length * e.batch * e.embeddingSize * g;
	_ % 4 != 0 && p.push(`${f}: the word row upload of ${_} B is not a multiple of 4 (writeBuffer needs it)`);
	for (let t of e.buffers) m(`buffer ${t.id}`, t.bytes, "maxStorageBufferBindingSize", a), m(`buffer ${t.id}`, t.bytes, "maxBufferSize", c);
	m("the output staging buffer", e.output.bytes, "maxBufferSize", c);
	let v = e.length * e.batch;
	for (let { op: t, at: n } of h) {
		for (let e of t.dispatch) m(`dispatch ${n}`, s(e, v), "maxComputeWorkgroupsPerDimension", l);
		for (let e of t.alts ?? []) for (let t of e.dispatch) m(`dispatch ${n} (${e.kernel})`, s(t, v), "maxComputeWorkgroupsPerDimension", l);
		i.batchOnly || (m(`op ${n}`, t.bind.length, "maxStorageBuffersPerShaderStage", d), m(`workgroup memory of ${n} (${t.kernel})`, o[t.kernel](e.length, t.constants), "maxComputeWorkgroupStorageSize", u));
	}
	if (r && !i.batchOnly) {
		let e = /* @__PURE__ */ new Set();
		for (let { op: t } of h) for (let n of t.bind) {
			if (!n.startsWith("w:") || e.has(n)) continue;
			e.add(n);
			let t = r(n.slice(2));
			t !== void 0 && m(`weight ${n.slice(2)}`, t, "maxStorageBufferBindingSize", a);
		}
	}
	return [...new Set(p)];
}
function l(e, t, n) {
	let r = c(e, t, n);
	if (r.length) throw Error(r.join("; "));
}
function u(e, t, n) {
	for (let r of [...i].reverse()) {
		if (r === 1 || r > n) continue;
		let i = e(r);
		if (c(i, t, void 0, { batchOnly: !0 }).length === 0) return i;
	}
}
//#endregion
//#region src/half.ts
function d(e) {
	let t = e & 32768 ? -1 : 1, n = e >> 10 & 31, r = e & 1023;
	return n === 0 ? t * r * 2 ** -24 : n === 31 ? r === 0 ? t * Infinity : NaN : t * (1 + r / 1024) * 2 ** (n - 15);
}
function f(e, t = !0) {
	let n = new Float32Array(e.length), r = globalThis.Float16Array;
	if (t && r) {
		let t = new r(e.buffer, e.byteOffset, e.length);
		for (let r = 0; r < e.length; r += 1) n[r] = t[r];
		return n;
	}
	for (let t = 0; t < e.length; t += 1) n[t] = d(e[t]);
	return n;
}
//#endregion
//#region src/kernels/add.wgsl?raw
var p = "{{ENABLE}}override TOTAL:u32=1u;override N:u32=384u;override MODE:u32=0u;override L:u32=128u;@group(0)@binding(0)var<storage,read_write>dst:array<{{F}}>;@group(0)@binding(1)var<storage,read>src:array<{{F}}>;@compute@workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3<u32>,@builtin(num_workgroups)nwg:vec3<u32>,){for(var i=gid.x;i<TOTAL;i+=64u*nwg.x){let s=select(src[i],src[(i/(N*L))*N+i%N],MODE==1u);dst[i]={{F}}(f32(dst[i])+f32(s));}}", m = "{{ENABLE}}override L:u32=128u;override H:u32=6u;override D:u32=64u;override SCALE:f32=13.856406;@group(0)@binding(0)var<storage,read>qkv:array<{{F}}>;@group(0)@binding(1)var<storage,read>posKey:array<{{F}}>;@group(0)@binding(2)var<storage,read>posQuery:array<{{F}}>;@group(0)@binding(3)var<storage,read>relidx:array<u32>;@group(0)@binding(4)var<storage,read>mask:array<f32>;@group(0)@binding(5)var<storage,read_write>ctx:array<{{F}}>;var<workgroup>scores:array<f32,L>;var<workgroup>red:array<f32,64>;const LOWEST=-3.40282346638528859812e38f;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let h=wid.x;let i=wid.y;let b=i/L;let il=i-b*L;let kbase=b*L;let hd=H*D;let qb=i*3u*hd+h*D;var qreg:array<f32,64>;for(var d=0u;d<D;d+=1u){qreg[d]=f32(qkv[qb+d]);}if(mask[i]<=0.5){for(var d=lid.x;d<D;d+=64u){ctx[i*hd+h*D+d]={{F}}(0.0);}return;}for(var j=lid.x;j<L;j+=64u){if(mask[kbase+j]<=0.5){scores[j]=0.0;continue;}var s=0.0;let p=relidx[il*L+j]*hd;let kb=(kbase+j)*3u*hd+(H+h)*D;let pk=p+h*D;let pq=p+h*D;for(var d=0u;d+3u<D;d+=4u){let q0=qreg[d];let q1=qreg[d+1u];let q2=qreg[d+2u];let q3=qreg[d+3u];let k0=f32(qkv[kb+d]);let k1=f32(qkv[kb+d+1u]);let k2=f32(qkv[kb+d+2u]);let k3=f32(qkv[kb+d+3u]);s+=q0*k0+q1*k1+q2*k2+q3*k3;s+=q0*f32(posKey[pk+d])+q1*f32(posKey[pk+d+1u])+q2*f32(posKey[pk+d+2u])+q3*f32(posKey[pk+d+3u]);s+=k0*f32(posQuery[pq+d])+k1*f32(posQuery[pq+d+1u])+k2*f32(posQuery[pq+d+2u])+k3*f32(posQuery[pq+d+3u]);}for(var d=D-D%4u;d<D;d+=1u){let q0=qreg[d];let k0=f32(qkv[kb+d]);s+=q0*k0;s+=q0*f32(posKey[pk+d]);s+=k0*f32(posQuery[pq+d]);}scores[j]=s/SCALE;}workgroupBarrier();var m=LOWEST;for(var j=lid.x;j<L;j+=64u){if(mask[kbase+j]>0.5){m=max(m,scores[j]);}}red[lid.x]=m;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]=max(red[lid.x],red[lid.x+o]);}workgroupBarrier();}let mx=red[0];workgroupBarrier();var sum=0.0;for(var j=lid.x;j<L;j+=64u){var e=0.0;if(mask[kbase+j]>0.5){e=exp(scores[j]-mx);}scores[j]=e;sum+=e;}red[lid.x]=sum;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let invTotal=select(0.0,1.0/red[0],red[0]>0.0);workgroupBarrier();for(var d=lid.x;d<D;d+=64u){let vb=(2u*H+h)*D+d;var acc0=0.0;var acc1=0.0;var acc2=0.0;var acc3=0.0;for(var j=0u;j+3u<L;j+=4u){let s0=scores[j];let s1=scores[j+1u];let s2=scores[j+2u];let s3=scores[j+3u];if(s0!=0.0){acc0+=s0*f32(qkv[(kbase+j+0u)*3u*hd+vb]);}if(s1!=0.0){acc1+=s1*f32(qkv[(kbase+j+1u)*3u*hd+vb]);}if(s2!=0.0){acc2+=s2*f32(qkv[(kbase+j+2u)*3u*hd+vb]);}if(s3!=0.0){acc3+=s3*f32(qkv[(kbase+j+3u)*3u*hd+vb]);}}for(var j=L-L%4u;j<L;j+=1u){let s0=scores[j];if(s0!=0.0){acc0+=s0*f32(qkv[(kbase+j)*3u*hd+vb]);}}var acc=(acc0+acc1+acc2+acc3)*invTotal;ctx[i*hd+h*D+d]={{F}}(acc);}}", h = "{{ENABLE}}override L:u32=128u;override H:u32=12u;override D:u32=64u;override ROWS:u32=128u;@group(0)@binding(0)var<storage,read>probs:array<vec4<f32>>;@group(0)@binding(1)var<storage,read>qkv:array<vec4<{{F}}>>;@group(0)@binding(2)var<storage,read>mask:array<f32>;@group(0)@binding(3)var<storage,read>kinfo:array<u32>;@group(0)@binding(4)var<storage,read_write>ctx:array<{{F}}>;var<workgroup>sa:array<vec4<f32>,256>;var<workgroup>sb:array<vec4<{{F}}>,256>;var<workgroup>keyFirst:u32;var<workgroup>keyLast:u32;@compute@workgroup_size(8,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let nd=D/32u;let h=wid.x/nd;let dt=wid.x%nd;let tid=lid.y*8u+lid.x;let row0=wid.y*32u;let kbase=(row0/L)*L;let tileRow=lid.y*4u;let L4=L/4u;let D4=D/4u;let row4=3u*H*D4;var acc:array<vec4<f32>,4>;if(tid==0u){keyFirst=kinfo[2u*(row0/L)];}let first=workgroupUniformLoad(&keyFirst);if(tid==0u){keyLast=kinfo[2u*(row0/L)+1u];}let last=workgroupUniformLoad(&keyLast);for(var t4=0u;t4<L4;t4+=8u){if(first>=L||t4*4u>last||t4*4u+31u<first){continue;}for(var run=tid;run<256u;run+=64u){let r=row0+run/8u;sa[run]=select(vec4<f32>(0.0),probs[(r*H+h)*L4+t4+run%8u],r<ROWS);let key=kbase+t4*4u+run/8u;sb[run]=select(vec4<{{F}}>(0.0),qkv[key*row4+(2u*H+h)*D4+dt*8u+run%8u],mask[key]>0.5);}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sb[(k4*4u)*8u+lid.x]);let b1=vec4<f32>(sb[(k4*4u+1u)*8u+lid.x]);let b2=vec4<f32>(sb[(k4*4u+2u)*8u+lid.x]);let b3=vec4<f32>(sb[(k4*4u+3u)*8u+lid.x]);for(var i=0u;i<4u;i+=1u){let av=sa[(tileRow+i)*8u+k4];acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let col=h*D+dt*32u+lid.x*4u;for(var i=0u;i<4u;i+=1u){let row=row0+tileRow+i;if(row>=ROWS){continue;}for(var c=0u;c<4u;c+=1u){ctx[row*H*D+col+c]={{F}}(acc[i][c]);}}}", g = "{{ENABLE}}override H:u32=12u;override D:u32=64u;override ROWS:u32=128u;override NM:u32=256u;override MOFF:u32=0u;override PART:u32=0u;override L:u32=128u;@group(0)@binding(0)var<storage,read>qkv:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>table:array<vec4<{{F}}>>;@group(0)@binding(2)var<storage,read>kinfo:array<u32>;@group(0)@binding(3)var<storage,read>relidx:array<u32>;@group(0)@binding(4)var<storage,read_write>rel:array<f32>;var<workgroup>sa:array<vec4<{{F}}>,256>;var<workgroup>sw:array<vec4<{{F}}>,512>;var<workgroup>tileLive:u32;@compute@workgroup_size(16,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let nmt=(NM+63u)/64u;let h=wid.x/nmt;let m0=(wid.x%nmt)*64u;let tid=lid.y*16u+lid.x;let row0=wid.y*32u;let tileRow=lid.y*4u;let D4=D/4u;let hd4=H*D4;let row4=3u*hd4;if(tid==0u){let b=row0/L;let first=kinfo[2u*b];let last=kinfo[2u*b+1u];var live=0u;if(first<L){let lo=relidx[first*L+last]-MOFF;let hi=relidx[last*L+first]-MOFF;live=select(0u,1u,m0<=hi&&m0+63u>=lo);}tileLive=live;}if(workgroupUniformLoad(&tileLive)==0u){return;}var acc:array<vec4<f32>,4>;for(var t4=0u;t4<D4;t4+=8u){for(var run=tid;run<256u;run+=128u){let r=row0+run/8u;sa[run]=select(vec4<{{F}}>(0.0),qkv[r*row4+(PART*H+h)*D4+t4+run%8u],r<ROWS);}let ng=tid/8u;let kv=tid%8u;var m:array<vec4<{{F}}>,4>;for(var q=0u;q<4u;q+=1u){let mm=m0+ng*4u+q;m[q]=select(vec4<{{F}}>(0.0),table[(MOFF+mm)*hd4+h*D4+t4+kv],mm<NM);}for(var e=0u;e<4u;e+=1u){sw[(kv*4u+e)*16u+ng]=vec4<{{F}}>(m[0][e],m[1][e],m[2][e],m[3][e]);}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sw[(k4*4u)*16u+lid.x]);let b1=vec4<f32>(sw[(k4*4u+1u)*16u+lid.x]);let b2=vec4<f32>(sw[(k4*4u+2u)*16u+lid.x]);let b3=vec4<f32>(sw[(k4*4u+3u)*16u+lid.x]);for(var i=0u;i<4u;i+=1u){let av=vec4<f32>(sa[(tileRow+i)*8u+k4]);acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let mc=m0+lid.x*4u;for(var i=0u;i<4u;i+=1u){let row=row0+tileRow+i;if(row>=ROWS){continue;}for(var c=0u;c<4u;c+=1u){if(mc+c<NM){rel[(row*H+h)*NM+mc+c]=acc[i][c];}}}}", _ = "{{ENABLE}}override L:u32=128u;override H:u32=12u;override D:u32=64u;override SCALE:f32=0.125;override ROWS:u32=128u;@group(0)@binding(0)var<storage,read>qkv:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>kinfo:array<u32>;@group(0)@binding(2)var<storage,read_write>scores:array<f32>;var<workgroup>sa:array<vec4<{{F}}>,256>;var<workgroup>sw:array<vec4<{{F}}>,512>;var<workgroup>tileLive:u32;@compute@workgroup_size(16,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let nkt=(L+63u)/64u;let h=wid.x/nkt;let j0=(wid.x%nkt)*64u;let tid=lid.y*16u+lid.x;let row0=wid.y*32u;let kbase=(row0/L)*L;let tileRow=lid.y*4u;let D4=D/4u;let row4=3u*H*D4;if(tid==0u){let b=row0/L;let first=kinfo[2u*b];let last=kinfo[2u*b+1u];tileLive=select(0u,1u,first<L&&j0<=last&&j0+63u>=first);}if(workgroupUniformLoad(&tileLive)==0u){return;}var acc:array<vec4<f32>,4>;for(var t4=0u;t4<D4;t4+=8u){for(var run=tid;run<256u;run+=128u){let r=row0+run/8u;sa[run]=select(vec4<{{F}}>(0.0),qkv[r*row4+h*D4+t4+run%8u],r<ROWS);}let ng=tid/8u;let kv=tid%8u;var m:array<vec4<{{F}}>,4>;for(var q=0u;q<4u;q+=1u){let j=j0+ng*4u+q;m[q]=select(vec4<{{F}}>(0.0),qkv[(kbase+j)*row4+(H+h)*D4+t4+kv],j<L);}for(var e=0u;e<4u;e+=1u){sw[(kv*4u+e)*16u+ng]=vec4<{{F}}>(m[0][e],m[1][e],m[2][e],m[3][e]);}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sw[(k4*4u)*16u+lid.x]);let b1=vec4<f32>(sw[(k4*4u+1u)*16u+lid.x]);let b2=vec4<f32>(sw[(k4*4u+2u)*16u+lid.x]);let b3=vec4<f32>(sw[(k4*4u+3u)*16u+lid.x]);for(var i=0u;i<4u;i+=1u){let av=vec4<f32>(sa[(tileRow+i)*8u+k4]);acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let j=j0+lid.x*4u;for(var i=0u;i<4u;i+=1u){let row=row0+tileRow+i;if(row>=ROWS){continue;}for(var c=0u;c<4u;c+=1u){if(j+c<L){scores[(row*H+h)*L+j+c]=acc[i][c]*SCALE;}}}}", v = "override L:u32=128u;override H:u32=12u;override WINDOW:u32=0u;@group(0)@binding(0)var<storage,read>mask:array<f32>;@group(0)@binding(1)var<storage,read_write>scores:array<f32>;var<workgroup>red:array<f32,64>;fn keep(il:u32,j:u32,kbase:u32)->bool{var k=mask[kbase+j]>0.5;if(WINDOW>0u){let dist=select(il-j,j-il,j>il);k=k&&dist<=WINDOW;}return k;}@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let h=wid.x;let i=wid.y;let kbase=(i/L)*L;let il=i-kbase;let base=(i*H+h)*L;var m=-3.40282346638528859812e38f;for(var j=lid.x;j<L;j+=64u){if(keep(il,j,kbase)){m=max(m,scores[base+j]);}}red[lid.x]=m;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]=max(red[lid.x],red[lid.x+o]);}workgroupBarrier();}let mx=red[0];workgroupBarrier();var sum=0.0;for(var j=lid.x;j<L;j+=64u){var e=0.0;if(keep(il,j,kbase)){e=exp(scores[base+j]-mx);}scores[base+j]=e;sum+=e;}red[lid.x]=sum;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let inv=select(0.0,1.0/red[0],mask[i]>0.5&&red[0]>0.0);for(var j=lid.x;j<L;j+=64u){scores[base+j]*=inv;}}", y = "override L:u32=128u;override H:u32=12u;override NM:u32=256u;override MOFF:u32=0u;override INVSCALE:f32=0.0721688;@group(0)@binding(0)var<storage,read>mask:array<f32>;@group(0)@binding(1)var<storage,read>relidx:array<u32>;@group(0)@binding(2)var<storage,read>c2p:array<f32>;@group(0)@binding(3)var<storage,read>p2c:array<f32>;@group(0)@binding(4)var<storage,read_write>scores:array<f32>;var<workgroup>red:array<f32,64>;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let h=wid.x;let i=wid.y;let kbase=(i/L)*L;let il=i-kbase;let base=(i*H+h)*L;let cbase=(i*H+h)*NM;var m=-3.40282346638528859812e38f;for(var j=lid.x;j<L;j+=64u){if(mask[kbase+j]>0.5){let p=relidx[il*L+j]-MOFF;let s=scores[base+j]+(c2p[cbase+p]+p2c[((kbase+j)*H+h)*NM+p])*INVSCALE;scores[base+j]=s;m=max(m,s);}}red[lid.x]=m;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]=max(red[lid.x],red[lid.x+o]);}workgroupBarrier();}let mx=red[0];workgroupBarrier();var sum=0.0;for(var j=lid.x;j<L;j+=64u){var e=0.0;if(mask[kbase+j]>0.5){e=exp(scores[base+j]-mx);}scores[base+j]=e;sum+=e;}red[lid.x]=sum;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let inv=select(0.0,1.0/red[0],mask[i]>0.5&&red[0]>0.0);for(var j=lid.x;j<L;j+=64u){scores[base+j]*=inv;}}", b = "{{ENABLE}}override N:u32=384u;override L:u32=128u;override OFFSET:u32=0u;override MAXPOS:u32=0xffffffffu;override EPS:f32=1e-12;override MASKMUL:u32=0u;override POSIDS:u32=0u;@group(0)@binding(0)var<storage,read>word:array<{{F}}>;@group(0)@binding(1)var<storage,read>pos:array<{{F}}>;@group(0)@binding(2)var<storage,read>typ:array<{{F}}>;@group(0)@binding(3)var<storage,read>tt:array<u32>;@group(0)@binding(4)var<storage,read>weight:array<{{F}}>;@group(0)@binding(5)var<storage,read>bias:array<{{F}}>;@group(0)@binding(6)var<storage,read>mask:array<f32>;@group(0)@binding(7)var<storage,read_write>out:array<{{F}}>;var<workgroup>xs:array<f32,N>;var<workgroup>red:array<f32,64>;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let row=wid.x;let base=row*N;let t=tt[row];let prow=select((row%L)+OFFSET,t>>16u,POSIDS==1u);let pbase=min(prow,MAXPOS-1u)*N;let tbase=(t&0xffffu)*N;var s=0.0;for(var i=lid.x;i<N;i+=64u){let v=f32(word[base+i])+f32(pos[pbase+i])+f32(typ[tbase+i]);xs[i]=v;s+=v;}red[lid.x]=s;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let mean=red[0]/f32(N);workgroupBarrier();var sq=0.0;for(var i=lid.x;i<N;i+=64u){let d=xs[i]-mean;sq+=d*d;}red[lid.x]=sq;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let variance=red[0]/f32(N);let inv=1.0/sqrt(variance+EPS);for(var i=lid.x;i<N;i+=64u){var y=(xs[i]-mean)*inv*f32(weight[i])+f32(bias[i]);if(MASKMUL==1u){y*=mask[row];}out[base+i]={{F}}(y);}}", x = "{{ENABLE}}override K:u32=16u;override D:u32=384u;override L:u32=128u;@group(0)@binding(0)var<storage,read>markers:array<u32>;@group(0)@binding(1)var<storage,read>x:array<{{F}}>;@group(0)@binding(2)var<storage,read_write>states:array<{{F}}>;@compute@workgroup_size(64)fn main(@builtin(local_invocation_id)lid:vec3<u32>){let total=arrayLength(&markers)/3u;for(var i=lid.x;i<total*D;i+=64u){let g=i/D;let b=g/K;let d=i-g*D;states[i]=x[(b*L+markers[b*3u*K+(g-b*K)])*D+d];}}", S = "{{ENABLE}}override I:u32=1152u;@group(0)@binding(0)var<storage,read>mid:array<{{F}}>;@group(0)@binding(1)var<storage,read_write>gate:array<{{F}}>;fn gelu(v:f32)->f32{let u=v*0.7071067811865476;let t=1.0/(1.0+0.3275911*abs(u));let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;let e=1.0-p*exp(-u*u);return 0.5*v*(1.0+select(-e,e,v>=0.0));}@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let row=wid.x;let base=row*2u*I;for(var j=lid.x;j<I;j+=64u){gate[row*I+j]={{F}}(gelu(f32(mid[base+j]))*f32(mid[base+I+j]));}}", C = "{{ENABLE}}override N:u32=768u;override L:u32=128u;override KS:u32=3u;@group(0)@binding(0)var<storage,read>emb:array<{{F}}>;@group(0)@binding(1)var<storage,read>mask:array<f32>;@group(0)@binding(2)var<storage,read_write>out:array<{{F}}>;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let row=wid.x;let seq=row/L;let i=i32(row%L);let pad=i32((KS-1u)/2u);for(var t=0u;t<KS;t+=1u){let j=i+i32(t)-pad;let srcRow=seq*L+u32(max(j,0));let ok=j>=0&&j<i32(L)&&mask[srcRow]!=0.0;let src=srcRow*N;for(var c=lid.x;c<N;c+=64u){var v={{F}}(0.0);if(ok){v=emb[src+c];}out[row*KS*N+t*N+c]=v;}}}", w = "{{ENABLE}}override N:u32=384u;override MODE:u32=0u;override EPS:f32=1e-7;@group(0)@binding(0)var<storage,read>a:array<{{F}}>;@group(0)@binding(1)var<storage,read>b:array<{{F}}>;@group(0)@binding(2)var<storage,read>weight:array<{{F}}>;@group(0)@binding(3)var<storage,read>bias:array<{{F}}>;@group(0)@binding(4)var<storage,read>mask:array<f32>;@group(0)@binding(5)var<storage,read_write>out:array<{{F}}>;var<workgroup>xs:array<f32,N>;var<workgroup>red:array<f32,64>;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let row=wid.x;let base=row*N;var s=0.0;for(var i=lid.x;i<N;i+=64u){var v=f32(a[base+i]);if(MODE==2u){v+=f32(b[base+i]);}xs[i]=v;s+=v;}red[lid.x]=s;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let mean=red[0]/f32(N);workgroupBarrier();var sq=0.0;for(var i=lid.x;i<N;i+=64u){let d=xs[i]-mean;sq+=d*d;}red[lid.x]=sq;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let variance=red[0]/f32(N);let inv=1.0/sqrt(variance+EPS);for(var i=lid.x;i<N;i+=64u){var y=(xs[i]-mean)*inv*f32(weight[i])+f32(bias[i]);if(MODE==1u){y*=mask[row];}out[base+i]={{F}}(y);}}", T = "{{ENABLE}}override K:u32=16u;override TEMP:f32=1.0;@group(0)@binding(0)var<storage,read>raw:array<{{F}}>;@group(0)@binding(1)var<storage,read>packed:array<u32>;@group(0)@binding(2)var<storage,read_write>logits:array<f32>;@compute@workgroup_size(64)fn main(@builtin(local_invocation_id)lid:vec3<u32>){let total=arrayLength(&packed)/3u;for(var k=lid.x;k<total;k+=64u){let b=k/K;let m=bitcast<f32>(packed[b*3u*K+K+(k-b*K)]);logits[k]=select(-1e4,f32(raw[k])/TEMP,m>0.5);}}", E = "{{ENABLE}}override M:u32=1u;override N:u32=1u;override K:u32=1u;override ACT:u32=0u;@group(0)@binding(0)var<storage,read>a:array<{{F}}>;@group(0)@binding(1)var<storage,read>w:array<{{F}}>;@group(0)@binding(2)var<storage,read>bias:array<{{F}}>;@group(0)@binding(3)var<storage,read_write>c:array<{{F}}>;var<workgroup>ta:array<f32,256>;var<workgroup>tw:array<f32,256>;fn activate(v:f32)->f32{if(ACT==1u){return max(v,0.0);}if(ACT==2u){let u=v*0.7071067811865476;let t=1.0/(1.0+0.3275911*abs(u));let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;let e=1.0-p*exp(-u*u);return 0.5*v*(1.0+select(-e,e,v>=0.0));}if(ACT==3u){return tanh(v);}if(ACT==4u){return v/(1.0+exp(-v));}return v;}@compute@workgroup_size(16,16)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let row=wid.y*16u+lid.y;let col=wid.x*16u+lid.x;var acc=0.0;for(var t=0u;t<K;t+=16u){let ai=t+lid.x;let wi=t+lid.y;ta[lid.y*16u+lid.x]=select(0.0,f32(a[row*K+ai]),row<M&&ai<K);tw[lid.y*16u+lid.x]=select(0.0,f32(w[col*K+wi]),col<N&&wi<K);workgroupBarrier();for(var k=0u;k<16u;k+=1u){acc+=ta[lid.y*16u+k]*tw[k*16u+lid.x];}workgroupBarrier();}if(row<M&&col<N){c[row*N+col]={{F}}(activate(acc+f32(bias[col])));}}", D = "{{ENABLE}}override L:u32=128u;override H:u32=6u;override D:u32=64u;override SCALE:f32=0.125;override WINDOW:u32=0u;@group(0)@binding(0)var<storage,read>qkv:array<{{F}}>;@group(0)@binding(1)var<storage,read>mask:array<f32>;@group(0)@binding(2)var<storage,read_write>ctx:array<{{F}}>;var<workgroup>scores:array<f32,L>;var<workgroup>red:array<f32,64>;const LOWEST=-3.40282346638528859812e38f;fn keep(il:u32,j:u32,kbase:u32)->bool{var k=mask[kbase+j]>0.5;if(WINDOW>0u){let dist=select(il-j,j-il,j>il);k=k&&dist<=WINDOW;}return k;}@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let h=wid.x;let i=wid.y;let b=i/L;let il=i-b*L;let kbase=b*L;let hd=H*D;let qb=i*3u*hd+h*D;var qreg:array<f32,64>;for(var d=0u;d<D;d+=1u){qreg[d]=f32(qkv[qb+d]);}if(mask[i]<=0.5){for(var d=lid.x;d<D;d+=64u){ctx[i*hd+h*D+d]={{F}}(0.0);}return;}for(var j=lid.x;j<L;j+=64u){if(!keep(il,j,kbase)){scores[j]=0.0;continue;}var s=0.0;let kb=(kbase+j)*3u*hd+(H+h)*D;for(var d=0u;d+3u<D;d+=4u){s+=qreg[d]*f32(qkv[kb+d])+qreg[d+1u]*f32(qkv[kb+d+1u])+qreg[d+2u]*f32(qkv[kb+d+2u])+qreg[d+3u]*f32(qkv[kb+d+3u]);}for(var d=D-D%4u;d<D;d+=1u){s+=qreg[d]*f32(qkv[kb+d]);}scores[j]=s*SCALE;}workgroupBarrier();var m=LOWEST;for(var j=lid.x;j<L;j+=64u){if(keep(il,j,kbase)){m=max(m,scores[j]);}}red[lid.x]=m;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]=max(red[lid.x],red[lid.x+o]);}workgroupBarrier();}let mx=red[0];workgroupBarrier();var sum=0.0;for(var j=lid.x;j<L;j+=64u){var e=0.0;if(keep(il,j,kbase)){e=exp(scores[j]-mx);}scores[j]=e;sum+=e;}red[lid.x]=sum;workgroupBarrier();for(var o=32u;o>0u;o>>=1u){if(lid.x<o){red[lid.x]+=red[lid.x+o];}workgroupBarrier();}let invTotal=select(0.0,1.0/red[0],red[0]>0.0);workgroupBarrier();for(var d=lid.x;d<D;d+=64u){let vb=(2u*H+h)*D+d;var acc0=0.0;var acc1=0.0;var acc2=0.0;var acc3=0.0;for(var j=0u;j+3u<L;j+=4u){let s0=scores[j];let s1=scores[j+1u];let s2=scores[j+2u];let s3=scores[j+3u];if(s0!=0.0){acc0+=s0*f32(qkv[(kbase+j+0u)*3u*hd+vb]);}if(s1!=0.0){acc1+=s1*f32(qkv[(kbase+j+1u)*3u*hd+vb]);}if(s2!=0.0){acc2+=s2*f32(qkv[(kbase+j+2u)*3u*hd+vb]);}if(s3!=0.0){acc3+=s3*f32(qkv[(kbase+j+3u)*3u*hd+vb]);}}for(var j=L-L%4u;j<L;j+=1u){let s0=scores[j];if(s0!=0.0){acc0+=s0*f32(qkv[(kbase+j)*3u*hd+vb]);}}var acc=(acc0+acc1+acc2+acc3)*invTotal;ctx[i*hd+h*D+d]={{F}}(acc);}}", O = "{{ENABLE}}override L:u32=128u;override H:u32=6u;override D:u32=64u;override SCALE:f32=0.125;override WINDOW:u32=0u;override ROWS:u32=128u;@group(0)@binding(0)var<storage,read>qkv:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>mask:array<f32>;@group(0)@binding(2)var<storage,read_write>ctx:array<vec4<{{F}}>>;var<workgroup>sk:array<vec4<{{F}}>,512>;var<workgroup>sv:array<vec4<{{F}}>,512>;@compute@workgroup_size(32)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let h=wid.x;let i=wid.y*32u+lid.x;let b=(wid.y*32u)/L;let il=i-b*L;let kbase=b*L;let D4=D/4u;let hd4=H*D4;let row4=3u*hd4;let live=i<ROWS&&mask[min(i,ROWS-1u)]>0.5;var q:array<vec4<f32>,16>;var o:array<vec4<f32>,16>;if(live){for(var d=0u;d<D4;d+=1u){q[d]=vec4<f32>(qkv[i*row4+h*D4+d])*SCALE;}}var m=-3.40282346638528859812e38f;var l=0.0;for(var j0=0u;j0<L;j0+=32u){for(var e=lid.x;e<32u*D4;e+=32u){let kj=kbase+j0+e/D4;let d=e%D4;sk[e]=qkv[kj*row4+(H+h)*D4+d];sv[e]=qkv[kj*row4+(2u*H+h)*D4+d];}workgroupBarrier();if(live){for(var jj=0u;jj<32u;jj+=1u){let j=j0+jj;var skip=mask[kbase+j]<=0.5;if(WINDOW>0u){let dist=select(il-j,j-il,j>il);skip=skip||dist>WINDOW;}if(skip){continue;}var s=0.0;for(var d=0u;d<D4;d+=1u){s+=dot(q[d],vec4<f32>(sk[jj*D4+d]));}if(s>m){let c=exp(m-s);l*=c;for(var d=0u;d<D4;d+=1u){o[d]*=c;}m=s;}let p=exp(s-m);l+=p;for(var d=0u;d<D4;d+=1u){o[d]+=p*vec4<f32>(sv[jj*D4+d]);}}}workgroupBarrier();}if(i<ROWS){let inv=select(0.0,1.0/l,live);for(var d=0u;d<D4;d+=1u){ctx[i*hd4+h*D4+d]=vec4<{{F}}>(o[d]*inv);}}}", k = "{{ENABLE}}override M:u32=1u;override N:u32=1u;override K:u32=32u;override ACT:u32=0u;@group(0)@binding(0)var<storage,read>a:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>w:array<vec4<{{F}}>>;@group(0)@binding(2)var<storage,read>bias:array<{{F}}>;@group(0)@binding(3)var<storage,read_write>c:array<{{F}}>;var<workgroup>sa:array<vec4<{{F}}>,256>;var<workgroup>sw:array<vec4<{{F}}>,512>;fn activate(v:f32)->f32{if(ACT==1u){return max(v,0.0);}if(ACT==2u){let u=v*0.7071067811865476;let t=1.0/(1.0+0.3275911*abs(u));let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;let e=1.0-p*exp(-u*u);return 0.5*v*(1.0+select(-e,e,v>=0.0));}if(ACT==3u){return tanh(v);}if(ACT==4u){return v/(1.0+exp(-v));}return v;}@compute@workgroup_size(16,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let tid=lid.y*16u+lid.x;let row0=wid.y*32u;let col0=wid.x*64u;let tileRow=lid.y*4u;let K4=K/4u;var acc:array<vec4<f32>,4>;for(var t4=0u;t4<K4;t4+=8u){for(var run=tid;run<256u;run+=128u){let ar=row0+run/8u;sa[run]=select(vec4<{{F}}>(0.0),a[ar*K4+t4+run%8u],ar<M);}for(var g=tid;g<128u;g+=128u){let ng=g/8u;let kv=g%8u;var m:array<vec4<{{F}}>,4>;for(var q=0u;q<4u;q+=1u){let wr=col0+ng*4u+q;m[q]=select(vec4<{{F}}>(0.0),w[wr*K4+t4+kv],wr<N);}for(var e=0u;e<4u;e+=1u){sw[(kv*4u+e)*16u+ng]=vec4<{{F}}>(m[0][e],m[1][e],m[2][e],m[3][e]);}}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sw[(k4*4u)*16u+lid.x]);let b1=vec4<f32>(sw[(k4*4u+1u)*16u+lid.x]);let b2=vec4<f32>(sw[(k4*4u+2u)*16u+lid.x]);let b3=vec4<f32>(sw[(k4*4u+3u)*16u+lid.x]);for(var i=0u;i<4u;i+=1u){let av=vec4<f32>(sa[(tileRow+i)*8u+k4]);acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let col=col0+lid.x*4u;for(var i=0u;i<4u;i+=1u){let row=row0+tileRow+i;if(row>=M){continue;}for(var j=0u;j<4u;j+=1u){if(col+j<N){c[row*N+col+j]={{F}}(activate(acc[i][j]+f32(bias[col+j])));}}}}", A = "{{ENABLE}}override M:u32=1u;override N:u32=1u;override K:u32=32u;override ACT:u32=0u;@group(0)@binding(0)var<storage,read>a:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>w:array<vec4<{{F}}>>;@group(0)@binding(2)var<storage,read>bias:array<{{F}}>;@group(0)@binding(3)var<storage,read_write>c:array<{{F}}>;var<workgroup>sa:array<vec4<{{F}}>,128>;var<workgroup>sw:array<vec4<{{F}}>,256>;fn activate(v:f32)->f32{if(ACT==1u){return max(v,0.0);}if(ACT==2u){let u=v*0.7071067811865476;let t=1.0/(1.0+0.3275911*abs(u));let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;let e=1.0-p*exp(-u*u);return 0.5*v*(1.0+select(-e,e,v>=0.0));}if(ACT==3u){return tanh(v);}if(ACT==4u){return v/(1.0+exp(-v));}return v;}@compute@workgroup_size(8,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let tid=lid.y*8u+lid.x;let row0=wid.y*16u;let col0=wid.x*32u;let tileRow=lid.y*2u;let K4=K/4u;var acc:array<vec4<f32>,2>;for(var t4=0u;t4<K4;t4+=8u){for(var run=tid;run<128u;run+=64u){let ar=row0+run/8u;sa[run]=select(vec4<{{F}}>(0.0),a[ar*K4+t4+run%8u],ar<M);}for(var g=tid;g<64u;g+=64u){let ng=g/8u;let kv=g%8u;var m:array<vec4<{{F}}>,4>;for(var q=0u;q<4u;q+=1u){let wr=col0+ng*4u+q;m[q]=select(vec4<{{F}}>(0.0),w[wr*K4+t4+kv],wr<N);}for(var e=0u;e<4u;e+=1u){sw[(kv*4u+e)*8u+ng]=vec4<{{F}}>(m[0][e],m[1][e],m[2][e],m[3][e]);}}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sw[(k4*4u)*8u+lid.x]);let b1=vec4<f32>(sw[(k4*4u+1u)*8u+lid.x]);let b2=vec4<f32>(sw[(k4*4u+2u)*8u+lid.x]);let b3=vec4<f32>(sw[(k4*4u+3u)*8u+lid.x]);for(var i=0u;i<2u;i+=1u){let av=vec4<f32>(sa[(tileRow+i)*8u+k4]);acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let col=col0+lid.x*4u;for(var i=0u;i<2u;i+=1u){let row=row0+tileRow+i;if(row>=M){continue;}for(var j=0u;j<4u;j+=1u){if(col+j<N){c[row*N+col+j]={{F}}(activate(acc[i][j]+f32(bias[col+j])));}}}}", j = "{{ENABLE}}override M:u32=1u;override N:u32=1u;override K:u32=32u;override ACT:u32=0u;@group(0)@binding(0)var<storage,read>a:array<vec4<{{F}}>>;@group(0)@binding(1)var<storage,read>w:array<vec4<{{F}}>>;@group(0)@binding(2)var<storage,read>bias:array<{{F}}>;@group(0)@binding(3)var<storage,read_write>c:array<{{F}}>;var<workgroup>sa:array<vec4<{{F}}>,64>;var<workgroup>sw:array<vec4<{{F}}>,256>;fn activate(v:f32)->f32{if(ACT==1u){return max(v,0.0);}if(ACT==2u){let u=v*0.7071067811865476;let t=1.0/(1.0+0.3275911*abs(u));let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;let e=1.0-p*exp(-u*u);return 0.5*v*(1.0+select(-e,e,v>=0.0));}if(ACT==3u){return tanh(v);}if(ACT==4u){return v/(1.0+exp(-v));}return v;}@compute@workgroup_size(8,8)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let tid=lid.y*8u+lid.x;let row0=wid.y*8u;let col0=wid.x*32u;let tileRow=lid.y*1u;let K4=K/4u;var acc:array<vec4<f32>,1>;for(var t4=0u;t4<K4;t4+=8u){for(var run=tid;run<64u;run+=64u){let ar=row0+run/8u;sa[run]=select(vec4<{{F}}>(0.0),a[ar*K4+t4+run%8u],ar<M);}for(var g=tid;g<64u;g+=64u){let ng=g/8u;let kv=g%8u;var m:array<vec4<{{F}}>,4>;for(var q=0u;q<4u;q+=1u){let wr=col0+ng*4u+q;m[q]=select(vec4<{{F}}>(0.0),w[wr*K4+t4+kv],wr<N);}for(var e=0u;e<4u;e+=1u){sw[(kv*4u+e)*8u+ng]=vec4<{{F}}>(m[0][e],m[1][e],m[2][e],m[3][e]);}}workgroupBarrier();for(var k4=0u;k4<8u;k4+=1u){let b0=vec4<f32>(sw[(k4*4u)*8u+lid.x]);let b1=vec4<f32>(sw[(k4*4u+1u)*8u+lid.x]);let b2=vec4<f32>(sw[(k4*4u+2u)*8u+lid.x]);let b3=vec4<f32>(sw[(k4*4u+3u)*8u+lid.x]);for(var i=0u;i<1u;i+=1u){let av=vec4<f32>(sa[(tileRow+i)*8u+k4]);acc[i]=b0*av.x+acc[i];acc[i]=b1*av.y+acc[i];acc[i]=b2*av.z+acc[i];acc[i]=b3*av.w+acc[i];}}workgroupBarrier();}let col=col0+lid.x*4u;for(var i=0u;i<1u;i+=1u){let row=row0+tileRow+i;if(row>=M){continue;}for(var j=0u;j<4u;j+=1u){if(col+j<N){c[row*N+col+j]={{F}}(activate(acc[i][j]+f32(bias[col+j])));}}}}", M = "{{ENABLE}}override L:u32=128u;override N:u32=384u;override MODE:u32=0u;@group(0)@binding(0)var<storage,read>x:array<{{F}}>;@group(0)@binding(1)var<storage,read>mask:array<f32>;@group(0)@binding(2)var<storage,read_write>out:array<{{F}}>;@compute@workgroup_size(64)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let d=wid.x*64u+lid.x;let b=wid.y;if(d>=N){return;}let rbase=b*L;if(MODE==0u){var acc=0.0;var cnt=0.0;for(var i=0u;i<L;i+=1u){let m=mask[rbase+i];if(m!=0.0){acc+=m*f32(x[(rbase+i)*N+d]);cnt+=m;}}out[b*N+d]={{F}}(acc/max(cnt,1e-9));}else{var best=0.0;var found=false;for(var i=0u;i<L;i+=1u){if(mask[rbase+i]>0.5){let v=f32(x[(rbase+i)*N+d]);best=select(v,max(best,v),found);found=true;}}out[b*N+d]={{F}}(best);}}", N = "{{ENABLE}}override L:u32=128u;override H:u32=6u;override D:u32=64u;@group(0)@binding(0)var<storage,read_write>qkv:array<{{F}}>;@group(0)@binding(1)var<storage,read>cossin:array<f32>;@compute@workgroup_size(32)fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>,){let i=wid.x;let il=i-(i/L)*L;let h=wid.y/2u;let seg=wid.y%2u;let hd=H*D;let base=i*3u*hd+(seg*H+h)*D;let half=D/2u;let d=lid.x;if(d>=half){return;}let a=f32(qkv[base+d]);let b=f32(qkv[base+d+half]);let c=cossin[il*2u*D+d];let s=cossin[il*2u*D+D+d];qkv[base+d]={{F}}(a*c-b*s);qkv[base+d+half]={{F}}(b*c+a*s);}";
//#endregion
//#region src/kernels/index.ts
function P(e, t) {
	return e.replace("{{ENABLE}}", t ? "enable f16;\n" : "").replaceAll("{{F}}", t ? "f16" : "f32");
}
var F = {
	add: p,
	attention: m,
	attpv: h,
	attrel: g,
	attscore: _,
	attsoftmax: v,
	attsoftrel: y,
	embln: b,
	gather: x,
	geglu: S,
	im2col: C,
	layernorm: w,
	masklogits: T,
	matmul: E,
	mbattention: D,
	mbflash: O,
	mmtile: k,
	mmtile16: A,
	mmtile8: j,
	pool: M,
	rope: N
}, I = {
	rw: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	rwSrc: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	storage: () => GPUBufferUsage.STORAGE,
	logits: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
}, L = (e) => Object.keys(e).sort().map((t) => `${t}=${e[t]}`).join(","), R = class {
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
				usage: I[e.usage]()
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
			let n = `${t.kernel}|${L(t.constants)}`, r = e.get(n);
			return r || (r = this.device.createComputePipeline({
				layout: "auto",
				compute: {
					module: this.device.createShaderModule({ code: P(F[t.kernel], this.plan.f16) }),
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
		let e = this.plan.captureSlots * this.plan.captureSlotBytes, t = this.device.limits?.maxBufferSize ?? n.maxBufferSize;
		if (e > t) throw Error(`capture buffer needs ${e} B, maxBufferSize is ${t}`);
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
	dispatchOp(e, t, n) {
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
				this.dispatchOp(n, t, c), n.end();
			}
		} else if (!e && !n) {
			let e = s();
			for (let t of this.segments) for (let n of l(t)) this.dispatchOp(e, n, c);
			e.end();
		} else {
			let t = this.plan.captureSlotBytes;
			for (let n of this.segments) {
				let r = s();
				if (e) for (let e of n.captureOps) this.dispatchOp(r, e, c);
				for (let e of l(n)) this.dispatchOp(r, e, c);
				if (r.end(), e) for (let e of n.capture) a.copyBufferToBuffer(e.buffer, 0, this.captureBuffer(), e.slot * t, t);
			}
		}
		return a.copyBufferToBuffer(this.resolve(this.plan.output.buffer), 0, this.staging, 0, this.plan.output.bytes), n && (a.resolveQuerySet(n.querySet, 0, o * 2, n.resolve, 0), a.copyBufferToBuffer(n.resolve, 0, n.staging, 0, o * 16)), a.finish();
	}
	submit(e, t = {}) {
		this.device.queue.submit([this.encode(e, t)]);
	}
	run(e, t = !1) {
		this.submit(t, {
			seqLen: e,
			qtype: this.qtype
		});
	}
	get bytes() {
		return this.gpuBytes;
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
	qtype;
	upload(e) {
		this.qtype = e.qtype;
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
		return e.dtype === "storage" && this.plan.f16 ? f(new Uint16Array(t, 0, n)) : new Float32Array(t, 0, n);
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
}, z = class {
	kh;
	weights;
	kind = "webgpu";
	weightBytes;
	constructor(e, t) {
		this.kh = e, this.weights = t, this.weightBytes = t.gpuBytes;
	}
	check(e) {
		l(e, this.kh.device.limits, (e) => this.weights.tensors.get(e)?.size);
	}
	runner(e) {
		return new R(this.kh.device, e, this.weights.tensors);
	}
	fitBatch(e, t) {
		return u(e, this.kh.device.limits, t);
	}
	prepare(e) {
		return r(this.kh, e);
	}
	call(e, t) {
		return a(this.kh, e, t);
	}
	info() {
		return {
			adapter: this.kh.adapterInfo,
			limitsMode: this.kh.limitsMode,
			timestamps: this.kh.hasTimestamps
		};
	}
	dispose() {
		this.kh.device.destroy();
	}
};
async function B(r, i, a, o) {
	let { manifest: s, shardBytes: c, downloadBytes: l, fetchMs: u, bodyMs: d, sha256Ms: f } = await t(i, a), p = r.limits?.maxBufferSize ?? n.maxBufferSize;
	for (let e of s.tensors) {
		let t = Math.ceil(e.byteLength / 4) * 4;
		if (!e.keepOnCpu && t > p) throw Error(`tensor ${e.name} needs ${t} B, maxBufferSize is ${p}`);
	}
	let m = /* @__PURE__ */ new Map(), h = 0, g = performance.now();
	for (let e of s.tensors) {
		if (e.keepOnCpu) continue;
		let t = new Uint8Array(c[e.shard], e.offset, e.byteLength), n = Math.ceil(e.byteLength / 4) * 4, i = r.createBuffer({
			size: n,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		if (e.byteLength % 4 == 0) r.queue.writeBuffer(i, 0, t);
		else {
			let e = new Uint8Array(n);
			e.set(t), r.queue.writeBuffer(i, 0, e);
		}
		h += n, m.set(e.name, i);
	}
	let _ = performance.now() - g, v = i.slice(0, i.lastIndexOf("/") + 1), y = await e(v, s, c, o);
	return {
		manifest: s,
		tensors: m,
		words: y.table,
		downloadBytes: l + y.bytes,
		gpuBytes: h,
		timing: {
			fetchMs: u,
			bodyMs: d,
			sha256Ms: f,
			uploadMs: _,
			embedJoinMs: y.timing.embedJoinMs,
			wordsFetchMs: y.timing.wordsFetchMs,
			wordsSha256Ms: y.timing.wordsSha256Ms
		}
	};
}
async function V(e, t, n, r) {
	let i = await B(e.device, t, n, r);
	return {
		backend: new z(e, i),
		...i
	};
}
//#endregion
export { z as GpuBackend, R as PlanExecutor, V as loadGpu, B as loadWeights };
