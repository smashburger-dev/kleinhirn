// Plan IR (K28 design section 3): a list of operations over named buffers.
// Pure data. No import from kernels/, no GPU types, so a CPU executor can
// read the same list later.

export type KernelName =
  | 'add' | 'attention' | 'attpv' | 'attrel' | 'attscore' | 'attsoftmax' | 'attsoftrel' | 'embln' | 'gather' | 'geglu' | 'im2col' | 'layernorm'
  | 'masklogits' | 'matmul' | 'mbattention' | 'mbflash' | 'mmtile' | 'mmtile16' | 'mmtile8' | 'pool' | 'rope';

// Resolved per call from the row count: 'rows' = rows, 'rows16' = ceil(rows / 16),
// 'rows32' = ceil(rows / 32), 'rows8' = ceil(rows / 8).
export type Dim = number | 'rows' | 'rows8' | 'rows16' | 'rows32';

export interface BufferDecl {
  id: string;
  bytes: number;
  usage: 'rw' | 'rwSrc' | 'storage' | 'logits';
  init?: ArrayBufferView; // written once at build (relidx, cossin, zero)
}

export interface Op {
  name: string;          // dispatch-profile key suffix ('qkv', 'lnA', ...)
  kernel: KernelName;
  constants: Record<string, number>;
  bind: string[];        // buffer ids by binding index; weights as 'w:<tensor>'
  dispatch: [Dim, Dim];
  // K27: kernels with the same bindings and the same results (bit-equal), by ascending maxRows;
  // per call the first whose maxRows covers the dispatched rows runs (small row counts need
  // smaller tiles), else the op's own kernel.
  alts?: { kernel: KernelName; dispatch: [Dim, Dim]; maxRows: number }[];
}

export interface Segment {
  name: string;          // pass-profile key ('embed', 'layer3', 'final', ...)
  prefix: string;        // dispatch-profile prefix ('', 'L3.', 'head.', ...)
  ops: Op[];
  captureOps?: Op[];     // run before ops in capture mode only (GLiNER pre-mask LN)
  capture?: { buffer: string; slot: number }[]; // copies after the pass, capture mode
  skippable?: boolean;   // the debug skip set indexes into ops of these segments
}

export interface Plan {
  f16: boolean;
  length: number;
  batch: number;
  markers: number;
  embeddingSize: number; // width of one word row in the 'emb' input (upload payload check)
  buffers: BufferDecl[];
  segments: Segment[];
  // markers: packed marker indices (GLiNER, Julia); typeIds: token type ids
  // as u32 per row (absolute-position families).
  inputs: { embeddings: string; mask: string; markers?: string; typeIds?: string };
  rowSelect?: { table: string; dst: string; rowBytes: number }; // Julia type row
  // dtype 'f32': the buffer holds f32 (GLiNER, Julia logits from masklogits).
  // 'storage': the buffer is in the storage format of the plan (f16 or f32);
  // the executor copies it raw and JS converts. rows x cols are the valid
  // elements; bytes is rounded up to a multiple of 4.
  output: { buffer: string; bytes: number; dtype: 'f32' | 'storage'; rows: number; cols: number };
  captureSlots: number;
  captureSlotBytes: number;
}
