// WGSL kernel sources. Storage dtype is templated: {{F}} becomes f16 or f32,
// {{ENABLE}} becomes the `enable f16;` directive when needed. All compute
// accumulates in f32 either way.

import addSrc from './add.wgsl?raw';
import attentionSrc from './attention.wgsl?raw';
import attpvSrc from './attpv.wgsl?raw';
import attrelSrc from './attrel.wgsl?raw';
import attscoreSrc from './attscore.wgsl?raw';
import attsoftmaxSrc from './attsoftmax.wgsl?raw';
import attsoftrelSrc from './attsoftrel.wgsl?raw';
import emblnSrc from './embln.wgsl?raw';
import gatherSrc from './gather.wgsl?raw';
import gegluSrc from './geglu.wgsl?raw';
import im2colSrc from './im2col.wgsl?raw';
import layernormSrc from './layernorm.wgsl?raw';
import masklogitsSrc from './masklogits.wgsl?raw';
import matmulSrc from './matmul.wgsl?raw';
import mbattentionSrc from './mbattention.wgsl?raw';
import mbflashSrc from './mbflash.wgsl?raw';
import mmtileSrc from './mmtile.wgsl?raw';
import mmtile16Src from './mmtile16.wgsl?raw';
import mmtile8Src from './mmtile8.wgsl?raw';
import poolSrc from './pool.wgsl?raw';
import ropeSrc from './rope.wgsl?raw';

export function wgsl(source: string, f16: boolean): string {
  return source
    .replace('{{ENABLE}}', f16 ? 'enable f16;\n' : '')
    .replaceAll('{{F}}', f16 ? 'f16' : 'f32');
}

export const KERNELS = {
  add: addSrc,
  attention: attentionSrc,
  attpv: attpvSrc,
  attrel: attrelSrc,
  attscore: attscoreSrc,
  attsoftmax: attsoftmaxSrc,
  attsoftrel: attsoftrelSrc,
  embln: emblnSrc,
  gather: gatherSrc,
  geglu: gegluSrc,
  im2col: im2colSrc,
  layernorm: layernormSrc,
  masklogits: masklogitsSrc,
  matmul: matmulSrc,
  mbattention: mbattentionSrc,
  mbflash: mbflashSrc,
  mmtile: mmtileSrc,
  mmtile16: mmtile16Src,
  mmtile8: mmtile8Src,
  pool: poolSrc,
  rope: ropeSrc,
};
