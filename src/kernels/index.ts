// WGSL kernel sources. Storage dtype is templated: {{F}} becomes f16 or f32,
// {{ENABLE}} becomes the `enable f16;` directive when needed. All compute
// accumulates in f32 either way.

import addSrc from './add.wgsl?raw';
import attentionSrc from './attention.wgsl?raw';
import gatherSrc from './gather.wgsl?raw';
import gegluSrc from './geglu.wgsl?raw';
import layernormSrc from './layernorm.wgsl?raw';
import masklogitsSrc from './masklogits.wgsl?raw';
import matmulSrc from './matmul.wgsl?raw';
import mbattentionSrc from './mbattention.wgsl?raw';
import ropeSrc from './rope.wgsl?raw';

export function wgsl(source: string, f16: boolean): string {
  return source
    .replace('{{ENABLE}}', f16 ? 'enable f16;\n' : '')
    .replaceAll('{{F}}', f16 ? 'f16' : 'f32');
}

export const KERNELS = {
  add: addSrc,
  attention: attentionSrc,
  gather: gatherSrc,
  geglu: gegluSrc,
  layernorm: layernormSrc,
  masklogits: masklogitsSrc,
  matmul: matmulSrc,
  mbattention: mbattentionSrc,
  rope: ropeSrc,
};
