// R6 stage 1 (workorder Festlegung 5): WGSL without comments and without whitespace the parser
// does not need. The sources in src/kernels stay commented; vite.config.ts applies this to every
// `*.wgsl?raw` import in build and dev. The output keeps the exact token sequence of the input
// (maximal munch over WGSL tokens plus the {{F}} and {{ENABLE}} placeholders); compactWgsl throws
// if re-tokenizing its own output gives a different sequence.

import type { Plugin } from 'vite';

const TOKEN = new RegExp([
  String.raw`\{\{[A-Z]+\}\}`,
  String.raw`[A-Za-z_][A-Za-z0-9_]*`,
  String.raw`0[xX][0-9a-fA-F]*(?:\.[0-9a-fA-F]*)?(?:[pP][+-]?[0-9]+)?[fhiu]?`,
  String.raw`(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[fhiu]?`,
  String.raw`>>=|<<=|->|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|\+=|-=|\*=|\/=|%=|&=|\|=|\^=`,
  String.raw`[-+*/%&|^~!<>=@(){}\[\],;:.]`,
].join('|'), 'y');

/** Removes line comments and (nested) block comments; keeps line breaks of line comments. */
export function stripComments(src: string): string {
  let out = '';
  for (let i = 0; i < src.length;) {
    if (src.startsWith('//', i)) {
      while (i < src.length && src[i] !== '\n') i += 1;
    } else if (src.startsWith('/*', i)) {
      let depth = 0;
      do {
        if (src.startsWith('/*', i)) { depth += 1; i += 2; }
        else if (src.startsWith('*/', i)) { depth -= 1; i += 2; }
        else if (i >= src.length) throw new Error('unterminated block comment');
        else i += 1;
      } while (depth > 0);
      out += ' ';
    } else {
      out += src[i];
      i += 1;
    }
  }
  return out;
}

/** WGSL tokens of a comment-free source by maximal munch; throws on a character it does not know. */
export function tokens(src: string): string[] {
  const list: string[] = [];
  let i = 0;
  while (i < src.length) {
    if (/\s/.test(src[i])) { i += 1; continue; }
    TOKEN.lastIndex = i;
    const m = TOKEN.exec(src);
    if (!m || m[0].length === 0) throw new Error(`wgsl_compact: unknown character ${JSON.stringify(src[i])} at ${i}`);
    list.push(m[0]);
    i += m[0].length;
  }
  return list;
}

const same = (a: string[], b: string[]): boolean => a.length === b.length && a.every((t, i) => t === b[i]);

/** The shortest join of the tokens: a space only where two tokens would otherwise merge. */
export function compactWgsl(src: string): string {
  const toks = tokens(stripComments(src));
  let out = '';
  let prev = '';
  for (const t of toks) {
    if (prev && !same(tokens(prev + t), [prev, t])) out += ' ';
    out += t;
    prev = t;
  }
  if (!same(tokens(out), toks)) throw new Error('wgsl_compact: token sequence changed');
  return out;
}

/** Vite plugin: `*.wgsl?raw` imports load the compact source. */
export function wgslCompact(): Plugin {
  return {
    name: 'kh-wgsl-compact',
    enforce: 'pre',
    async load(id: string) {
      if (!/\.wgsl\?raw$/.test(id)) return null;
      const { readFile } = await import('node:fs/promises');
      const src = await readFile(id.slice(0, -'?raw'.length), 'utf8');
      return `export default ${JSON.stringify(compactWgsl(src))};`;
    },
  };
}
