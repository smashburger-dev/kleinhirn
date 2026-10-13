// R6 stage 1: the compact WGSL keeps the token sequence of every kernel source and both
// placeholders; nothing of a comment survives.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { compactWgsl, stripComments, tokens } from '../tools/wgsl_compact.ts';

const DIR = new URL('../src/kernels/', import.meta.url);
const files = readdirSync(DIR).filter((f) => f.endsWith('.wgsl'));

test('every kernel keeps its tokens and placeholders', () => {
  assert.ok(files.length >= 21);
  for (const f of files) {
    const src = readFileSync(new URL(f, DIR), 'utf8');
    const out = compactWgsl(src);
    assert.deepEqual(tokens(out), tokens(stripComments(src)), f);
    for (const ph of ['{{F}}', '{{ENABLE}}']) assert.equal(out.split(ph).length, src.split(ph).length, `${f} ${ph}`);
    assert.ok(!out.includes('//') && !out.includes('/*') && !/\s\s/.test(out), f);
  }
});

test('tokens that would merge keep a space', () => {
  assert.equal(compactWgsl('let a = b - -c; let d = e > > f;'), 'let a=b- -c;let d=e> >f;');
  assert.equal(compactWgsl('x /* a /* nested */ b */ y // tail\nz'), 'x y z');
  assert.equal(compactWgsl('a = 1.0 + .5e-3f; b = 0x1p4;'), 'a=1.0+.5e-3f;b=0x1p4;');
});
