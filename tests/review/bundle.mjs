// Offline build analysis, no server. Uses installed Vite/Rolldown and source maps.
// node tests/review/bundle.mjs tests/review/bundle-results.json
import { build } from 'vite';
import { gzipSync } from 'node:zlib';
import { writeFileSync, readFileSync } from 'node:fs';
import { relative } from 'node:path';

const zip = (s) => gzipSync(s, { level: 9 }).byteLength;
const classify = (p) => p.includes('/kernels/') ? 'WGSL + kernel registry'
  : p.includes('/tokenizer/hf/') ? 'HF tokenizer'
    : p.includes('/tokenizer/') ? 'GLiNER/Julia tokenizer + input'
      : p.includes('/plan/') ? 'plan builder + executor + spec'
        : p.includes('/wasm') ? 'WASM client + worker URL'
          : p.includes('unmapped') ? 'unmapped'
            : 'API + device + weights + tasks + cache + half';
const base64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function vlq(segment) {
  const out = [];
  let n = 0, shift = 0;
  for (const c of segment) {
    const b = base64.indexOf(c);
    n += (b & 31) * 2 ** shift;
    if (b & 32) { shift += 5; continue; }
    out.push(n & 1 ? -Math.floor(n / 2) : Math.floor(n / 2));
    n = 0; shift = 0;
  }
  return out;
}
function mappedPieces(chunk) {
  const lines = chunk.code.split('\n');
  const mappings = chunk.map.mappings.split(';');
  const pieces = [];
  let src = 0;
  lines.forEach((code, i) => {
    const line = mappings[i] ?? '';
    let col = 0;
    const points = [];
    for (const segment of line.split(',').filter(Boolean)) {
      const v = vlq(segment);
      col += v[0];
      if (v.length > 1) src += v[1];
      points.push({ col, path: v.length > 1 ? chunk.map.sources[src] : 'unmapped' });
    }
    points.unshift({ col: 0, path: 'unmapped' });
    points.forEach((p, j) => {
      const end = points[j + 1]?.col ?? code.length;
      if (end > p.col) pieces.push({ path: p.path, code: code.slice(p.col, end) });
    });
    if (i < lines.length - 1) pieces.push({ path: 'unmapped', code: '\n' });
  });
  return pieces;
}

function attributeRawLiterals(pieces, chunk) {
  const ranges = Object.keys(chunk.modules).filter((id) => id.endsWith('.wgsl?raw')).map((id) => {
    const literal = JSON.stringify(readFileSync(id.slice(0, -4), 'utf8'));
    const start = chunk.code.indexOf(literal);
    if (start < 0) throw new Error(`raw literal not found: ${id}`);
    return { start, end: start + literal.length, path: relative(process.cwd(), id) };
  });
  let pos = 0;
  return pieces.flatMap((p) => {
    const start = pos, end = pos + p.code.length;
    pos = end;
    const overlaps = ranges.filter((r) => r.start < end && r.end > start);
    const cuts = [start, ...overlaps.flatMap((r) => [Math.max(start, r.start), Math.min(end, r.end)]), end];
    const unique = [...new Set(cuts)].sort((a, b) => a - b);
    return unique.slice(0, -1).map((a, i) => ({
      path: overlaps.find((r) => a >= r.start && a < r.end)?.path ?? p.path,
      code: p.code.slice(a - start, unique[i + 1] - start),
    }));
  });
}

let report;
await build({ logLevel: 'silent', define: { __KH_BUILD_ID__: JSON.stringify('review0000') },
  build: { write: false, sourcemap: 'hidden' }, plugins: [{
  name: 'review-rendered-modules',
  generateBundle(_options, bundle) {
    const chunk = Object.values(bundle).find((c) => c.type === 'chunk' && c.fileName === 'kleinhirn.js');
    if (!chunk) return;
    const rows = Object.entries(chunk.modules).map(([id, m]) => ({
      path: relative(process.cwd(), id), renderedLength: m.renderedLength,
      gzipPreAssembly: zip(m.code ?? ''), renderedExports: m.renderedExports,
    }));
    const pieces = attributeRawLiterals(mappedPieces(chunk), chunk);
    if (pieces.map((p) => p.code).join('') !== chunk.code) throw new Error('source-map reconstruction differs');
    const groups = [...new Set(pieces.map((p) => classify(p.path)))].map((group) => ({
      group,
      mappedBytes: pieces.filter((p) => classify(p.path) === group)
        .reduce((n, p) => n + Buffer.byteLength(p.code), 0),
      gzipMarginal: zip(chunk.code) - zip(pieces.filter((p) => classify(p.path) !== group).map((p) => p.code).join('')),
    }));
    const modules = [...new Set(pieces.map((p) => p.path))].map((path) => ({
      path, mappedBytes: pieces.filter((p) => p.path === path).reduce((n, p) => n + Buffer.byteLength(p.code), 0),
      gzipMarginal: zip(chunk.code) - zip(pieces.filter((p) => p.path !== path).map((p) => p.code).join('')),
    })).sort((a, b) => b.mappedBytes - a.mappedBytes);
    const workerAssets = Object.values(bundle).filter((b) => b.type === 'asset' && b.fileName.endsWith('.js'))
      .map((b) => ({ path: b.fileName, bytes: Buffer.byteLength(b.source), gzip9: zip(b.source) }));
    report = { node: process.version, raw: Buffer.byteLength(chunk.code), gzip9: zip(chunk.code), groups, modules,
      attributedBytes: groups.reduce((n, g) => n + g.mappedBytes, 0), workerAssets, preAssembly: rows,
      warning: 'Mapped bytes include source-map separator ownership; WGSL JSON literals assigned by exact match. Marginal gzip saving is recompression after removing mapped spans, NOT a working smaller bundle and NOT additive. preAssembly is before final chunk minification.' };
  },
}] });

// In-memory experiments only. No src/ file changes. WGSL contains line comments, no strings.
async function variant(name, load) {
  let sizes;
  await build({ logLevel: 'silent', define: { __KH_BUILD_ID__: JSON.stringify('review0000') }, build: { write: false },
    plugins: [{ name, enforce: 'pre', load }, { name: `${name}-sizes`, generateBundle(_o, bundle) {
      const c = bundle['kleinhirn.js'];
      sizes = { raw: Buffer.byteLength(c.code), gzip9: zip(c.code) };
    } }] });
  return sizes;
}
report.noWgslComments = await variant('review-strip-wgsl-comments', (id) => {
  if (!id.endsWith('.wgsl?raw')) return;
  const code = readFileSync(id.slice(0, -4), 'utf8').replace(/\/\/[^\n]*/g, '').replace(/\n\s*\n/g, '\n');
  return `export default ${JSON.stringify(code)};`;
});
report.noDeadUnigramScores = await variant('review-drop-dead-map', (id) => {
  if (!id.endsWith('/src/tokenizer/hf/unigram.ts')) return;
  return readFileSync(id, 'utf8').replace('  private scores = new Map<string, number>();\n', '')
    .replace('      this.scores.set(token, score);\n', '');
});
const actual = readFileSync('dist/kleinhirn.js');
report.dist = { bytes: actual.byteLength, gzip9: zip(actual) };
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
