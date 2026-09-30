#!/usr/bin/env node
// Local static server for the device page: site/dist plus /models/small-upstream/
// (weights=local). Localhost only, correct content types, HTTP Range,
// no-store so every load is a cold load.
// Usage: node tools/serve_site.mjs [port] [--accept-results [--results-dir <dir>]]
//   port                 default 5201
//   --accept-results     enable POST /__result?rid=<id>: the body is written to
//                        <results-dir>/<id>.report.json (default bench/results
//                        in this checkout). Off by default; GitHub Pages has no
//                        such endpoint. GET /__result answers 200 when enabled.
// /diag/ serves bench/diag (standalone diagnosis pages).
import { createReadStream, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const flagValue = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
const PORT = Number(argv.find((a, i) => !a.startsWith('--') && argv[i - 1] !== '--results-dir') ?? 5201);
const ACCEPT_RESULTS = argv.includes('--accept-results');
const RESULTS_DIR = resolve(flagValue('--results-dir') ?? resolve(ROOT, 'bench/results'));
const MAX_BODY = 64 * 1024 * 1024;
const MOUNTS = [
  { prefix: '/diag/', dir: resolve(ROOT, 'bench/diag') },
  { prefix: '/models/small-upstream/', dir: resolve(ROOT, 'models/small-upstream') },
  { prefix: '/', dir: resolve(ROOT, 'site/dist') },
];
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
};

function locate(pathname) {
  for (const { prefix, dir } of MOUNTS) {
    if (!pathname.startsWith(prefix)) continue;
    const rel = normalize(decodeURIComponent(pathname.slice(prefix.length) || 'index.html'));
    const file = resolve(dir, rel);
    if (file !== dir && !file.startsWith(dir + sep)) return null;
    return file;
  }
  return null;
}

function acceptResult(req, res, url) {
  if (!ACCEPT_RESULTS) { res.writeHead(404).end('not found'); return; }
  if (req.method === 'GET') { res.writeHead(200).end('accepting results'); return; }
  const rid = url.searchParams.get('rid') ?? '';
  const host = String(req.headers.host ?? '');
  if (req.method !== 'POST' || !/^[\w.-]{1,120}$/.test(rid) || !/^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host)) {
    res.writeHead(400).end('bad request');
    return;
  }
  const chunks = [];
  let size = 0;
  req.on('data', (c) => {
    size += c.length;
    if (size > MAX_BODY) { res.writeHead(413).end('too large'); req.destroy(); return; }
    chunks.push(c);
  });
  req.on('end', () => {
    if (size > MAX_BODY) return;
    mkdirSync(RESULTS_DIR, { recursive: true });
    const file = resolve(RESULTS_DIR, `${rid}.report.json`);
    writeFileSync(file, Buffer.concat(chunks));
    console.log(`result ${rid}: ${size} bytes -> ${file}`);
    res.writeHead(204).end();
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (url.pathname === '/__result') { acceptResult(req, res, url); return; }
  const file = locate(url.pathname.endsWith('/') && url.pathname === '/' ? '/index.html' : url.pathname);
  let st;
  try { st = file && statSync(file); } catch { st = null; }
  if (!file || !st?.isFile()) { res.writeHead(404).end('not found'); return; }
  const headers = {
    'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
    'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store',
  };
  let start = 0;
  let end = st.size - 1;
  let status = 200;
  const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
  if (m && (m[1] || m[2])) {
    if (m[1]) { start = Number(m[1]); if (m[2]) end = Math.min(end, Number(m[2])); }
    else { start = Math.max(0, st.size - Number(m[2])); }
    if (start > end) { res.writeHead(416, { 'Content-Range': `bytes */${st.size}` }).end(); return; }
    status = 206;
    headers['Content-Range'] = `bytes ${start}-${end}/${st.size}`;
  }
  headers['Content-Length'] = end - start + 1;
  res.writeHead(status, headers);
  if (req.method === 'HEAD') { res.end(); return; }
  createReadStream(file, { start, end }).pipe(res);
});
server.listen(PORT, '127.0.0.1', () => console.log(`serving site/dist on http://127.0.0.1:${PORT}/`));
