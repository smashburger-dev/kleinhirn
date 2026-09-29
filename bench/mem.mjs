// OS-level memory metrics for the Playwright Chromium process tree
// (browser PID plus all children: renderer and GPU processes). Candidate
// metrics per docs/ARCHITECTURE.md: footprint, top -stats mem, ps rss.

import { execFileSync, spawnSync } from 'node:child_process';

export function ownTreePids(rootPid = process.pid) {
  // Every descendant of the runner process (Playwright driver, browser,
  // renderer, GPU, utility). The earlier approach matched the ms-playwright
  // binary path globally, which also collected OTHER agents' Playwright
  // browsers on the same machine and inflated both baseline and peak.
  const pids = [];
  const queue = [rootPid];
  const seen = new Set(queue);
  while (queue.length) {
    const pid = queue.shift();
    pids.push(pid);
    let out = '';
    try {
      out = execFileSync('pgrep', ['-P', String(pid)], { encoding: 'utf8' });
    } catch { /* no children */ }
    for (const line of out.split('\n')) {
      const child = Number(line.trim());
      if (child && !seen.has(child)) {
        seen.add(child);
        queue.push(child);
      }
    }
  }
  return pids;
}

function toBytes(value, unit) {
  const n = Number(value);
  if (unit === 'K' || unit === 'KB') return n * 1024;
  if (unit === 'M' || unit === 'MB') return n * 1024 * 1024;
  if (unit === 'G' || unit === 'GB') return n * 1024 * 1024 * 1024;
  return n;
}

export function rssBytes(pids) {
  if (!pids.length) return 0;
  const out = execFileSync('ps', ['-o', 'rss=', '-p', pids.join(',')], { encoding: 'utf8' });
  return out.trim().split('\n').reduce((sum, l) => sum + Number(l.trim()) * 1024, 0);
}

export function topBytes(pids) {
  const out = execFileSync('top', ['-l', '1', '-stats', 'pid,mem'], { encoding: 'utf8' });
  const wanted = new Set(pids);
  let total = 0;
  for (const line of out.split('\n')) {
    const m = line.trim().match(/^(\d+)\s+(\d+(?:\.\d+)?)([KMG])\+?$/);
    if (m && wanted.has(Number(m[1]))) total += toBytes(m[2], m[3]);
  }
  return total;
}

export function footprintBytes(pids) {
  // One footprint call for all pids (it accepts repeated -p and prints a
  // "Summary Footprint" line). Only the header field "Footprint:" counts;
  // scanning any KB/MB/GB token would pick up larger Clean columns.
  if (!pids.length) return 0;
  const args = [];
  for (const pid of pids) args.push('-p', String(pid));
  const res = spawnSync('footprint', args, { encoding: 'utf8', timeout: 20000 });
  const out = res.stdout ?? '';
  const summary = out.match(/^Summary Footprint:\s*([\d.]+)\s*(KB|MB|GB|B)\b/m);
  if (summary) return toBytes(summary[1], summary[2]);
  let total = 0;
  const perProcess = /^.*\[\d+\]:\s*64-bit\s+Footprint:\s*([\d.]+)\s*(KB|MB|GB|B)\b/gm;
  for (const m of out.matchAll(perProcess)) total += toBytes(m[1], m[2]);
  return total;
}

export const METRICS = {
  footprint: footprintBytes,
  top_mem: topBytes,
  ps_rss: rssBytes,
};

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Samples every metric in `names` every `intervalMs` for `ticks` rounds.
// Returns [{t, pids, <name>: bytes}]. Continues at once if a tick overruns.
export async function sampleSeries(names, ticks, intervalMs, between) {
  const rows = [];
  for (let i = 0; i < ticks; i += 1) {
    const pids = ownTreePids();
    const row = { t: Date.now(), pids: pids.length };
    for (const name of names) {
      try {
        row[name] = METRICS[name](pids);
      } catch {
        row[name] = null;
      }
    }
    rows.push(row);
    if (between) await between(row);
    const elapsed = Date.now() - row.t;
    if (elapsed < intervalMs) {
      await new Promise((r) => setTimeout(r, intervalMs - elapsed));
    }
  }
  return rows;
}
