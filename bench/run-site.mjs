// Runner for the device page (site/): opens ?autorun=1 in a visible browser,
// waits for body[data-state], saves window.__khResult and appends a runs.tsv
// line (pending first, finished afterwards).
//
// Usage: node bench/run-site.mjs --browser <name> [--stages f16,f32,wasm]
//        [--weights default|local|<url>] [--change k5-site] [--device "<name>"]
//        [--screenshots <dir>] [--timeout-min 45]
//        [--max-load <n> [--wait-min 20] [--no-wait] [--run-anyway]] [--reps <n>]
//        [--extra "trace=1&limit=20&warmup=5"] [--poll-s <n>] [--out-root <dir>]
//        [--safari-mode webdriver|local] [--diag readback] [--n <iterations>]
//        [--browser ios-safari [--safari-profiling] [--inspector-shots <dir>] [--tag <name>]]
//
// --extra       raw URL params appended to the page URL (trace, limit, warmup).
// --poll-s      seconds between state polls (default 1; safari webdriver 3).
// --out-root    directory that receives data/hillclimb/runs.tsv and
//               bench/results/ (default: the current directory). Official runs
//               start this script from the official worktree (site/dist, server,
//               git commit and dirty flag come from there) and write into the
//               main checkout, so the worktree stays clean.
// --safari-mode local: no WebDriver. The runner starts the server with
//               --accept-results, opens the page with `open -a Safari` and
//               report=local, waits for the POSTed file and closes the tab with
//               AppleScript.
// --diag readback: run /diag/readback.html (bench/diag) instead of the page.
//
// ios-safari: Safari on the plugged-in iPhone over safaridriver (no simulator),
// page = the public Pages URL, default device "iPhone 16 Pro 256GB", os
// "iOS 27.0 (24A437)", no local server, poll every 15 s, timeout 40 min. Each poll
// logs visibilityState and the progress text; a poll with visibilityState other
// than 'visible' marks the run backgrounded:true. Memory is not measurable from
// the Mac; the Mac load average is recorded but there is no max-load gate.
// --safari-profiling sets safari:automaticProfiling (Web Inspector timeline on
// the Mac). --inspector-shots <dir>: after the page reports done, wait 5 s,
// bring Safari to the front, screenshot the screen (ios-inspector-1.png), end
// the session, screenshot again (ios-inspector-2.png); Safari window names are
// logged at every poll. Files: k5-iphone16pro-<tag>-<stamp>.json / .result.json.
//
// Browsers: chromium, chromium-nogpu, webkit, firefox, brave, safari. All run
// visible (headless: false). One fresh browser per rep.
//
// Load policy: without --max-load the run is provisional. With --max-load n
// the runner refuses to start while the 1-min load is above n: it polls every
// 60 s for up to --wait-min minutes (default 20) and then fails, unless
// --no-wait (start at once) or --run-anyway (start after the wait) is given.
// A run that started above n, or from a dirty tree, is marked provisional.
//
// Memory: Playwright browsers use the process-tree footprint sampler from
// bench/mem.mjs (peak minus median of the samples taken before the page
// loads). WebKit runs its content and GPU processes as XPC services outside
// the runner's process tree, so for Playwright WebKit the sampler adds, and
// for Safari it uses only, the footprint of com.apple.WebKit.WebContent and
// com.apple.WebKit.GPU processes (Playwright WebKit: those that appeared after
// the browser started; Safari: all on the machine, because it reuses its
// process pool between sessions). Best effort in both cases.

import { chromium, firefox, webkit } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { loadavg } from 'node:os';
import { resolve } from 'node:path';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, statSync, unlinkSync, writeFileSync,
} from 'node:fs';
import { footprintBytes, median, ownTreePids } from './mem.mjs';

const PORT = 5201;
const BRAVE = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const MB = 1048576;
const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];

const BOOLEAN_FLAGS = new Set(['no-wait', 'run-anyway', 'no-activate', 'safari-profiling']);
const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const key = process.argv[i].replace(/^--/, '');
  if (BOOLEAN_FLAGS.has(key)) args.set(key, true);
  else { args.set(key, process.argv[i + 1]); i += 1; }
}
const browserName = args.get('browser') ?? 'chromium';
const stages = args.get('stages') ?? 'f16,f32,wasm';
const weights = args.get('weights') ?? 'default';
const change = args.get('change') ?? (browserName === 'ios-safari' ? 'k5-iphone' : 'k5-site');
const isIos = browserName === 'ios-safari';
const IOS_URL = 'https://smashburger-dev.github.io/kleinhirn';
const deviceName = args.get('device') ?? (isIos ? 'iPhone 16 Pro 256GB' : 'MacBook Pro M1 Pro 16 GB (runner)');
const osName = args.get('os') ?? (isIos ? 'iOS 27.0 (24A437)' : null);
const tag = args.get('tag') ?? stages.replaceAll(',', '-');
const inspectorShots = args.get('inspector-shots') ?? null;
const screenshotDir = args.get('screenshots');
const timeoutMs = Number(args.get('timeout-min') ?? (isIos ? 40 : 45)) * 60000;
const maxLoad = args.has('max-load') ? Number(args.get('max-load')) : null;
const waitMin = Number(args.get('wait-min') ?? 20);
const noWait = args.get('no-wait') === true;
const runAnyway = args.get('run-anyway') === true;
const reps = Number(args.get('reps') ?? 1);
// --rep-index n: label of this single-rep invocation (file suffix -r<n>, run.rep);
// the orchestrator uses it to interleave browsers.
const repIndex = args.has('rep-index') ? Number(args.get('rep-index')) : null;
const extra = args.get('extra') ?? '';
const OUT = resolve(args.get('out-root') ?? '.');
const RUNS = resolve(OUT, 'data/hillclimb/runs.tsv');
const RESULTS = resolve(OUT, 'bench/results');
const safariMode = args.get('safari-mode') ?? 'webdriver';
const diag = args.get('diag') ?? null;
const pollMs = Number(args.get('poll-s') ?? (isIos ? 15 : browserName === 'safari' && safariMode === 'webdriver' ? 3 : 1)) * 1000;
const localReport = browserName === 'safari' && safariMode === 'local';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const gitCommit = () => execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const gitDirty = () => execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim().length > 0;
const load1 = () => loadavg()[0];
const loadStr = () => loadavg().map((v) => v.toFixed(2)).join(' ');
const queryOf = (rid) => (diag ? `n=${args.get('n') ?? 200}` : `autorun=1&stages=${stages}${weights === 'default' ? '' : `&weights=${encodeURIComponent(weights)}`}`
  + `&device=${encodeURIComponent(deviceName)}` + (osName ? `&os=${encodeURIComponent(osName)}` : ''))
  + (extra ? `&${extra}` : '') + (localReport ? `&report=local&rid=${encodeURIComponent(rid)}` : '');
const pagePath = () => (diag ? '/diag/readback.html' : isIos ? '/' : '/index.html');
const resultGlobal = () => (diag ? '__diagResult' : '__khResult');

// ---- memory samplers -------------------------------------------------------

function psRows() {
  const out = execFileSync('ps', ['-axo', 'pid=,command='], { encoding: 'utf8', maxBuffer: 64 * MB });
  return out.split('\n').map((l) => l.trim().match(/^(\d+)\s+(.*)$/)).filter(Boolean)
    .map((m) => ({ pid: Number(m[1]), command: m[2] }));
}
const WEBKIT_PROC = /com\.apple\.WebKit\.(WebContent|GPU)\b/;
const webkitPids = () => psRows().filter((r) => WEBKIT_PROC.test(r.command)).map((r) => r.pid);

const SAMPLER_METHOD = {
  tree: 'footprint of the runner process tree, every 100 ms; peak minus median of the samples before page load',
  'tree+webkit': 'footprint of the runner process tree plus com.apple.WebKit.WebContent and com.apple.WebKit.GPU processes that appeared after the browser started, every 100 ms; peak minus median of the samples before page load (best effort, WebKit XPC processes are not in the runner tree)',
  webkit: 'summed footprint of all com.apple.WebKit.WebContent and com.apple.WebKit.GPU processes on the machine, every 100 ms; peak minus median of the samples before page load (best effort: Safari keeps its process pool between sessions, so new processes cannot be told apart, and other WebKit apps add noise to baseline and peak)',
};

function makeSampler(kind) {
  const samples = [];
  let timer = null;
  let before = new Set();
  let loadAt = null;
  const pids = () => {
    if (kind === 'webkit') return webkitPids();
    const fresh = kind === 'tree' ? [] : webkitPids().filter((p) => !before.has(p));
    return [...ownTreePids(), ...fresh];
  };
  const tick = () => {
    try {
      const p = pids();
      samples.push({ t: Date.now(), pids: p.length, bytes: p.length ? footprintBytes(p) : 0 });
    } catch { /* process set changing */ }
  };
  return {
    markBefore() { before = new Set(webkitPids()); },
    start() { timer = setInterval(tick, 100); },
    markLoad() { loadAt = Date.now(); },
    halt() { if (timer) clearInterval(timer); timer = null; },
    stop() {
      if (timer) clearInterval(timer);
      timer = null;
      if (!samples.length) return null;
      tick();
      const pre = samples.filter((s) => s.t < (loadAt ?? Infinity) && s.bytes > 0).slice(0, 15);
      const peak = Math.max(0, ...samples.map((s) => s.bytes));
      const method = SAMPLER_METHOD[kind];
      if (!pre.length || !peak) {
        return {
          method, available: false, baselineMb: null, peakMb: null, peakMinusBaselineMb: null,
          samples: samples.length,
          note: 'no footprint samples for the browser processes (process set empty or footprint failed); no number recorded',
        };
      }
      const baseline = median(pre.map((s) => s.bytes));
      return {
        method, available: true,
        baselineMb: Number((baseline / MB).toFixed(1)), peakMb: Number((peak / MB).toFixed(1)),
        peakMinusBaselineMb: Number(((peak - baseline) / MB).toFixed(1)),
        samples: samples.length, preLoadSamples: pre.length,
        maxPids: Math.max(...samples.map((s) => s.pids)),
      };
    },
  };
}

// ---- frontmost app ---------------------------------------------------------
// A browser window that is not frontmost gets background QoS and throttled
// timers: Safari wasm calls then take 3 to 7 s instead of 0.7 s and single
// calls stall for minutes (K5 diagnosis c). The runner logs the frontmost app
// at every poll (entries only when it changes) and, for drivers that have an
// activate(), brings the browser back to the front unless --no-activate.
const frontLog = [];
function frontApp() {
  try {
    const asn = execFileSync('lsappinfo', ['front'], { encoding: 'utf8' }).trim();
    return /^"([^"]*)"/.exec(execFileSync('lsappinfo', ['info', '-only', 'name', asn], { encoding: 'utf8' }))?.[1] ?? asn;
  } catch (e) { return `unknown (${String(e.message).slice(0, 60)})`; }
}
async function tickFront(drv) {
  const front = frontApp();
  if (!frontLog.length || frontLog[frontLog.length - 1].front !== front) {
    frontLog.push({ t: new Date().toISOString(), front });
  }
  if (!args.has('no-activate') && drv.activate && !drv.frontNames.includes(front)) {
    try { await drv.activate(); } catch { /* best effort */ }
  }
}

// ---- drivers ---------------------------------------------------------------
// A driver: { label, version, flags, notes, base, goto(url), state(),
// progressText(), resultObject(), url(), screenshot(dir), close() }.

function pageDriver(browser, page, { label, flags, notes }) {
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page error]', m.text()); });
  return {
    frontNames: [],
    label: `${label}-${browser.version()}`, version: browser.version(), flags, notes,
    base: `http://127.0.0.1:${PORT}`,
    async goto(url) { await page.goto(url); },
    state: () => page.evaluate(() => document.body.dataset.state),
    progressText: () => page.evaluate(() => document.getElementById('progress')?.textContent),
    resultObject: () => page.evaluate((g) => window[g], resultGlobal()),
    url: () => page.url(),
    async screenshot(dir) {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.screenshot({ path: `${dir}/site-375-${label}.png`, fullPage: true });
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.screenshot({ path: `${dir}/site-desktop-${label}.png`, fullPage: true });
    },
    close: () => browser.close(),
  };
}

function chromiumFlags() {
  const calibration = JSON.parse(readFileSync(resolve(RESULTS, 'calibration-latest.json'), 'utf8'));
  return [...(calibration.flags ?? []), ...(process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean)];
}

async function launch(type, sampler, { label, executablePath, flags = [] }) {
  sampler.markBefore();
  const browser = await type.launch({ headless: false, ...(executablePath ? { executablePath } : {}), args: flags });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  return pageDriver(browser, page, { label, flags, notes: {} });
}

// chromium-nogpu. In Chromium 151 `--disable-webgpu --disable-features=WebGPU`
// leave navigator.gpu present. The candidates below are tried in order on a
// secure-context page and the first one that makes the page's auto stage pick
// wasm (navigator.gpu undefined, or requestAdapter() null) is kept. What each
// attempt observed goes into the run wrapper under flagProbe.
// Observed on Chromium 151.0.7922.34 (macOS, Apple GPU): `--disable-blink-features=WebGPU`
// leaves navigator.gpu an object and requestAdapter() returns an adapter, so it
// does not work; `--disable-gpu --disable-software-rasterizer` keeps navigator.gpu
// an object but requestAdapter() returns null, and the page picks wasm. That
// second set is the one used.
const NOGPU_CANDIDATES = [
  ['--disable-blink-features=WebGPU'],
  ['--disable-gpu', '--disable-software-rasterizer'],
];
async function nogpuDriver(sampler) {
  const attempts = [];
  for (const flags of NOGPU_CANDIDATES) {
    sampler.markBefore();
    const browser = await chromium.launch({ headless: false, args: flags });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.goto(`http://127.0.0.1:${PORT}/device-result.schema.json`);
    const observed = await page.evaluate(async () => {
      const out = { navigatorGpu: typeof navigator.gpu, adapter: 'n/a', error: null };
      try {
        if (navigator.gpu) out.adapter = (await navigator.gpu.requestAdapter()) === null ? 'null' : 'adapter';
      } catch (e) { out.error = String(e); }
      return out;
    });
    const wasm = observed.navigatorGpu === 'undefined' || observed.adapter === 'null';
    attempts.push({ flags, observed, autoPicksWasm: wasm });
    if (wasm) {
      const drv = pageDriver(browser, page, { label: 'chromium-nogpu', flags, notes: { flagProbe: attempts } });
      return drv;
    }
    await browser.close();
  }
  throw new Error(`no nogpu flag set removed WebGPU: ${JSON.stringify(attempts)}`);
}

// Safari through safaridriver, raw WebDriver over HTTP. Long runs poll
// execute/sync; nothing depends on one long request.
async function safariDriver(sampler) {
  const dPort = 4444 + Math.floor(Math.random() * 500);
  const driver = spawn('safaridriver', ['-p', String(dPort)], { stdio: 'ignore' });
  const root = `http://127.0.0.1:${dPort}`;
  const call = async (method, path, body) => {
    const res = await fetch(`${root}${path}`, {
      method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`webdriver ${method} ${path}: ${res.status} ${JSON.stringify(json.value)}`);
    return json.value;
  };
  let sid = null;
  const cleanup = async () => {
    if (sid) { try { await call('DELETE', `/session/${sid}`); } catch { /* gone */ } sid = null; }
    driver.kill();
  };
  try {
    for (let i = 0; i < 100; i += 1) {
      try { if ((await fetch(`${root}/status`)).ok) break; } catch { /* not up yet */ }
      await sleep(100);
    }
    sampler.markBefore();
    const created = await call('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } });
    sid = created.sessionId;
    const version = created.capabilities.browserVersion;
    const exec = (script) => call('POST', `/session/${sid}/execute/sync`, { script, args: [] });
    return {
      label: `safari-${version}`, version, flags: [], base: `http://localhost:${PORT}`,
      frontNames: ['Safari'],
      activate: () => execFileSync('osascript', ['-e', 'tell application "Safari" to activate']),
      notes: { webdriver: { browserName: created.capabilities.browserName, platformName: created.capabilities.platformName } },
      async goto(url) { await call('POST', `/session/${sid}/url`, { url }); },
      state: () => exec('return document.body.dataset.state'),
      progressText: () => exec("return document.getElementById('progress') && document.getElementById('progress').textContent"),
      resultObject: async () => JSON.parse(await exec(`return JSON.stringify(window.${resultGlobal()})`)),
      url: () => call('GET', `/session/${sid}/url`),
      async screenshot(dir) {
        const b64 = await call('GET', `/session/${sid}/screenshot`);
        writeFileSync(`${dir}/site-safari.png`, Buffer.from(b64, 'base64'));
      },
      close: cleanup,
    };
  } catch (e) {
    await cleanup();
    throw e;
  }
}

// Safari on the iPhone over safaridriver. Same raw WebDriver calls as above, with
// the iOS capabilities. visibility() and safariWindows() feed the poll log.
async function iosSafariDriver() {
  const dPort = 4444 + Math.floor(Math.random() * 500);
  const driver = spawn('safaridriver', ['-p', String(dPort)], { stdio: 'ignore' });
  const root = `http://127.0.0.1:${dPort}`;
  const call = async (method, path, body) => {
    const res = await fetch(`${root}${path}`, {
      method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`webdriver ${method} ${path}: ${res.status} ${JSON.stringify(json.value)}`);
    return json.value;
  };
  let sid = null;
  const cleanup = async () => {
    if (sid) { try { await call('DELETE', `/session/${sid}`); } catch { /* gone */ } sid = null; }
    driver.kill();
  };
  const shot = (name) => {
    try { execFileSync('screencapture', ['-x', `${inspectorShots}/${name}`]); return `${inspectorShots}/${name}`; } catch (e) { return `screencapture failed: ${String(e.message).slice(0, 120)}`; }
  };
  const windowNames = () => {
    try {
      return execFileSync('osascript', ['-e', 'tell application "System Events" to if exists process "Safari" then return name of every window of process "Safari"'], { encoding: 'utf8' }).trim();
    } catch (e) { return `osascript failed: ${String(e.message).slice(0, 120)}`; }
  };
  const inspector = {};
  try {
    for (let i = 0; i < 100; i += 1) {
      try { if ((await fetch(`${root}/status`)).ok) break; } catch { /* not up yet */ }
      await sleep(100);
    }
    const alwaysMatch = { browserName: 'safari', platformName: 'iOS', 'safari:useSimulator': false };
    if (args.has('safari-profiling')) alwaysMatch['safari:automaticProfiling'] = true;
    // The first POST right after startup can fail while safaridriver still
    // enumerates devices ("Some devices were found, but could not be used").
    await sleep(3000);
    let created = null;
    for (let attempt = 1; !created; attempt += 1) {
      try { created = await call('POST', '/session', { capabilities: { alwaysMatch } }); } catch (e) {
        if (attempt >= 5) throw e;
        console.log(`session create attempt ${attempt} failed: ${String(e.message).slice(0, 160)}`);
        await sleep(5000);
      }
    }
    sid = created.sessionId;
    const version = created.capabilities.browserVersion;
    const exec = (script) => call('POST', `/session/${sid}/execute/sync`, { script, args: [] });
    return {
      label: `ios-safari-${version}`, version, flags: [], base: IOS_URL, frontNames: [],
      notes: { webdriver: { capabilities: alwaysMatch, returned: created.capabilities }, inspector },
      async goto(url) { await call('POST', `/session/${sid}/url`, { url }); },
      state: () => exec('return document.body.dataset.state'),
      progressText: () => exec("return document.getElementById('progress') && document.getElementById('progress').textContent"),
      visibility: () => exec('return document.visibilityState'),
      safariWindows: windowNames,
      resultObject: async () => JSON.parse(await exec(`return JSON.stringify(window.${resultGlobal()}) || 'null'`)),
      url: () => call('GET', `/session/${sid}/url`),
      async beforeClose() {
        if (!inspectorShots) return;
        mkdirSync(inspectorShots, { recursive: true });
        await sleep(5000);
        try { execFileSync('osascript', ['-e', 'tell application "Safari" to activate']); } catch { /* best effort */ }
        await sleep(1500);
        inspector.windowsBeforeClose = windowNames();
        inspector.shot1 = shot('ios-inspector-1.png');
      },
      async screenshot() { /* page screenshot not needed */ },
      async close() {
        await cleanup();
        if (inspectorShots) {
          await sleep(3000);
          inspector.windowsAfterClose = windowNames();
          inspector.shot2 = shot('ios-inspector-2.png');
        }
      },
    };
  } catch (e) {
    await cleanup();
    throw e;
  }
}

// Safari without WebDriver: the page reports itself (report=local) to the local
// server, the runner only opens the URL and closes the tab afterwards.
async function safariLocalDriver(sampler) {
  sampler.markBefore();
  const version = execFileSync('plutil', ['-extract', 'CFBundleShortVersionString', 'raw',
    '/Applications/Safari.app/Contents/Info.plist'], { encoding: 'utf8' }).trim();
  let rid = null;
  const file = () => resolve(RESULTS, `${rid}.report.json`);
  const closeTabs = () => {
    const script = `tell application "Safari"
  set n to count of windows
  repeat with wi from n to 1 by -1
    try
      set w to window wi
      set m to count of tabs of w
      repeat with ti from m to 1 by -1
        if (URL of tab ti of w) starts with "http://localhost:${PORT}" then close tab ti of w
      end repeat
    end try
  end repeat
end tell`;
    try { execFileSync('osascript', ['-e', script], { encoding: 'utf8' }); return 'closed'; } catch (e) { return `close failed: ${String(e.message).slice(0, 200)}`; }
  };
  return {
    label: `safari-${version}`, version, flags: [], base: `http://localhost:${PORT}`,
    notes: { safariMode: 'local (open -a Safari, report=local, no WebDriver)', activate: !args.has('no-activate') },
    async goto(url) {
      rid = /[?&]rid=([^&]+)/.exec(url)[1];
      execFileSync('open', ['-a', 'Safari', url]);
    },
    frontNames: ['Safari'],
    activate() { execFileSync('osascript', ['-e', 'tell application "Safari" to activate']); },
    state: () => (existsSync(file()) ? (JSON.parse(readFileSync(file(), 'utf8')).error ? 'error' : 'done') : 'running'),
    progressText: () => `waiting for ${file()}`,
    resultObject: () => JSON.parse(readFileSync(file(), 'utf8')),
    url: () => `http://localhost:${PORT}/`,
    async screenshot() { /* none */ },
    async close() {
      console.log(`safari tab: ${closeTabs()}`);
      // The wrapper file holds the result; drop the raw POST body.
      try { if (rid) unlinkSync(file()); } catch { /* not written */ }
    },
  };
}

const BROWSERS = {
  chromium: (s) => launch(chromium, s, { label: 'chromium', flags: chromiumFlags() }),
  brave: (s) => launch(chromium, s, { label: 'brave', executablePath: BRAVE, flags: chromiumFlags() }),
  webkit: (s) => launch(webkit, s, { label: 'webkit' }),
  firefox: (s) => launch(firefox, s, { label: 'firefox' }),
  'chromium-nogpu': nogpuDriver,
  'ios-safari': () => iosSafariDriver(),
  safari: (s) => (safariMode === 'local' ? safariLocalDriver(s) : safariDriver(s)),
};
const SAMPLER_KIND = { webkit: 'tree+webkit', safari: 'webkit' };

// ---- runs.tsv --------------------------------------------------------------

function appendRun(runId, note) {
  const cells = COLS.map(() => '');
  Object.assign(cells, {
    0: new Date().toISOString().slice(0, 10), 1: runId, 2: change, 3: gitCommit(),
    4: diag ? 'diag-readback' : 'kleinhirn-site', 5: diag ? '-' : 'small-upstream',
    6: diag ? '-' : 'L128+L256', 7: diag ? 'plain/done/emptydone' : stages.replaceAll(',', '/'),
    20: 'pending', 21: note,
  });
  appendFileSync(RUNS, cells.join('\t') + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) cells[COLS.indexOf(k)] = String(v).replace(/[\t\n]/g, ' ');
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

async function serverUp() {
  try { return (await fetch(`http://127.0.0.1:${PORT}/index.html`)).ok; } catch { return false; }
}

// ---- load gate -------------------------------------------------------------

async function loadGate() {
  const policy = maxLoad === null
    ? 'none (provisional run)'
    : `max-load ${maxLoad}, ${noWait ? 'no-wait' : `wait up to ${waitMin} min, ${runAnyway ? 'then run anyway' : 'then refuse'}`}`;
  let waitedMin = 0;
  if (maxLoad !== null && !noWait) {
    while (load1() > maxLoad && waitedMin < waitMin) {
      console.log(`1-min load ${load1().toFixed(2)} > ${maxLoad}; waiting (${waitedMin}/${waitMin} min)`);
      await sleep(60000);
      waitedMin += 1;
    }
  }
  const above = maxLoad !== null && load1() > maxLoad;
  if (above && !noWait && !runAnyway) throw new Error(`load ${load1().toFixed(2)} still above ${maxLoad} after ${waitedMin} min`);
  return { policy, waitedMin, startedAboveMax: above };
}

// ---- one run ---------------------------------------------------------------

async function oneRun(rep) {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  const suffix = repIndex !== null ? `-r${repIndex}` : reps > 1 ? `-r${rep}` : '';
  const runId = `${change}-${browserName}-${stamp}${suffix}`;
  const gate = isIos ? { policy: 'none (device run, Mac load recorded only)', waitedMin: 0, startedAboveMax: false } : await loadGate();
  const loadStart = loadStr();
  const commit = gitCommit();
  const dirty = gitDirty();
  const provisional = isIos ? dirty : (maxLoad === null || gate.startedAboveMax || dirty);
  appendRun(runId, `provisional=${provisional}; 1-min load ${loadStart} at start; pending`);
  mkdirSync(RESULTS, { recursive: true });
  const sampler = isIos ? {
    markBefore() {}, start() {}, markLoad() {}, halt() {},
    stop: () => ({ available: false, peakMb: null, note: 'not measurable from the Mac (iOS has no memory web API; Web Inspector timeline only with --safari-profiling)' }),
  } : makeSampler(SAMPLER_KIND[browserName] ?? 'tree');
  let drv = null;
  let t0 = Date.now();
  let backgrounded = false;
  const pollLog = [];
  try {
    drv = await BROWSERS[browserName](sampler);
    sampler.start();
    await sleep(2000); // pre-load samples for the baseline
    sampler.markLoad();
    await drv.goto(`${drv.base}${pagePath()}?${queryOf(runId)}`);
    t0 = Date.now();
    let lastLog = 0;
    let lastWindows = null;
    try {
      for (;;) {
        await tickFront(drv);
        const state = await drv.state();
        if (drv.visibility) {
          const visibility = await drv.visibility();
          const progress = await drv.progressText();
          const windows = args.has('safari-profiling') ? drv.safariWindows() : undefined;
          if (visibility !== 'visible') backgrounded = true;
          pollLog.push({ t: new Date().toISOString(), elapsedS: Math.round((Date.now() - t0) / 1000), state, visibility, progress, ...(windows !== undefined && windows !== lastWindows ? { safariWindows: windows } : {}) });
          if (windows !== undefined) lastWindows = windows;
          console.log(`${pollLog[pollLog.length - 1].t} ${visibility} ${state ?? '-'} ${progress}`);
        }
        if (state === 'done' || state === 'error') break;
        if (Date.now() - t0 > timeoutMs) throw new Error('timeout waiting for body[data-state]');
        if (!drv.visibility && Date.now() - lastLog > 30000) {
          lastLog = Date.now();
          console.log(await drv.progressText());
        }
        await sleep(pollMs);
      }
    } catch (e) {
      if (drv.visibility) {
        const at = new Date().toISOString();
        const last = pollLog[pollLog.length - 1];
        const fail = { runId, failedAt: at, elapsedS: Math.round((Date.now() - t0) / 1000), error: e.message, lastPoll: last ?? null, backgrounded, pollLog, loadAvgStart: loadStart, loadAvgEnd: loadStr() };
        const fbase = `${RESULTS}/k5-iphone16pro-${tag}-${stamp}`;
        writeFileSync(`${fbase}.json`, JSON.stringify({ run: fail, result: null }));
        console.log(`run failed, wrote ${fbase}.json`);
      }
      throw e;
    }
    const result = await drv.resultObject();
    if (drv.beforeClose) await drv.beforeClose();
    const memory = sampler.stop();
    const loadEnd = loadStr();
    const base = isIos ? `${RESULTS}/k5-iphone16pro-${tag}-${stamp}` : `${RESULTS}/${change}-${browserName}-${stamp}${suffix}`;
    const rel = (f) => f.replace(`${OUT}/`, '');
    const run = {
      runId, browser: drv.label, browserVersion: drv.version, flags: drv.flags, ...drv.notes,
      url: await drv.url(), loadAvgStart: loadStart, loadAvgEnd: loadEnd, maxLoadPolicy: gate.policy,
      waitedMin: gate.waitedMin, provisional, commit, dirtyTree: dirty, weights, stages, rep: repIndex ?? rep, reps, extra, serverRoot: process.cwd(), outRoot: OUT, memory,
      frontLog: frontLog.splice(0),
      ...(isIos ? { backgrounded, pollLog, osName, deviceName } : {}),
    };
    writeFileSync(`${base}.json`, JSON.stringify({ run, result }));
    if (result && !result.error) writeFileSync(`${base}.result.json`, JSON.stringify(result, null, 1));
    console.log(`wrote ${base}.json (${statSync(`${base}.json`).size} bytes)`);
    if (screenshotDir) {
      mkdirSync(screenshotDir, { recursive: true });
      await drv.screenshot(screenshotDir);
    }
    if (result?.error) {
      finishRun(runId, { note: `failed: ${result.error}; ${rel(base)}.json`, browser: drv.label });
      throw new Error(`page: ${result.error}`);
    }
    if (diag) {
      const v = result.variants;
      const cell = (k, m) => Object.keys(v).map((name) => v[name][k].toFixed(m)).join('/');
      finishRun(runId, {
        median_ms: cell('medianMs', 2), p95_ms: cell('p95Ms', 2), browser: drv.label,
        adapter: JSON.stringify(result.adapter),
        note: `diag readback variants ${Object.keys(v).join('/')}; n=${result.n}; load ${loadStart} -> ${loadEnd}; ${rel(base)}.json`,
      });
      console.log(Object.entries(v).map(([n, x]) => `${n} median ${x.medianMs.toFixed(2)} p95 ${x.p95Ms.toFixed(2)} min ${x.minMs.toFixed(2)} max ${x.maxMs.toFixed(2)}`).join('\n'));
      return;
    }
    const per = (f) => result.stages.map((s) => (s.ok ? f(s) : 'ERR')).join('/');
    const lat = (s, k, w, m) => s.latency[k][w][m].toFixed(2);
    const par = (s, m) => Math.max(...['L128', 'L256'].map((k) => s.parity[k].summary[m]));
    const adapter = result.environment.webgpu?.adapter?.info ?? {};
    finishRun(runId, {
      argmax_agreement: per((s) => Math.min(...['L128', 'L256'].map((k) => s.parity[k].summary.argmaxAgreement)).toFixed(4)),
      max_abs_logit_diff: per((s) => par(s, 'maxAbsLogitDiff').toExponential(2)),
      max_abs_prob_diff: per((s) => par(s, 'maxAbsProbDiff').toExponential(2)),
      median_ms: per((s) => lat(s, 'L128', 'e2e', 'medianMs')),
      p95_ms: per((s) => lat(s, 'L128', 'e2e', 'p95Ms')),
      model_only_median_ms: per((s) => lat(s, 'L128', 'modelOnly', 'medianMs')),
      load_ms: per((s) => s.load.wallMs.toFixed(0)),
      download_mb: per((s) => ((s.load.downloadBytes ?? 0) / MB).toFixed(1)),
      gpu_mb: per((s) => ((s.load.gpuBytes ?? 0) / MB).toFixed(0)),
      peak_mem_mb: memory?.available ? memory.peakMinusBaselineMb.toFixed(0) : 'n/a',
      browser: drv.label,
      adapter: JSON.stringify({ vendor: adapter.vendor, architecture: adapter.architecture }),
      kept: '',
      note: `provisional=${provisional}; ${isIos ? `backgrounded=${backgrounded}; ` : ''}load ${loadStart} -> ${loadEnd}; weights=${weights}; `
        + `auto=${result.autoStage?.picked ?? 'none'}; L256 model-only median ${per((s) => lat(s, 'L256', 'modelOnly', 'medianMs'))}; ${rel(base)}.json`,
    });
    for (const s of result.stages) {
      console.log(s.name, s.ok ? `ok parity=${s.parityPass}` : `FAILED ${s.error}`);
    }
  } catch (e) {
    try { finishRun(runId, { note: `failed: ${e.message}` }); } catch { /* line missing */ }
    throw e;
  } finally {
    sampler.halt();
    if (drv) await drv.close().catch((e) => console.log('close failed:', e.message));
  }
}

async function main() {
  if (!BROWSERS[browserName]) throw new Error(`unknown browser ${browserName}; one of ${Object.keys(BROWSERS).join(', ')}`);
  if (isIos) { await oneRun(1); return; }
  if (!existsSync('site/dist/index.html')) throw new Error('site/dist missing; run npm run build:site');
  // Always our own server: a leftover one could serve another checkout's dist.
  if (await serverUp()) throw new Error(`port ${PORT} is already served; stop that server first`);
  const server = spawn('node', ['tools/serve_site.mjs', String(PORT), '--accept-results', '--results-dir', RESULTS], { stdio: 'inherit' });
  for (let i = 0; i < 50 && !(await serverUp()); i += 1) await sleep(100);
  if (!(await serverUp())) throw new Error('server did not start');
  try {
    for (let rep = 1; rep <= reps; rep += 1) await oneRun(rep);
  } finally {
    if (server) server.kill();
  }
}

await main();
