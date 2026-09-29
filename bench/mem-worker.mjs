// Worker-thread memory sampler. The metric calls (pgrep, footprint)
// are synchronous and can take seconds on a large Chromium process
// tree; inside a worker they block only this thread, never the
// Playwright event loop. Posts {t, bytes} every intervalMs; a failed
// probe posts {t, bytes: null}.

import { parentPort, workerData } from 'node:worker_threads';
import { METRICS, ownTreePids } from './mem.mjs';

const metric = METRICS[workerData.metric];
if (!metric) throw new Error(`unknown metric ${workerData.metric}`);
const rootPid = workerData.rootPid;
const intervalMs = workerData.intervalMs ?? 1000;

setInterval(() => {
  let bytes = null;
  try {
    bytes = metric(ownTreePids(rootPid));
  } catch { /* process tree mid-change */ }
  parentPort.postMessage({ t: Date.now(), bytes });
}, intervalMs);
