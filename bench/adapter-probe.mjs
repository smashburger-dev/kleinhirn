// Print WebGPU adapterInfo as JSON and exit non-zero when no adapter
// is available. Used as the hardware gate before cloud runs.

import { createServer } from 'node:http';
import { chromium } from '@playwright/test';

// navigator.gpu is SecureContext-only; about:blank is not secure, so
// the probe serves an empty page over localhost.
const server = createServer((req, res) => {
  res.setHeader('content-type', 'text/html');
  res.end('<html></html>');
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;

const args = (process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean);
const browser = await chromium.launch({ headless: false, args });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/`);
  const info = await page.evaluate(async () => {
    if (!navigator.gpu) return { error: 'navigator.gpu missing' };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: 'requestAdapter returned null' };
    // GPUAdapterInfo attributes are non-enumerable; JSON.stringify on
    // the object itself yields {}. Read fields explicitly (vendor and
    // device ids are numeric, e.g. NVIDIA = 0x10de).
    const i = adapter.info;
    return {
      vendor: i.vendor, architecture: i.architecture,
      device: i.device, description: i.description,
      subgroupMinSize: i.subgroupMinSize, subgroupMaxSize: i.subgroupMaxSize,
      isFallbackAdapter: i.isFallbackAdapter,
      features: [...adapter.features].sort(),
    };
  });
  console.log(JSON.stringify(info));
  if (!info || info.error) process.exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
