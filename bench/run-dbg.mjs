// Diagnostic runner: loads bench/dbg.html and dumps window.khDbg.
// Usage: node bench/run-dbg.mjs
import { chromium } from '@playwright/test';

const browser = await chromium.launch({ headless: false });
try {
  const page = await browser.newPage();
  await new Promise((r) => setTimeout(r, 1500));
  await page.goto('http://localhost:5199/bench/dbg.html');
  const t0 = Date.now();
  while (Date.now() - t0 < 600000) {
    if (await page.evaluate(() => window.khDbg?.done ?? false)) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  console.log(JSON.stringify(await page.evaluate(() => window.khDbg), null, 1));
} finally {
  await browser.close();
}
