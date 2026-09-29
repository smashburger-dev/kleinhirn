import { chromium } from '@playwright/test';
const browser = await chromium.launch({ headless: false });
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
await page.goto('http://localhost:5199/bench/dbg.html');
const out = await page.evaluate(async () => {
  try {
    const { Kleinhirn } = await import('/dist/kleinhirn.js');
    const lines = [];
    for (const prec of ['f16', 'f32']) {
      const t0 = performance.now();
      const kh = await Kleinhirn.load({
        manifestUrl: `/models/small-upstream/${prec}/manifest.json`,
        buckets: [128], limits: 'minimum',
      });
      const total = performance.now() - t0;
      lines.push(`${prec} total=${total.toFixed(0)}ms timing=${JSON.stringify(kh.info().loadTiming)}`);
      kh.dispose();
    }
    return lines.join('\n');
  } catch (e) { return 'ERR ' + String(e); }
});
console.log(out);
await browser.close();
