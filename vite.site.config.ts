// Builds the static device page (site/) into site/dist/. The engine is not
// bundled into the page: the build copies dist/kleinhirn.js and its wasm
// worker unchanged next to the page, so the page can hash the exact bytes it
// runs. Run `npm run build:site` (builds the engine first).
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const git = (...args: string[]): string =>
  execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();

// A copy without .git (a source archive) has no commit to stamp.
const stampCommit = (): string => {
  try {
    return git('rev-parse', '--short', 'HEAD')
      + (git('status', '--porcelain', '--untracked-files=no') ? '-dirty' : '');
  } catch {
    return 'unknown';
  }
};
const commit = stampCommit();

export default defineConfig({
  root: 'site',
  base: './',
  publicDir: 'public',
  define: {
    __SITE_BUILD_ID__: JSON.stringify(Date.now().toString(36)),
    __SITE_COMMIT__: JSON.stringify(commit),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
  },
  plugins: [{
    name: 'copy-engine-bundle',
    apply: 'build',
    writeBundle() {
      const engine = resolve('dist/kleinhirn.js');
      const assets = resolve('dist/assets');
      if (!existsSync(engine) || !existsSync(assets)) {
        throw new Error('dist/kleinhirn.js missing; run `npm run build` first');
      }
      const out = resolve('site/dist');
      mkdirSync(resolve(out, 'assets'), { recursive: true });
      copyFileSync(engine, resolve(out, 'kleinhirn.js'));
      // the chunks the engine loads on demand: dist/wasm-backend-*.js and the plan worker in assets/
      const chunks = readdirSync(resolve('dist')).filter((f) => /^wasm-backend-.*\.js$/.test(f));
      const workers = readdirSync(assets).filter((f) => /^wasm-plan-worker-.*\.js$/.test(f));
      if (!chunks.length || !workers.length) throw new Error('no dist/wasm-backend-*.js or dist/assets/wasm-plan-worker-*.js');
      for (const f of chunks) copyFileSync(resolve('dist', f), resolve(out, f));
      for (const f of workers) copyFileSync(resolve(assets, f), resolve(out, 'assets', f));
    },
  }],
});
