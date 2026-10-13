import { defineConfig } from 'vite';
import { wgslCompact } from './tools/wgsl_compact.ts';

// R4 G3 test switch: KH_NO_RANGE=1 drops the Range header of every /models/ request, so the
// server answers 200 with the whole file and RemoteRows must fall back to the full download.
const noRange = {
  name: 'kh-no-range',
  configureServer(server: { middlewares: { use: (fn: (req: { url?: string; headers: Record<string, unknown> }, res: unknown, next: () => void) => void) => void } }) {
    if (process.env.KH_NO_RANGE !== '1') return;
    server.middlewares.use((req, _res, next) => {
      if (req.url?.startsWith('/models/')) delete req.headers.range;
      next();
    });
  },
};

export default defineConfig({
  // Relative asset URLs: the lib bundle emits the wasm worker as
  // assets/wasm-plan-worker-*.js and resolves it against import.meta.url; an
  // absolute base would point at the origin root and 404 whenever the
  // bundle is not served from /.
  base: './',
  // R6 stage 1: kernels load without comments and spare whitespace, in build and dev alike, so the
  // parity pages run the shaders the bundle ships.
  plugins: [wgslCompact(), noRange],
  server: {
    port: 5199,
    strictPort: true,
  },
  resolve: {
    // Official runs live in a git worktree whose models/ dir is a symlink
    // to the private repo. Without preserveSymlinks vite realpaths the
    // upstream julia bench pages outside the root and the html-proxy
    // lookup fails with an error overlay that blocks page clicks.
    preserveSymlinks: true,
  },
  define: {
    // Stamped into the lib bundle and exposed via Kleinhirn.info().buildId.
    // The dev server can serve a stale transform of dist/, so runners compare
    // this value against the file on disk before trusting a measurement.
    __KH_BUILD_ID__: JSON.stringify(Date.now().toString(36)),
  },
  // R8 hc6: the executor worker loads one build of plan.wasm per browser through a dynamic
  // import; code splitting in a worker needs the es format (the worker starts as type module).
  worker: { format: 'es' },
  build: {
    lib: {
      entry: { kleinhirn: 'src/index.ts', tokenizer: 'src/tokenizer-entry.ts' },
      formats: ['es'],
      fileName: (_format, name) => `${name}.js`,
    },
    // R6: modules shared by the core and a lazy chunk (cache.ts, device.ts, half.ts, ...) stay in
    // kleinhirn.js, which then exports them to the chunk. With the library default 'strict' they
    // became an extra static chunk that every load fetched before the core could run.
    rolldownOptions: { preserveEntrySignatures: 'allow-extension' },
  },
});
