import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs: the lib bundle emits the wasm worker as
  // assets/wasm-plan-worker-*.js and resolves it against import.meta.url; an
  // absolute base would point at the origin root and 404 whenever the
  // bundle is not served from /.
  base: './',
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
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: () => 'kleinhirn.js',
    },
  },
});
