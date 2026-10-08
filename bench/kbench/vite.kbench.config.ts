// R1 bench server: the K28.8 server (cross-origin isolation) plus all three ORT Web builds bundled
// by the dependency optimizer at startup. Without the list Vite found onnxruntime-web/webgpu and
// /jspi only when the first page imported them, bundled again under a new hash, and pages that
// still asked for the old hash failed to import it (Firefox f32 memory pass, 06.10., 15 cells).
// bench/k28/vite.k28-8.config.ts stays as it is: the official K28.8 numbers hang on it.
// R8: the isolation headers go on every response. Vite's server.headers misses 304 answers, and
// WebKit then blocks a worker whose script it loads a second time (r8-prof, 07.10.).
// KBENCH_NO_COI=1 (R8): the same server without isolation headers, for the threads fallback and the
// no-isolation report of gate R8 (pages then see crossOriginIsolated false).
import { defineConfig, mergeConfig, type Plugin } from 'vite';
import base from '../../vite.config.ts';
import k288 from '../k28/vite.k28-8.config.ts';

const noCoi = process.env.KBENCH_NO_COI === '1';

const isolationOnEveryResponse: Plugin = {
  name: 'kh-isolation-headers',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      next();
    });
  },
};

export default mergeConfig(noCoi ? base : k288, defineConfig({
  plugins: noCoi ? [] : [isolationOnEveryResponse],
  optimizeDeps: {
    include: ['onnxruntime-web', 'onnxruntime-web/webgpu', 'onnxruntime-web/jspi'],
  },
}));
