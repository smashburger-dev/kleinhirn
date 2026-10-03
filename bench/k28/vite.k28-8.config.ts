// K28.8 Festlegung 5: both sides of the ORT comparison run on one Vite server
// with cross-origin isolation. Without it Chromium rounds performance.now() to
// 100 µs. Same settings as vite.config.ts otherwise; the pages record
// crossOriginIsolated and the measured timer step.
// Usage: vite --config bench/k28/vite.k28-8.config.ts --port <port> (cwd: worktree root)
import { defineConfig, mergeConfig } from 'vite';
import base from '../../vite.config.ts';

export default mergeConfig(base, defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
}));
