// Node loader hooks for tests that import engine code with GPU-free mocks.
// Two things plain Node cannot load from src/: Vite's `*.wgsl?raw` imports
// and TypeScript parameter properties (type stripping rejects them). The
// hooks serve the kernel text as a default export and transpile src/*.ts
// with the repo's own TypeScript.
// Usage: node --import ./tests/helpers/node-hooks.mjs --test tests/<name>.test.ts

import { createRequire, registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const ts = createRequire(import.meta.url)('typescript');

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.wgsl?raw')) {
      const resolved = nextResolve(specifier.slice(0, -4), context);
      return { url: `${resolved.url}?raw`, format: 'module', shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.endsWith('.wgsl?raw')) {
      const text = readFileSync(fileURLToPath(url.slice(0, -4)), 'utf8');
      return {
        format: 'module',
        source: `export default ${JSON.stringify(text)};`,
        shortCircuit: true,
      };
    }
    if (url.endsWith('.ts') && url.includes('/src/')) {
      const out = ts.transpileModule(readFileSync(fileURLToPath(url), 'utf8'), {
        compilerOptions: {
          module: ts.ModuleKind.ESNext,
          target: ts.ScriptTarget.ES2022,
          verbatimModuleSyntax: true,
        },
      });
      return { format: 'module', source: out.outputText, shortCircuit: true };
    }
    return nextLoad(url, context);
  },
});
