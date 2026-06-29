import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Threads need SharedArrayBuffer, which requires cross-origin isolation. The
// asset canister sets these in public/.ic-assets.json5; mirror them here so the
// vite dev/preview servers are isolated too.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react()],
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      }
    }
  },
  publicDir: 'public',
  assetsInclude: ['**/*.wasm', '**/*.gguf'],
});