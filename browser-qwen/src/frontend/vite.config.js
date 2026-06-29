import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Cross-origin isolation for SharedArrayBuffer (threads); matches .ic-assets.json5.
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