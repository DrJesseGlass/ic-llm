// src/frontend/src/components/ModelLoader.jsx
import { useEffect, useState } from 'react';

export function useQwenModel() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadModel() {
      try {
        // Fetch WASM runtime and model weights from canister assets
        const [wasmResponse, weightsResponse] = await Promise.all([
          fetch('/wasm/ml-runtime.wasm'),
          fetch('/models/model-weights.bin')
        ]);

        const wasmBuffer = await wasmResponse.arrayBuffer();
        const weightsBuffer = await weightsResponse.arrayBuffer();

        // Initialize your WASM model (adjust to your library's API)
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const modelInstance = await initializeQwen(wasmModule, weightsBuffer);

        setModel(modelInstance);
      } catch (err) {
        console.error('Model loading failed:', err);
      } finally {
        setLoading(false);
      }
    }

    loadModel();
  }, []);

  return { model, loading };
}