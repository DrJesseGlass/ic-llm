import { useEffect, useState, useCallback, useRef } from 'react';
import init, { Model } from '../../assets/wasm/candle_wasm_example_quant_qwen3.js';

export function useQwenModel() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const modelRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      try {
        setLoadProgress(5);

        // Initialize WASM
        await init();

        if (cancelled) return;
        setLoadProgress(15);

        // Fetch weights with fallback progress (chunked transfer)
        const weightsResponse = await fetch('/assets/wasm/Qwen3-0.6B-Q8_0.gguf');
        if (!weightsResponse.ok) throw new Error('Failed to load model weights');

        // DEBUG: Log all response headers
        console.log('Response headers:');
        for (let [key, value] of weightsResponse.headers.entries()) {
          console.log(`  ${key}: ${value}`);
        }

        const weightsTotal = parseInt(weightsResponse.headers.get('content-length'));
        const hasContentLength = weightsTotal && weightsTotal > 1;

        console.log('Content-Length:', weightsTotal, 'Has valid length:', hasContentLength);

        const weightsReader = weightsResponse.body.getReader();
        const weightsChunks = [];
        let weightsLoaded = 0;
        const startTime = Date.now();
        const EXPECTED_SIZE = 640 * 1024 * 1024; // ~640MB estimate

        while (true) {
          const { done, value } = await weightsReader.read();
          if (done) break;

          weightsChunks.push(value);
          weightsLoaded += value.length;

          // Progress updates
          if (hasContentLength && weightsTotal > 1) {
            // Real progress based on actual content-length
            setLoadProgress(15 + (weightsLoaded / weightsTotal) * 76);
          } else {
            // Fallback: estimate based on bytes loaded (assume 640MB total)
            const estimatedProgress = Math.min(91, 15 + (weightsLoaded / EXPECTED_SIZE) * 76);
            setLoadProgress(estimatedProgress);
          }

          // Log progress every 50MB
          if (weightsLoaded % (50 * 1024 * 1024) < value.length) {
            console.log(`Downloaded: ${(weightsLoaded / (1024 * 1024)).toFixed(1)}MB`);
          }

          // Yield to event loop every 10MB to keep UI responsive
          if (weightsLoaded % (10 * 1024 * 1024) < value.length) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        console.log(`Total downloaded: ${(weightsLoaded / (1024 * 1024)).toFixed(1)}MB`);
        setLoadProgress(92);
        await new Promise(resolve => setTimeout(resolve, 100));

        // More efficient array combination
        console.log('Combining chunks into single array...');
        const weights = new Uint8Array(weightsLoaded);
        let offset = 0;
        for (const chunk of weightsChunks) {
          weights.set(chunk, offset);
          offset += chunk.length;

          // Yield every 50MB during combining
          if (offset % (50 * 1024 * 1024) < chunk.length) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }
        weightsChunks.length = 0; // Free memory
        console.log('Array combined');

        if (cancelled) return;
        setLoadProgress(95);
        await new Promise(resolve => setTimeout(resolve, 50));

        // Fetch tokenizer (small, no progress needed)
        const tokenizerResponse = await fetch('/assets/wasm/tokenizer.json');
        if (!tokenizerResponse.ok) throw new Error('Failed to load tokenizer');
        const tokenizerData = await tokenizerResponse.arrayBuffer();
        const tokenizer = new Uint8Array(tokenizerData);

        if (cancelled) return;
        setLoadProgress(98);

        // Create model instance - pass empty config
        const modelInstance = new Model(
          weights,
          tokenizer,
          new Uint8Array(0) // Empty config
        );

        if (cancelled) {
          modelInstance.reset();
          return;
        }

        modelRef.current = modelInstance;
        setModel(modelInstance);
        setLoadProgress(100);
        setLoading(false);

      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    }

    loadModel();

    return () => {
      cancelled = true;
      if (modelRef.current) {
        modelRef.current.reset();
      }
    };
  }, []);

  const generate = useCallback(async (prompt, options = {}) => {
    if (!model) throw new Error('Model not loaded');

    const {
      maxTokens = 100,
      temperature = 0.0, //0.7,
      topP = 0.0, //0.9,
      repeatPenalty = 1.1,
      repeatLastN = 64,
      seed = 42, //Date.now(),
      enableThinking = true,
      onToken = () => {},
      signal = null
    } = options;

    let fullText = prompt;

    // Initialize with prompt
    const firstToken = model.init_with_prompt(
      prompt,
      temperature,
      topP,
      repeatPenalty,
      repeatLastN,
      seed,
      enableThinking
    );

    fullText += firstToken;
    onToken(firstToken, 1);

    // Generate tokens
    for (let i = 1; i < maxTokens; i++) {
      if (signal?.aborted) break;
      if (model.is_eos()) break;

      const token = model.next_token();

      // Stop if we hit special tokens
      if (token.includes('<|im_end|>') || token.includes('<|endoftext|>')) {
        break;
      }

      fullText += token;
      onToken(token, i + 1);

      // Yield to event loop every 10 tokens
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    return fullText;
  }, [model]);

  const reset = useCallback(() => {
    if (model) {
      model.reset();
    }
  }, [model]);

  return {
    model,
    loading,
    error,
    loadProgress,
    generate,
    reset
  };
}
