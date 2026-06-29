import { useEffect, useState, useCallback, useRef } from 'react';

// Static path (not a bundled import) so import.meta.url anchors at /assets/wasm/
// and the rayon worker resolves its snippets there. @vite-ignore prevents bundling.
const WASM_MODULE_URL = '/assets/wasm/candle_wasm_example_quant_qwen3.js';

// Cap the rayon pool; 4 balances decode throughput against per-thread wasm memory.
const MAX_WORKER_THREADS = 4;

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

        // Start both downloads before wasm init so connection setup and transfer
        // overlap with instantiation and the thread-pool spawn; the bytes are only
        // consumed once init completes.
        const weightsPromise = fetch('/assets/wasm/Qwen3-0.6B-allq4k-f16src.gguf');
        const tokenizerPromise = fetch('/assets/wasm/tokenizer.json');

        // Init wasm, then the rayon pool; single-threaded if not cross-origin isolated.
        const { default: init, ModelLoader, initThreadPool } =
          await import(/* @vite-ignore */ WASM_MODULE_URL);
        await init();
        if (globalThis.crossOriginIsolated) {
          const threads = Math.max(1, Math.min(MAX_WORKER_THREADS, navigator.hardwareConcurrency || MAX_WORKER_THREADS));
          await initThreadPool(threads);
        } else {
          console.warn('Not cross-origin isolated; running single-threaded (check COOP/COEP headers).');
        }

        if (cancelled) return;
        setLoadProgress(15);

        // Fetch weights with fallback progress (chunked transfer)
        const weightsResponse = await weightsPromise;
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
        const loader = new ModelLoader();
        let weightsLoaded = 0;
        const EXPECTED_SIZE = 326 * 1024 * 1024; // ~326MB (all-Q4_K)

        // Stream each chunk straight into the wasm loader, which quantizes tensors
        // in file order and frees consumed bytes. The whole file is never held in
        // JS or wasm at once, roughly halving peak memory vs the all-at-once load.
        while (true) {
          const { done, value } = await weightsReader.read();
          if (done) break;

          loader.push(value);
          weightsLoaded += value.length;

          const total = hasContentLength ? weightsTotal : EXPECTED_SIZE;
          const pct = 15 + (weightsLoaded / total) * 80;
          setLoadProgress(hasContentLength ? pct : Math.min(95, pct));

          if (weightsLoaded % (50 * 1024 * 1024) < value.length) {
            console.log(`Streamed: ${(weightsLoaded / (1024 * 1024)).toFixed(1)}MB`);
          }
          // Yield to the event loop every 10MB to keep the UI responsive.
          if (weightsLoaded % (10 * 1024 * 1024) < value.length) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        console.log(`Total streamed: ${(weightsLoaded / (1024 * 1024)).toFixed(1)}MB`);
        if (cancelled) return;
        setLoadProgress(96);

        // Tokenizer downloaded in parallel with the weights stream; finalize the model.
        const tokenizerResponse = await tokenizerPromise;
        if (!tokenizerResponse.ok) throw new Error('Failed to load tokenizer');
        const tokenizer = new Uint8Array(await tokenizerResponse.arrayBuffer());

        if (cancelled) return;
        setLoadProgress(98);

        const modelInstance = loader.finish(tokenizer);

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

  // Start a new conversation
  const startConversation = useCallback((systemPrompt = null, enableThinking = true) => {
    if (!model) return;
    model.start_conversation(systemPrompt, enableThinking);
  }, [model]);

  // Send a message in the conversation (multi-turn with KV cache reuse)
  const chat = useCallback(async (message, options = {}) => {
    if (!model) throw new Error('Model not loaded');

    const {
      maxTokens = 500,
      temperature = 0.6,
      topP = 0.9,
      repeatPenalty = 1.1,
      repeatLastN = 64,
      seed = Date.now(),
      enableThinking = true,
      onToken = () => {},
      signal = null
    } = options;

    let fullResponse = '';

    // Start chat turn - uses KV cache efficiently!
    const firstToken = model.chat(
      message,
      temperature,
      topP,
      repeatPenalty,
      repeatLastN,
      seed,
      enableThinking
    );

    fullResponse += firstToken;
    onToken(firstToken, 1);

    // Generate remaining tokens
    for (let i = 1; i < maxTokens; i++) {
      if (signal?.aborted) break;
      if (model.is_eos()) break;

      const token = model.next_token();

      // Stop if we hit special tokens
      if (token.includes('<|im_end|>') || token.includes('<|endoftext|>')) {
        break;
      }

      fullResponse += token;
      onToken(token, i + 1);

      // Yield to event loop every 10 tokens
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    // End the turn - records response in conversation history, keeps KV cache
    model.end_turn();

    return fullResponse;
  }, [model]);

  // Clear conversation but keep system prompt
  const clearConversation = useCallback(() => {
    if (model) {
      model.clear_conversation();
    }
  }, [model]);

  // Full reset (alias for clearConversation for compatibility)
  const reset = useCallback(() => {
    if (model) {
      model.reset();
    }
  }, [model]);

  // Get conversation stats
  const getStats = useCallback(() => {
    if (!model) return { messages: 0, cachedTokens: 0 };
    return {
      messages: model.get_message_count(),
      cachedTokens: model.get_cached_token_count()
    };
  }, [model]);

  // Get conversation history as JSON
  const getConversationJson = useCallback(() => {
    if (!model) return '[]';
    return model.get_conversation_json();
  }, [model]);

  return {
    model,
    loading,
    error,
    loadProgress,
    startConversation,
    chat,
    clearConversation,
    reset,
    getStats,
    getConversationJson
  };
}