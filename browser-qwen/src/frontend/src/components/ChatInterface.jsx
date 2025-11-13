// src/frontend/src/components/ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { useQwenModel } from '../hooks/useQwenModel';
import './ChatInterface.css';

function parseResponse(fullText, enableThinking) {
  if (enableThinking) {
    const lastThinkStart = fullText.lastIndexOf('<think>');
    const lastThinkEnd = fullText.lastIndexOf('</think>');

    // If last <think> comes after last </think>, we're still in thinking mode
    if (lastThinkStart > lastThinkEnd) {
      // Currently inside a <think> block - everything is thinking
      return {
        thinking: cleanTokens(fullText),
        response: ''
      };
    } else if (lastThinkEnd !== -1) {
      // We have a closed </think> and no <think> after it - split here
      const thinkingRaw = fullText.substring(0, lastThinkEnd);
      const responseRaw = fullText.substring(lastThinkEnd + 8); // Skip '</think>'

      return {
        thinking: cleanTokens(thinkingRaw),
        response: cleanTokens(responseRaw)
      };
    } else {
      // No tags yet - still building initial thinking
      return {
        thinking: cleanTokens(fullText),
        response: ''
      };
    }
  } else {
    // Non-thinking mode: everything is response
    return {
      thinking: '',
      response: cleanTokens(fullText)
    };
  }
}

function cleanTokens(text) {
  return text
    .replace(/<\|im_start\|>/g, '')
    .replace(/<\|im_end\|>/g, '')
    .replace(/<\|endoftext\|>/g, '')
    .replace(/<think>/g, '')
    .replace(/<\/think>/g, '')
    .trim();
}

export function ChatInterface() {
  const { model, loading, error, loadProgress, generate, reset } = useQwenModel();

  const [prompt, setPrompt] = useState('What is the capital of France?');
  const [maxTokens, setMaxTokens] = useState(100);
  const [output, setOutput] = useState('');
  const [thinkingOutput, setThinkingOutput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [stats, setStats] = useState(null);
  const [enableThinking, setEnableThinking] = useState(true);
  const [showThinking, setShowThinking] = useState(false);

  const abortControllerRef = useRef(null);
  const outputRef = useRef(null);
  const fullResponseRef = useRef('');

  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const handleGenerate = async () => {
    if (!model || generating) return;

    setGenerating(true);
    setOutput('');
    setThinkingOutput('');
    setStats(null);
    fullResponseRef.current = '';

    reset();

    abortControllerRef.current = new AbortController();
    const startTime = Date.now();
    let tokenCount = 0;

    try {

      await generate(prompt, {
        maxTokens,
        enableThinking,
        temperature: 0.0, //0.7,
        topP: 0.0, //0.9,
        repeatPenalty: 1.1,
        repeatLastN: 64,
        seed: 42, //Date.now(),
        onToken: (token, count) => {
          fullResponseRef.current += token;
          tokenCount = count;

          const fullText = fullResponseRef.current;

          const { thinking, response } = parseResponse(fullText, enableThinking);

          setThinkingOutput(thinking);
          setOutput(response);

          // Update stats every 10 tokens
          if (count % 10 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const tokPerSec = (count / elapsed).toFixed(2);
            setStats({
              tokens: count,
              time: elapsed.toFixed(2),
              speed: tokPerSec
            });
          }
        },

        signal: abortControllerRef.current.signal
      });

      // Final stats
      const totalTime = (Date.now() - startTime) / 1000;
      setStats({
        tokens: tokenCount,
        time: totalTime.toFixed(2),
        speed: (tokenCount / totalTime).toFixed(2)
      });

    } catch (err) {
      console.error('Generation error:', err);
      setOutput(prev => prev + '\n\n[Error: ' + err.message + ']');
    } finally {
      setGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleReset = () => {
    reset();
    setOutput('');
    setThinkingOutput('');
    setStats(null);
    fullResponseRef.current = '';
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <h2>Loading Qwen3 Model...</h2>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${loadProgress}%` }}
          ></div>
        </div>
        <p>{loadProgress.toFixed(0)}% complete</p>
        <p className="loading-hint">
          First load may take a minute (downloading ~645MB)
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error Loading Model</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>
          Reload Page
        </button>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Qwen3-0.6B in Browser</h1>
        <p>Running locally with WebAssembly + SIMD</p>
      </header>

      <div className="chat-main">
        <div className="input-section">
          <label htmlFor="prompt">Prompt:</label>
          <input
            id="prompt"
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={generating}
            placeholder="Ask me anything..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !generating) {
                handleGenerate();
              }
            }}
          />

          <div className="controls-row">
            <div className="control-group">
              <label htmlFor="maxTokens">Max Tokens:</label>
              <input
                id="maxTokens"
                type="number"
                min="1"
                max="500"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                disabled={generating}
              />
            </div>

            <div className="control-group">
              <label htmlFor="enableThinking">
                <input
                  id="enableThinking"
                  type="checkbox"
                  checked={enableThinking}
                  onChange={(e) => setEnableThinking(e.target.checked)}
                  disabled={generating}
                />
                Enable Thinking
              </label>
            </div>

            <div className="button-group">
              <button
                className="btn-primary"
                onClick={handleGenerate}
                disabled={generating || !prompt.trim()}
              >
                {generating ? 'Generating...' : 'Generate'}
              </button>

              {generating && (
                <button
                  className="btn-stop"
                  onClick={handleStop}
                >
                  Stop
                </button>
              )}

              <button
                className="btn-secondary"
                onClick={handleReset}
                disabled={generating}
              >
                Reset
              </button>
            </div>
          </div>

          {stats && (
            <div className="stats-bar">
              <span>📊 {stats.tokens} tokens</span>
              <span>⏱️ {stats.time}s</span>
              <span>⚡ {stats.speed} tok/s</span>
            </div>
          )}
        </div>

        {/* Thinking Section (collapsible) */}
        {thinkingOutput && (
          <div className="thinking-section">
            <button
              className="thinking-toggle"
              onClick={() => setShowThinking(!showThinking)}
            >
              {showThinking ? '▼' : '▶'} Reasoning Process
            </button>
            {showThinking && (
              <div className="thinking-output">
                {thinkingOutput}
              </div>
            )}
          </div>
        )}

        {/* Main Response Section */}
        <div className="output-section">
          <label>Response:</label>
          <div
            ref={outputRef}
            className="output-box"
          >
            {output || <span className="placeholder">Generated text will appear here...</span>}
          </div>
        </div>
      </div>

      <footer className="chat-footer">
        <p>
          Model: Qwen3-0.6B-Q8 • Framework: Candle • Built with Rust + WASM
        </p>
      </footer>
    </div>
  );
}