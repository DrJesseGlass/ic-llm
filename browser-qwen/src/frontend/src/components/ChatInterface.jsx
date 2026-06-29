// src/frontend/src/components/ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { useQwenModel } from '../hooks/useQwenModel';
import './ChatInterface.css';

function parseResponse(fullText, enableThinking) {
  if (enableThinking) {
    const lastThinkStart = fullText.lastIndexOf('<think>');
    const lastThinkEnd = fullText.lastIndexOf('</think>');

    if (lastThinkStart > lastThinkEnd) {
      return {
        thinking: cleanTokens(fullText),
        response: ''
      };
    } else if (lastThinkEnd !== -1) {
      const thinkingRaw = fullText.substring(0, lastThinkEnd);
      const responseRaw = fullText.substring(lastThinkEnd + 8);
      return {
        thinking: cleanTokens(thinkingRaw),
        response: cleanTokens(responseRaw)
      };
    } else {
      return {
        thinking: cleanTokens(fullText),
        response: ''
      };
    }
  } else {
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

// Individual message component
function Message({ message, isStreaming }) {
  const [showThinking, setShowThinking] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="message message-user">
        <div className="message-content">{message.content}</div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="message message-assistant">
      {message.thinking && (
        <div className="thinking-wrapper">
          <button
            className="thinking-toggle"
            onClick={() => setShowThinking(!showThinking)}
          >
            <span>{showThinking ? '▼' : '▶'}</span> Reasoning
          </button>
          {showThinking && (
            <div className="thinking-content">{message.thinking}</div>
          )}
        </div>
      )}
      <div className="message-content">
        {message.content || (isStreaming && <span className="streaming-cursor">▊</span>)}
      </div>
      {isStreaming && <span className="streaming-indicator"></span>}
    </div>
  );
}

export function ChatInterface() {
  const {
    model,
    loading,
    error,
    loadProgress,
    startConversation,
    chat,
    clearConversation,
    getStats,
    reset
  } = useQwenModel();

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [maxTokens, setMaxTokens] = useState(500);
  const [generating, setGenerating] = useState(false);
  const [enableThinking, setEnableThinking] = useState(true);
  const [stats, setStats] = useState({ messages: 0, cachedTokens: 0, tokensPerSec: 0 });
  const [conversationStarted, setConversationStarted] = useState(false);

  const abortControllerRef = useRef(null);
  const chatThreadRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll chat thread
  useEffect(() => {
    if (chatThreadRef.current) {
      chatThreadRef.current.scrollTop = chatThreadRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input when ready
  useEffect(() => {
    if (!loading && !generating && inputRef.current) {
      inputRef.current.focus();
    }
  }, [loading, generating]);

  const handleSend = async () => {
    if (!model || generating || !inputValue.trim()) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setGenerating(true);

    // Start conversation if needed
    if (!conversationStarted) {
      startConversation(null, enableThinking);
      setConversationStarted(true);
    }

    // Add user message to UI
    const userMsg = { role: 'user', content: userMessage };
    setMessages(prev => [...prev, userMsg]);

    // Add placeholder for assistant response
    const assistantMsgId = Date.now();
    setMessages(prev => [...prev, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      thinking: '',
      isStreaming: true
    }]);

    abortControllerRef.current = new AbortController();
    const startTime = Date.now();
    let tokenCount = 0;
    let fullResponse = '';

    try {
      await chat(userMessage, {
        maxTokens,
        enableThinking,
        temperature: 0.6,
        topP: 0.9,
        repeatPenalty: 1.1,
        repeatLastN: 64,
        seed: Date.now(),
        onToken: (token, count) => {
          fullResponse += token;
          tokenCount = count;

          const { thinking, response } = parseResponse(fullResponse, enableThinking);

          // Update the assistant message
          setMessages(prev => prev.map(msg =>
            msg.id === assistantMsgId
              ? { ...msg, content: response, thinking, isStreaming: true }
              : msg
          ));

          // Update stats every 10 tokens
          if (count % 10 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const modelStats = getStats();
            setStats({
              ...modelStats,
              tokensPerSec: (count / elapsed).toFixed(1)
            });
          }
        },
        signal: abortControllerRef.current.signal
      });

      // Final update
      const { thinking, response } = parseResponse(fullResponse, enableThinking);
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId
          ? { ...msg, content: response, thinking, isStreaming: false }
          : msg
      ));

      // Final stats
      const elapsed = (Date.now() - startTime) / 1000;
      const modelStats = getStats();
      setStats({
        ...modelStats,
        tokensPerSec: (tokenCount / elapsed).toFixed(1)
      });

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Generation error:', err);
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMsgId
            ? { ...msg, content: `Error: ${err.message}`, isStreaming: false }
            : msg
        ));
      }
    } finally {
      setGenerating(false);
      abortControllerRef.current = null;
      inputRef.current?.focus();
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleNewChat = () => {
    clearConversation();
    setMessages([]);
    setConversationStarted(false);
    setStats({ messages: 0, cachedTokens: 0, tokensPerSec: 0 });
    // Restart conversation with current thinking setting
    startConversation(null, enableThinking);
    setConversationStarted(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
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
        <h1>
          <span className="brand-mark" aria-hidden="true">Q</span>
          <span className="brand-name">Qwen3 <span className="brand-accent">Chat</span></span>
        </h1>
        <p>Running locally with WebAssembly + SIMD</p>
      </header>

      <div className="chat-main">
        {/* Status Bar */}
        <div className="status-bar">
          <span className="status-ready">Ready</span>
          <span className="cache-info">
            {stats.messages} messages | {stats.cachedTokens} tokens cached
            {stats.tokensPerSec > 0 && ` | ${stats.tokensPerSec} tok/s`}
          </span>
        </div>

        {/* Chat Thread */}
        <div className="chat-thread" ref={chatThreadRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <h3>Start a conversation</h3>
              <p>Type a message below to begin chatting with the model.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <Message
                key={msg.id || idx}
                message={msg}
                isStreaming={msg.isStreaming}
              />
            ))
          )}
        </div>

        {/* Input Area */}
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
              disabled={generating}
              rows={1}
            />
            <div className="input-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={enableThinking}
                  onChange={(e) => setEnableThinking(e.target.checked)}
                  disabled={generating}
                />
                Thinking mode
              </label>
              <label className="number-label">
                Max tokens:
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 500)}
                  min="1"
                  max="2000"
                  disabled={generating}
                />
              </label>
            </div>
          </div>
          <div className="button-group">
            <button
              className="btn-primary"
              onClick={handleSend}
              disabled={generating || !inputValue.trim()}
            >
              Send
            </button>
            {generating && (
              <button className="btn-stop" onClick={handleStop}>
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Footer Tools */}
        <div className="footer-tools">
          <button className="btn-secondary" onClick={handleNewChat}>
            New Chat
          </button>
        </div>
      </div>

      <footer className="chat-footer">
        <p>
          Model: Qwen3-0.6B Q4_K • Framework: Candle • Built with Rust + WASM
        </p>
      </footer>
    </div>
  );
}