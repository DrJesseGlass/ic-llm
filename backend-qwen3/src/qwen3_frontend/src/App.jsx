import { useState } from 'react';
import { qwen3_backend } from 'declarations/qwen3_backend';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(null);

  async function checkModel() {
    const loaded = await qwen3_backend.is_model_loaded();
    setModelLoaded(loaded);
  }

  async function handleGenerate(e) {
    e.preventDefault();
    setLoading(true);
    setResponse(null);

    try {
      const result = await qwen3_backend.generate({
        prompt,
        config: [],
      });
      setResponse(result);
    } catch (err) {
      setResponse({ success: false, error: err.message });
    }
    setLoading(false);
  }

  return (
    <div className="container">
      <header>
        <h1>Qwen3 on IC</h1>
        <p className="subtitle">0.6B parameter LLM running on-chain</p>
      </header>

      <div className="status-bar">
        <button onClick={checkModel} className="btn-secondary">
          Check Model
        </button>
        {modelLoaded !== null && (
          <span className={`status ${modelLoaded ? 'online' : 'offline'}`}>
            {modelLoaded ? '● Model Ready' : '○ Model Not Loaded'}
          </span>
        )}
      </div>

      <form onSubmit={handleGenerate}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          rows={4}
        />
        <button type="submit" className="btn-primary" disabled={loading || !prompt}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </form>

      {response && (
        <div className={`response ${response.success ? '' : 'error'}`}>
          {response.success ? (
            <>
              <div className="output">{response.generated_text}</div>
              <div className="meta">
                <span>Tokens: {Number(response.tokens_generated)}</span>
                <span>Instructions: {Number(response.instructions_used).toLocaleString()}</span>
              </div>
            </>
          ) : (
            <div className="error-msg">Error: {response.error}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;