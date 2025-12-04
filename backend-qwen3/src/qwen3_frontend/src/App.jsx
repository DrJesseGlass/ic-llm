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
        config: [],  // None - uses defaults
      });
      setResponse(result);
    } catch (err) {
      setResponse({ success: false, error: err.message });
    }
    setLoading(false);
  }

  return (
    <main style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Qwen3 on IC</h1>

      <button onClick={checkModel} style={{ marginBottom: '1rem' }}>
        Check Model Status
      </button>
      {modelLoaded !== null && (
        <p>Model loaded: {modelLoaded ? '✅' : '❌'}</p>
      )}

      <form onSubmit={handleGenerate}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter prompt..."
          rows={4}
          style={{ width: '100%', marginBottom: '1rem' }}
        />
        <button type="submit" disabled={loading || !prompt}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </form>

      {response && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#f5f5f5' }}>
          {response.success ? (
            <>
              <p><strong>Output:</strong> {response.generated_text}</p>
              <p><small>Tokens: {response.tokens_generated} | Instructions: {Number(response.instructions_used).toLocaleString()}</small></p>
            </>
          ) : (
            <p style={{ color: 'red' }}>Error: {response.error}</p>
          )}
        </div>
      )}
    </main>
  );
}

export default App;