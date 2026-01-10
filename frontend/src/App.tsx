import React, { useState, useCallback, useEffect } from 'react';
import './App.css';

interface TokenResult {
  token: string;
  probability: number;
  position: number;
}

interface InferenceResponse {
  tokens: TokenResult[];
  inference_time_ms: number;
  model_type: string;
  text: string;
}

// In production (via nginx): use relative URL (empty string)
// In local dev: set VITE_API_URL=http://localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_URL ?? '';

function App() {
  const [text, setText] = useState<string>('');
  const [results, setResults] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (inputText: string) => {
    if (!inputText.trim()) {
      setResults(null);
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: InferenceResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced predict function
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      predict(text);
    }, 300); // 300ms delay

    return () => clearTimeout(timeoutId);
  }, [text, predict]);

  const getTokenStyle = (probability: number, token: string) => {
    // Skip special tokens
    if (token.startsWith('<') || token.startsWith('[')) {
      return {
        backgroundColor: '#f5f5f5',
        color: '#666',
        padding: '2px 4px',
        margin: '1px',
        borderRadius: '3px',
        fontSize: '0.85em',
      };
    }

    // Color based on probability
    const intensity = Math.min(probability * 2, 1); // Cap at 1 for very high probabilities
    const red = Math.floor(255 * intensity);
    const green = Math.floor(255 * (1 - intensity));
    
    return {
      backgroundColor: `rgb(${red}, ${green}, 100)`,
      color: intensity > 0.5 ? 'white' : 'black',
      padding: '2px 4px',
      margin: '1px',
      borderRadius: '3px',
      fontWeight: intensity > 0.3 ? 'bold' : 'normal',
      fontSize: '0.9em',
    };
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Token Classification Demo</h1>
        <p>Type text to see token-level spell correction probabilities</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <label htmlFor="text-input">
            Input Text:
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your text here..."
            rows={4}
            className="text-input"
          />
          
          <div className="example-badges">
            <span className="badge-label">Quick Examples:</span>
            <button 
              className="example-badge typo"
              onClick={() => setText("This is a testt with some speeling errors and wrng words.")}
            >
              🔤 Typo
            </button>
            <button 
              className="example-badge grammar"
              onClick={() => setText("The childrens books was laying on the table, it's cover were torn.")}
            >
              📝 Grammar
            </button>
            <button 
              className="example-badge substitution"
              onClick={() => setText("I would advice you to loose some weight and accept there congratulations.")}
            >
              🔄 Real Word Substitution
            </button>
            <button 
              className="example-badge proper-noun"
              onClick={() => setText("I visited Parise, Californa and New Yourk last summr with Shakespear's novels.")}
            >
              🏛️ Proper Nouns
            </button>
            <button 
              className="example-badge long"
              onClick={() => setText("The advancement of artificial intelligence and machine learning technologies has revolutionized numerous industries across the globe. From healthcare and finance to transportation and entertainment, these cutting-edge innovations have transformed the way we approach complex problems and make data-driven decisions. In the medical field, AI algorithms can analyze vast amounts of patient data to assist doctors in diagnosing diseases more accurately and efficiently. Financial institutions leverage machine learning models to detect fraudulent transactions and assess credit risks with unprecedented precision. The automotive industry has embraced autonomous vehicle technology, promising safer roads and reduced traffic congestion. Meanwhile, streaming platforms use sophisticated recommendation systems to personalize content for millions of users worldwide. As we continue to push the bounderies of what's possible with artificial intelligence, we must also consider the ethical implications and ensure that these powerful tools are developed and deployed responsibly. The future holds immense potential for AI to address some of humanity's greatest challenges, from climate change mitigation to space exploration, but it requires careful consideration of privacy, security, and fairness to create a beneficial impact for all members of society.")}
            >
              📄 Long Text
            </button>
          </div>
        </div>

        {loading && (
          <div className="loading">
            Analyzing tokens...
          </div>
        )}

        {error && (
          <div className="error">
            Error: {error}
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="stats">
              <span>Model: {results.model_type.toUpperCase()}</span>
              <span>Inference: {results.inference_time_ms.toFixed(1)}ms</span>
              <span>Tokens: {results.tokens.length}</span>
            </div>

            <div className="tokens-container">
              <h3>Token Classification Results:</h3>
              <div className="tokens">
                {results.tokens.map((token, index) => (
                  <span
                    key={index}
                    style={getTokenStyle(token.probability, token.token)}
                    title={`Token: ${token.token}\nProbability: ${(token.probability * 100).toFixed(1)}%`}
                  >
                    {token.token.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n')}
                  </span>
                ))}
              </div>
            </div>

            <div className="token-details">
              <h4>High Probability Tokens (gt 10%):</h4>
              <div className="token-list">
                {results.tokens
                  .filter(t => t.probability > 0.1)
                  .sort((a, b) => b.probability - a.probability)
                  .slice(0, 10)
                  .map((token, index) => (
                    <div key={index} className="token-detail">
                      <span className="token-text">
                        "{token.token.replace(/Ġ/g, ' ')}"
                      </span>
                      <span className="token-prob">
                        {(token.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
