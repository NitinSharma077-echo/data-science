const { useEffect, useMemo, useState } = React;

const samples = [
  "The delivery was late, but customer support fixed everything quickly.",
  "I love the product quality and the app feels fast.",
  "This is terrible and awful. I want a refund.",
  "Not bad at all, the new update is actually useful.",
];

function Icon({ name }) {
  return (
    <span className="icon" aria-hidden="true">
      <i data-lucide={name}></i>
    </span>
  );
}

function Metric({ label, value, tone, icon }) {
  return (
    <section className={`metric ${tone || ""}`}>
      <Icon name={icon} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function SentimentBadge({ sentiment }) {
  const tone = (sentiment || "Neutral").toLowerCase();
  return <span className={`sentiment-badge ${tone}`}>{sentiment || "Neutral"}</span>;
}

function ProbabilityBar({ probabilities }) {
  const negative = probabilities?.negative || 0;
  const positive = probabilities?.positive || 0;
  return (
    <div className="probability-wrap">
      <div className="probability-labels">
        <span>Negative {negative.toFixed(1)}%</span>
        <span>Positive {positive.toFixed(1)}%</span>
      </div>
      <div className="probability-bar" aria-hidden="true">
        <span className="negative" style={{ width: `${negative}%` }}></span>
        <span className="positive" style={{ width: `${positive}%` }}></span>
      </div>
    </div>
  );
}

function ResultPanel({ result }) {
  if (!result) {
    return (
      <section className="panel result-panel empty-state">
        <Icon name="radar" />
        <h2>Live analysis</h2>
        <p>Submit a review to see sentiment, confidence, topics, and intent.</p>
      </section>
    );
  }

  return (
    <section className="panel result-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Result</span>
          <h2>{result.emotion}</h2>
        </div>
        <SentimentBadge sentiment={result.sentiment} />
      </div>

      <div className="confidence-ring" style={{ "--value": `${result.confidence}%` }}>
        <strong>{result.confidence.toFixed(1)}%</strong>
        <span>confidence</span>
      </div>

      <ProbabilityBar probabilities={result.probabilities} />

      <div className="detail-grid">
        <div>
          <span>Intent</span>
          <strong>{result.intent}</strong>
        </div>
        <div>
          <span>Latency</span>
          <strong>{result.latency_ms} ms</strong>
        </div>
      </div>

      <div className="topic-row">
        {result.topics.map((topic) => (
          <span key={topic}>{topic}</span>
        ))}
      </div>
    </section>
  );
}

function HistoryList({ history }) {
  if (!history.length) {
    return <p className="muted">No saved analyses yet.</p>;
  }

  return (
    <div className="history-list">
      {history.map((item) => (
        <article key={item.id} className="history-item">
          <div>
            <SentimentBadge sentiment={item.sentiment} />
            <span className="history-confidence">{item.confidence.toFixed(1)}%</span>
          </div>
          <p>{item.text}</p>
        </article>
      ))}
    </div>
  );
}

function App() {
  const [text, setText] = useState(samples[0]);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => text.trim().length > 0 && !loading, [text, loading]);

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    return payload;
  }

  async function refreshDashboard() {
    const [statsPayload, historyPayload] = await Promise.all([
      fetchJson("/api/stats"),
      fetchJson("/api/history"),
    ]);
    setStats(statsPayload);
    setHistory(historyPayload);
  }

  async function analyze(event) {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError("");
    try {
      const payload = await fetchJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setResult(payload);
      await refreshDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function clearHistory() {
    await fetchJson("/api/clear-history", { method: "POST" });
    await refreshDashboard();
  }

  useEffect(() => {
    refreshDashboard().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }, [result, stats, history, loading, error]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">NLP Flask App</span>
          <h1>Sentiment Studio</h1>
        </div>
        <div className="status-pill">
          <Icon name="activity" />
          Model connected
        </div>
      </header>

      <section className="metrics-row">
        <Metric label="Total" value={stats?.total ?? 0} icon="database" />
        <Metric label="Positive" value={`${stats?.positive_pct ?? 0}%`} tone="positive" icon="thumbs-up" />
        <Metric label="Negative" value={`${stats?.negative_pct ?? 0}%`} tone="negative" icon="thumbs-down" />
        <Metric label="NPS" value={stats?.nps_score ?? 0} tone="neutral" icon="gauge" />
      </section>

      <section className="workspace">
        <form className="panel input-panel" onSubmit={analyze}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Input</span>
              <h2>Review text</h2>
            </div>
            <button className="icon-button" type="button" title="Clear text" onClick={() => setText("")}>
              <Icon name="eraser" />
            </button>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength="5000"
            placeholder="Paste customer feedback, tweets, reviews, or support comments..."
          />

          <div className="sample-row" aria-label="Sample text">
            {samples.map((sample, index) => (
              <button key={sample} type="button" onClick={() => setText(sample)}>
                Sample {index + 1}
              </button>
            ))}
          </div>

          {error ? <div className="error-box">{error}</div> : null}

          <div className="form-actions">
            <span>{text.length}/5000</span>
            <button className="primary-button" type="submit" disabled={!canSubmit}>
              <Icon name={loading ? "loader-circle" : "sparkles"} />
              {loading ? "Analyzing" : "Analyze"}
            </button>
          </div>
        </form>

        <ResultPanel result={result} />

        <aside className="panel history-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Recent</span>
              <h2>History</h2>
            </div>
            <button className="icon-button" type="button" title="Clear history" onClick={clearHistory}>
              <Icon name="trash-2" />
            </button>
          </div>
          <HistoryList history={history} />
        </aside>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
