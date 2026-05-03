(function () {
  const samples = [
    "The delivery was late, but customer support fixed everything quickly.",
    "I love the product quality and the app feels fast.",
    "This is terrible and awful. I want a refund.",
    "Not bad at all, the new update is actually useful.",
  ];

  const state = {
    text: samples[0],
    result: null,
    stats: null,
    history: [],
    loading: false,
    error: "",
  };

  const root = document.getElementById("root");

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function icon(name) {
    const icons = {
      activity: "OK",
      database: "DB",
      up: "+",
      down: "-",
      gauge: "%",
      erase: "X",
      analyze: "*",
      radar: "AI",
      trash: "Del",
    };
    return `<span class="icon text-icon" aria-hidden="true">${icons[name] || ""}</span>`;
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    return payload;
  }

  async function refreshDashboard() {
    const [stats, history] = await Promise.all([
      fetchJson("/api/stats"),
      fetchJson("/api/history"),
    ]);
    state.stats = stats;
    state.history = history;
  }

  function metric(label, value, tone, iconName) {
    return `
      <section class="metric ${tone || ""}">
        ${icon(iconName)}
        <div>
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      </section>
    `;
  }

  function badge(sentiment) {
    const label = sentiment || "Neutral";
    return `<span class="sentiment-badge ${label.toLowerCase()}">${escapeHtml(label)}</span>`;
  }

  function probabilityBar(result) {
    const negative = Number(result?.probabilities?.negative || 0);
    const positive = Number(result?.probabilities?.positive || 0);
    return `
      <div class="probability-wrap">
        <div class="probability-labels">
          <span>Negative ${negative.toFixed(1)}%</span>
          <span>Positive ${positive.toFixed(1)}%</span>
        </div>
        <div class="probability-bar" aria-hidden="true">
          <span class="negative" style="width: ${negative}%"></span>
          <span class="positive" style="width: ${positive}%"></span>
        </div>
      </div>
    `;
  }

  function resultPanel() {
    if (!state.result) {
      return `
        <section class="panel result-panel empty-state">
          ${icon("radar")}
          <h2>Live analysis</h2>
          <p>Submit a review to see sentiment, confidence, topics, and intent.</p>
        </section>
      `;
    }

    const result = state.result;
    const topics = result.topics.map((topic) => `<span>${escapeHtml(topic)}</span>`).join("");
    return `
      <section class="panel result-panel">
        <div class="panel-heading">
          <div>
            <span class="eyebrow">Result</span>
            <h2>${escapeHtml(result.emotion)}</h2>
          </div>
          ${badge(result.sentiment)}
        </div>
        <div class="confidence-ring" style="--value: ${result.confidence}%">
          <strong>${Number(result.confidence).toFixed(1)}%</strong>
          <span>confidence</span>
        </div>
        ${probabilityBar(result)}
        <div class="detail-grid">
          <div>
            <span>Intent</span>
            <strong>${escapeHtml(result.intent)}</strong>
          </div>
          <div>
            <span>Latency</span>
            <strong>${escapeHtml(result.latency_ms)} ms</strong>
          </div>
        </div>
        <div class="topic-row">${topics}</div>
      </section>
    `;
  }

  function historyPanel() {
    const items = state.history.length
      ? state.history
          .map(
            (item) => `
              <article class="history-item">
                <div>
                  ${badge(item.sentiment)}
                  <span class="history-confidence">${Number(item.confidence).toFixed(1)}%</span>
                </div>
                <p>${escapeHtml(item.text)}</p>
              </article>
            `
          )
          .join("")
      : '<p class="muted">No saved analyses yet.</p>';

    return `
      <aside class="panel history-panel">
        <div class="panel-heading">
          <div>
            <span class="eyebrow">Recent</span>
            <h2>History</h2>
          </div>
          <button class="icon-button" type="button" title="Clear history" data-action="clear-history">
            ${icon("trash")}
          </button>
        </div>
        <div class="history-list">${items}</div>
      </aside>
    `;
  }

  function render() {
    const stats = state.stats || {};
    const sampleButtons = samples
      .map(
        (sample, index) =>
          `<button type="button" data-sample="${index}">Sample ${index + 1}</button>`
      )
      .join("");

    root.innerHTML = `
      <main class="app-shell">
        <header class="topbar">
          <div>
            <span class="eyebrow">NLP Flask App</span>
            <h1>Sentiment Studio</h1>
          </div>
          <div class="status-pill">${icon("activity")} Model connected</div>
        </header>

        <section class="metrics-row">
          ${metric("Total", stats.total ?? 0, "", "database")}
          ${metric("Positive", `${stats.positive_pct ?? 0}%`, "positive", "up")}
          ${metric("Negative", `${stats.negative_pct ?? 0}%`, "negative", "down")}
          ${metric("NPS", stats.nps_score ?? 0, "neutral", "gauge")}
        </section>

        <section class="workspace">
          <form class="panel input-panel" data-role="analyze-form">
            <div class="panel-heading">
              <div>
                <span class="eyebrow">Input</span>
                <h2>Review text</h2>
              </div>
              <button class="icon-button" type="button" title="Clear text" data-action="clear-text">
                ${icon("erase")}
              </button>
            </div>

            <textarea maxlength="5000" placeholder="Paste customer feedback, tweets, reviews, or support comments...">${escapeHtml(state.text)}</textarea>

            <div class="sample-row" aria-label="Sample text">${sampleButtons}</div>
            ${state.error ? `<div class="error-box">${escapeHtml(state.error)}</div>` : ""}

            <div class="form-actions">
              <span>${state.text.length}/5000</span>
              <button class="primary-button" type="submit" ${state.loading || !state.text.trim() ? "disabled" : ""}>
                ${icon("analyze")}
                ${state.loading ? "Analyzing" : "Analyze"}
              </button>
            </div>
          </form>

          ${resultPanel()}
          ${historyPanel()}
        </section>
      </main>
    `;
  }

  async function analyze() {
    if (!state.text.trim() || state.loading) return;
    state.loading = true;
    state.error = "";
    render();

    try {
      state.result = await fetchJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.text }),
      });
      await refreshDashboard();
    } catch (error) {
      state.error = error.message;
    } finally {
      state.loading = false;
      render();
    }
  }

  async function clearHistory() {
    await fetchJson("/api/clear-history", { method: "POST" });
    await refreshDashboard();
    render();
  }

  root.addEventListener("input", (event) => {
    if (event.target.matches("textarea")) {
      state.text = event.target.value;
      const counter = root.querySelector(".form-actions span");
      if (counter) counter.textContent = `${state.text.length}/5000`;
    }
  });

  root.addEventListener("submit", (event) => {
    if (event.target.matches('[data-role="analyze-form"]')) {
      event.preventDefault();
      analyze();
    }
  });

  root.addEventListener("click", (event) => {
    const sampleButton = event.target.closest("[data-sample]");
    if (sampleButton) {
      state.text = samples[Number(sampleButton.dataset.sample)];
      render();
      return;
    }

    if (event.target.closest('[data-action="clear-text"]')) {
      state.text = "";
      render();
      return;
    }

    if (event.target.closest('[data-action="clear-history"]')) {
      clearHistory().catch((error) => {
        state.error = error.message;
        render();
      });
    }
  });

  refreshDashboard()
    .catch((error) => {
      state.error = error.message;
    })
    .finally(render);
})();
