import { useMemo, useState } from 'react'

const presets = [
  'What is lung cancer?',
  'What are the main treatment options for breast cancer?',
  'What is immunotherapy used for in cancer treatment?',
  'What are the characteristics of a malignant tumor?',
  'What is the difference between benign and malignant tumors?',
  'Do I have cancer?',
]

const navItems = [
  { label: 'Console', hint: 'Live ask stream' },
  { label: 'Signals', hint: 'Confidence + method' },
  { label: 'Chunks', hint: 'Supporting evidence' },
  { label: 'Trace', hint: 'Recent prompts' },
]

const statCards = [
  { label: 'Mode', value: 'RAG + BioBERT + LLaMA' },
  { label: 'Tone', value: 'Clinical, cautious, grounded' },
  { label: 'Focus', value: 'Cancer-aware document QA' },
]

const systemBadges = [
  { label: 'API', value: 'Connected' },
  { label: 'Retrieval', value: 'FAISS online' },
  { label: 'Safety', value: 'Guard rails on' },
]

const defaultState = {
  answer: '',
  confidence: 0,
  method: '',
  used_chunks: [],
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

function confidenceClass(value) {
  if (value >= 0.75) return 'high'
  if (value >= 0.4) return 'mid'
  return 'low'
}

function chunkLabel(chunk) {
  if (typeof chunk !== 'string') return chunk?.id || 'chunk'
  return chunk
}

function Bubble({ role, title, body, meta }) {
  return (
    <article className={`bubble ${role}`}>
      <div className="bubble-top">
        <span className="bubble-role">{title}</span>
        {meta ? <span className="bubble-meta">{meta}</span> : null}
      </div>
      <p>{body}</p>
    </article>
  )
}

export default function App() {
  const [question, setQuestion] = useState(presets[1])
  const [result, setResult] = useState(defaultState)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [transcript, setTranscript] = useState([])

  const confidence = useMemo(() => Math.round((result.confidence || 0) * 100), [result.confidence])
  const confidenceAngle = `${confidence * 3.6}deg`

  async function submitQuestion(nextQuestion = question) {
    const trimmed = nextQuestion.trim()
    if (!trimmed || loading) return

    const userTurn = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      role: 'user',
      title: 'Operator',
      body: trimmed,
    }

    setLoading(true)
    setError('')
    setTranscript((prev) => [userTurn, ...prev])

    try {
      const response = await fetch(`${API_BASE || ''}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: trimmed }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setTranscript((prev) => [
        {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          role: 'assistant',
          title: 'OncoBot',
          body: data.answer,
          meta: `${data.method || 'unlabelled'} · confidence ${Math.round((data.confidence || 0) * 100)}%`,
          chunks: data.used_chunks || [],
        },
        ...prev,
      ])
    } catch (err) {
      setError(err.message || 'Failed to reach the API.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <div className="orb orb-c" />
      <div className="grid-noise" />

      <aside className="left-rail panel">
        <div className="rail-brand">
          <div className="rail-mark">◎</div>
          <div>
            <span className="panel-kicker">OncoBot Atlas</span>
            <strong>Research cockpit</strong>
          </div>
        </div>

        <div className="rail-pulse">
          <div className="pulse-ring" />
          <div>
            <span>Live system</span>
            <strong>Retrieval + generation</strong>
          </div>
        </div>

        <nav className="rail-nav">
          {navItems.map((item) => (
            <button key={item.label} className="rail-nav-item">
              <span>{item.label}</span>
              <small>{item.hint}</small>
            </button>
          ))}
        </nav>

        <div className="rail-badges">
          {systemBadges.map((badge) => (
            <div key={badge.label} className="rail-badge">
              <span>{badge.label}</span>
              <strong>{badge.value}</strong>
            </div>
          ))}
        </div>

        <div className="rail-note">
          <span className="panel-kicker">Safety note</span>
          <p>
            Medical guidance is not a diagnosis. Diagnostic questions are refused by design.
          </p>
        </div>
      </aside>

      <main className="main-stage">
        <section className="hero hero-expanded panel">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="dot" />
              Cancer QA Control Room
            </div>
            <h1>OncoBot Atlas</h1>
            <p>
              A dramatic retrieval console for cancer-awareness questions with source-grounded answers,
              confidence telemetry, and chunk-level provenance.
            </p>

            <div className="status-strip status-strip-inline">
              {statCards.map((card) => (
                <div key={card.label} className="status-card">
                  <span>{card.label}</span>
                  <strong>{card.value}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className={`confidence-orb ${confidenceClass(result.confidence || 0)}`}>
            <div className="confidence-ring" style={{ '--angle': confidenceAngle }} />
            <div className="confidence-core">
              <span>{confidence}%</span>
              <small>confidence</small>
            </div>
          </div>
        </section>

        <section className="composer panel">
          <div className="panel-header">
            <div>
              <span className="panel-kicker">Query Reactor</span>
              <h2>Ask the system</h2>
            </div>
            <button className="ghost-btn" onClick={() => submitQuestion()}>
              Run
            </button>
          </div>

          <div className="composer-grid">
            <div className="composer-main">
              <textarea
                className="question-box"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a cancer-aware question..."
                rows={6}
              />

              <div className="preset-row">
                {presets.map((item) => (
                  <button
                    key={item}
                    className={`preset-chip ${question === item ? 'active' : ''}`}
                    onClick={() => setQuestion(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>

              <div className="action-row">
                <button className="primary-btn" onClick={() => submitQuestion()}>
                  {loading ? 'Analyzing...' : 'Ask OncoBot'}
                </button>
                <button className="secondary-btn" onClick={() => setQuestion('')}>
                  Clear
                </button>
              </div>
            </div>

            <div className="composer-side">
              <div className="side-card glass">
                <div className="section-title">Signal deck</div>
                <div className="signal-list">
                  <div>
                    <span>Method</span>
                    <strong>{result.method || 'Awaiting prompt'}</strong>
                  </div>
                  <div>
                    <span>Chunks used</span>
                    <strong>{result.used_chunks?.length || 0}</strong>
                  </div>
                  <div>
                    <span>Confidence</span>
                    <strong>{confidence}%</strong>
                  </div>
                </div>
              </div>

              <div className="side-card glass">
                <div className="section-title">Quick launch</div>
                <p className="side-copy">
                  The frontend streams one question at a time and keeps a transcript of every answer
                  for quick inspection.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <article className="panel answer-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Response chamber</span>
                <h2>System output</h2>
              </div>
              <div className="method-pill">{result.method || 'Awaiting prompt'}</div>
            </div>

            {error ? (
              <div className="alert">{error}</div>
            ) : loading ? (
              <div className="loading-state">
                <div className="loader-line short" />
                <div className="loader-line" />
                <div className="loader-line wide" />
              </div>
            ) : result.answer ? (
              <div className="answer-block">
                <p>{result.answer}</p>
              </div>
            ) : (
              <div className="empty-state">
                <h3>Ready for a question</h3>
                <p>
                  The answer panel will show the model response, confidence level, and the chunks used
                  to support the answer.
                </p>
              </div>
            )}

            <div className="chunk-section">
              <div className="section-title">Used chunks</div>
              <div className="chunk-list">
                {result.used_chunks?.length ? (
                  result.used_chunks.map((chunk, index) => (
                    <span key={`${chunkLabel(chunk)}-${index}`} className="chunk-pill">
                      {chunkLabel(chunk)}
                    </span>
                  ))
                ) : (
                  <span className="muted">No chunks yet</span>
                )}
              </div>
            </div>
          </article>

          <article className="panel transcript-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Transcript</span>
                <h2>Conversation feed</h2>
              </div>
            </div>

            <div className="transcript-stack">
              {transcript.length ? (
                transcript.map((entry) => (
                  <Bubble
                    key={entry.id}
                    role={entry.role}
                    title={entry.title}
                    body={entry.body}
                    meta={entry.meta}
                  />
                ))
              ) : (
                <div className="empty-history">
                  Prompt history will appear here after the first successful request.
                </div>
              )}
            </div>
          </article>
        </section>
      </main>

      <footer className="footer-note">
        This interface is for research and educational use only. It does not provide medical advice.
      </footer>
    </div>
  )
}
