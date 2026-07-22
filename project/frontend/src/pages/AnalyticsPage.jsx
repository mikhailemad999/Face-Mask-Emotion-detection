import React, { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const EMOTION_COLORS = {
  angry:'#FF3366', disgust:'#FF6B35', fear:'#FFB700',
  happy:'#00FFB3', neutral:'#94A3B8', sad:'#7B61FF', surprise:'#E040FB',
}
const MASK_COLORS = { with_mask:'#00FFB3', without_mask:'#FF3366' }

/**
 * AnalyticsPage — Detailed analytics page showing historical prediction distribution charts and paginated logs.
 *
 * @returns {JSX.Element} Rendered analytics dashboard page component.
 */
export default function AnalyticsPage() {
  const [history, setHistory] = useState([])
  const [page,    setPage]    = useState(1)
  const [total,   setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetch(`${API_BASE}/detect/history/?page=${page}&page_size=20`)
      .then(r => r.json())
      .then(d => { setHistory(d.results || []); setTotal(d.count || 0) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page])

  // Compute distributions from loaded history
  const emotionCounts = history.reduce((acc, r) => {
    if (r.emotion_result) acc[r.emotion_result] = (acc[r.emotion_result] || 0) + 1
    return acc
  }, {})
  const maskCounts = history.reduce((acc, r) => {
    if (r.mask_result) acc[r.mask_result] = (acc[r.mask_result] || 0) + 1
    return acc
  }, {})

  /**
   * BarChart — Horizontal distribution bar chart visualization component.
   * @param {Object} props - Component properties.
   * @param {Object.<string, number>} props.data - Map of labels to count values.
   * @param {Object.<string, string>} props.colorMap - Map of labels to color strings.
   * @param {string} props.title - Chart section title.
   */
  const BarChart = ({ data, colorMap, title }) => {
    const max = Math.max(...Object.values(data), 1)
    return (
      <div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
          letterSpacing: '0.12em', textTransform: 'uppercase',
          color: 'var(--text-muted)', marginBottom: '16px' }}>
          {title}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {Object.entries(data).sort(([,a],[,b]) => b-a).map(([label, count]) => (
            <div key={label}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                color: 'var(--text-secondary)', marginBottom: '4px' }}>
                <span style={{ textTransform: 'capitalize' }}>{label.replace('_', ' ')}</span>
                <span style={{ color: colorMap[label] || 'var(--accent)' }}>{count}</span>
              </div>
              <div style={{ height: '8px', background: 'var(--bg-panel)',
                borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${(count / max) * 100}%`,
                  background: colorMap[label] || 'var(--accent)',
                  borderRadius: '4px', transition: 'width 600ms ease',
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const EMOTION_EMOJIS = {
    angry:'😠', disgust:'🤢', fear:'😨', happy:'😊',
    neutral:'😐', sad:'😢', surprise:'😲',
  }

  return (
    <div className="main">
      <div className="container" style={{ paddingBottom: '48px' }}>
        <div className="page-header animate-fade-in-up">
          <div className="page-header__eyebrow">// analytics.report</div>
          <h1 className="page-header__title">Detection <span>Analytics</span></h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '520px' }}>
            Historical analysis of mask compliance and emotion distribution across {total} detections.
          </p>
        </div>

        {/* Charts row */}
        <div className="grid grid--2 mb-8 animate-fade-in-up" style={{ marginBottom: '32px' }}>
          <div className="card">
            <BarChart data={emotionCounts} colorMap={EMOTION_COLORS} title="EMOTION_DISTRIBUTION" />
          </div>
          <div className="card">
            <BarChart data={maskCounts} colorMap={MASK_COLORS} title="MASK_DISTRIBUTION" />
          </div>
        </div>

        {/* Full history table */}
        <div className="card animate-fade-in-up">
          <div style={{ display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.12em', textTransform: 'uppercase',
                color: 'var(--text-muted)', marginBottom: '4px' }}>
                DETECTION_LOG
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                All Detections ({total})
              </h3>
            </div>
          </div>

          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[...Array(8)].map((_, i) => (
                <div key={i} className="skeleton" style={{ height: '44px' }} />
              ))}
            </div>
          ) : (
            <>
              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>#</th><th>TIMESTAMP</th><th>SRC</th>
                      <th>FACES</th><th>MASK</th><th>CONFIDENCE</th>
                      <th>EMOTION</th><th>CONFIDENCE</th><th>MS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map(log => (
                      <tr key={log.id}>
                        <td>{log.id}</td>
                        <td style={{ fontSize: '0.7rem' }}>
                          {new Date(log.timestamp).toLocaleString()}
                        </td>
                        <td>{log.source}</td>
                        <td style={{ color: 'var(--accent)' }}>{log.faces_detected}</td>
                        <td>
                          <span className={`detection-badge ${
                            log.mask_result === 'with_mask'
                              ? 'detection-badge--mask' : 'detection-badge--no-mask'
                          }`} style={{ fontSize: '0.6rem' }}>
                            {log.mask_result || '—'}
                          </span>
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                          color: 'var(--text-muted)' }}>
                          {log.mask_confidence ? `${(log.mask_confidence * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                            {EMOTION_EMOJIS[log.emotion_result] || ''} {log.emotion_result || '—'}
                          </span>
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                          color: 'var(--text-muted)' }}>
                          {log.emotion_confidence ? `${(log.emotion_confidence * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                          color: 'var(--text-muted)' }}>
                          {log.processing_time_ms?.toFixed(1) || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div style={{ display: 'flex', justifyContent: 'center', gap: '8px',
                marginTop: '20px', flexWrap: 'wrap' }}>
                <button
                  className="btn btn--ghost"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  style={{ minWidth: '44px' }}
                >
                  ← Prev
                </button>
                <span style={{ display: 'flex', alignItems: 'center',
                  fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
                  color: 'var(--text-muted)', padding: '0 12px' }}>
                  Page {page} of {Math.ceil(total / 20)}
                </span>
                <button
                  className="btn btn--ghost"
                  onClick={() => setPage(p => p + 1)}
                  disabled={page >= Math.ceil(total / 20)}
                  style={{ minWidth: '44px' }}
                >
                  Next →
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
