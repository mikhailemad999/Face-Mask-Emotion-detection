import React, { useEffect, useState } from 'react'
import StatsCard from '../components/StatsCard'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function DashboardPage() {
  const [history, setHistory] = useState([])
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [histRes] = await Promise.all([
          fetch(`${API_BASE}/detect/history/?page=1&page_size=10`),
        ])
        const histData = await histRes.json()
        setHistory(histData.results || [])
        // Compute quick stats from history
        const results = histData.results || []
        const withMask = results.filter(r => r.mask_result === 'with_mask').length
        const emotions  = results.map(r => r.emotion_result).filter(Boolean)
        const emotionFreq = emotions.reduce((acc, e) => {
          acc[e] = (acc[e] || 0) + 1; return acc
        }, {})
        const topEmotion = Object.entries(emotionFreq).sort(([,a],[,b]) => b-a)[0]
        setStats({
          total:       histData.count || 0,
          withMask:    withMask,
          noMask:      results.length - withMask,
          topEmotion:  topEmotion?.[0] || 'N/A',
          avgTime:     results.length
            ? (results.reduce((a, r) => a + (r.processing_time_ms||0), 0) / results.length).toFixed(1)
            : 0,
        })
      } catch (e) {
        console.error('Failed to fetch dashboard data', e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
    const interval = setInterval(fetchData, 15000)  // refresh every 15s
    return () => clearInterval(interval)
  }, [])

  const EMOTION_EMOJIS = {
    angry:'😠', disgust:'🤢', fear:'😨', happy:'😊',
    neutral:'😐', sad:'😢', surprise:'😲',
  }

  return (
    <div className="main">
      <div className="container" style={{ paddingBottom: '48px' }}>
        {/* Header */}
        <div className="page-header animate-fade-in-up">
          <div className="page-header__eyebrow">// system_dashboard</div>
          <h1 className="page-header__title">
            Real-time <span>Intelligence</span> Dashboard
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '560px' }}>
            Live monitoring of face mask compliance and emotion analytics.
            All predictions logged to SQL Server + MongoDB.
          </p>
        </div>

        {/* Stats row */}
        <div className="grid grid--4 animate-fade-in-up mb-8" style={{ marginBottom: '32px' }}>
          <StatsCard
            label="TOTAL_DETECTIONS"
            value={loading ? '…' : stats?.total ?? 0}
            sub="All time predictions"
            color="accent"
            icon="🎯"
          />
          <StatsCard
            label="MASK_COMPLIANT"
            value={loading ? '…' : stats?.withMask ?? 0}
            sub="With mask detected"
            color="accent"
            icon="😷"
          />
          <StatsCard
            label="NO_MASK_ALERTS"
            value={loading ? '…' : stats?.noMask ?? 0}
            sub="Without mask"
            color="red"
            icon="⚠"
          />
          <StatsCard
            label="TOP_EMOTION"
            value={loading ? '…' : (EMOTION_EMOJIS[stats?.topEmotion] || '') + ' ' + (stats?.topEmotion || 'N/A')}
            sub={`Avg inference: ${stats?.avgTime || 0}ms`}
            color="purple"
            icon="🧠"
          />
        </div>

        {/* Recent detections table */}
        <div className="card animate-fade-in-up">
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '20px',
          }}>
            <div>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.12em', textTransform: 'uppercase',
                color: 'var(--text-muted)', marginBottom: '4px',
              }}>
                DETECTION_LOG
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Recent Detections</h3>
            </div>
            <a href="/analytics"
              className="btn btn--ghost"
              style={{ fontSize: '0.8rem', textDecoration: 'none' }}>
              View All →
            </a>
          </div>

          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[...Array(5)].map((_, i) => (
                <div key={i} className="skeleton"
                  style={{ height: '48px', animationDelay: `${i * 0.1}s` }} />
              ))}
            </div>
          ) : history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '48px', color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              No detections yet. Upload an image to get started.
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>#ID</th>
                    <th>TIMESTAMP</th>
                    <th>SOURCE</th>
                    <th>FACES</th>
                    <th>MASK</th>
                    <th>EMOTION</th>
                    <th>TIME_MS</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((log) => (
                    <tr key={log.id}>
                      <td>{log.id}</td>
                      <td style={{ fontSize: '0.75rem' }}>
                        {new Date(log.timestamp).toLocaleString()}
                      </td>
                      <td>{log.source}</td>
                      <td style={{ color: 'var(--accent)' }}>{log.faces_detected}</td>
                      <td>
                        <span className={`detection-badge ${
                          log.mask_result === 'with_mask'
                            ? 'detection-badge--mask'
                            : 'detection-badge--no-mask'
                        }`} style={{ fontSize: '0.65rem' }}>
                          {log.mask_result || '—'}
                        </span>
                      </td>
                      <td>
                        <span className="detection-badge detection-badge--emotion"
                          style={{ fontSize: '0.65rem' }}>
                          {EMOTION_EMOJIS[log.emotion_result] || ''} {log.emotion_result || '—'}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-muted)' }}>
                        {log.processing_time_ms?.toFixed(1) || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
