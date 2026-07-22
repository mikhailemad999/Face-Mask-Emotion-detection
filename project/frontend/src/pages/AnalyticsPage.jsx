/**
 * @file AnalyticsPage.jsx
 * @description Executive Data Analytics & Statistics Dashboard for Face Mask & Emotion Detection system.
 */
import React, { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const EMOTION_EMOJIS = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  neutral: '😐',
  surprise: '😮',
  fear: '😱',
  disgust: '🤢',
}

const EMOTION_COLORS = {
  angry: '#FF3366',
  disgust: '#FF6B35',
  fear: '#FFB700',
  happy: '#00FFB3',
  neutral: '#94A3B8',
  sad: '#7B61FF',
  surprise: '#E040FB',
}

const MASK_COLORS = {
  with_mask: '#00FFB3',
  without_mask: '#FF3366',
}

const SOURCE_COLORS = {
  upload: '#00F0FF',
  batch: '#7B61FF',
  live: '#00FFB3',
}

export default function AnalyticsPage() {
  const [summary, setSummary]           = useState(null)
  const [history, setHistory]           = useState([])
  const [page, setPage]                 = useState(1)
  const [totalLogs, setTotalLogs]       = useState(0)
  const [loadingLogs, setLoadingLogs]   = useState(true)

  // Interactive filters
  const [sourceFilter, setSourceFilter]   = useState('')
  const [maskFilter, setMaskFilter]       = useState('')
  const [emotionFilter, setEmotionFilter] = useState('')
  const [searchQuery, setSearchQuery]     = useState('')

  // Fetch summary statistics from backend
  useEffect(() => {
    fetch(`${API_BASE}/analytics/summary/`)
      .then(r => r.json())
      .then(d => setSummary(d))
      .catch(() => {})
  }, [])

  // Fetch paginated history logs with filters
  useEffect(() => {
    setLoadingLogs(true)
    let url = `${API_BASE}/detect/history/?page=${page}&page_size=15`
    if (sourceFilter)  url += `&source=${sourceFilter}`
    if (maskFilter)    url += `&mask=${maskFilter}`
    if (emotionFilter) url += `&emotion=${emotionFilter}`

    fetch(url)
      .then(r => r.json())
      .then(d => {
        setHistory(d.results || [])
        setTotalLogs(d.count || 0)
      })
      .catch(() => {})
      .finally(() => setLoadingLogs(false))
  }, [page, sourceFilter, maskFilter, emotionFilter])

  /**
   * Export detection history logs to CSV file.
   */
  const handleExportCSV = () => {
    if (!history.length) return
    const headers = ['ID', 'Timestamp', 'Source', 'Faces Count', 'Mask Result', 'Mask Confidence', 'Emotion Result', 'Emotion Confidence', 'Processing Time (ms)']
    const rows = history.map(r => [
      r.id,
      `"${new Date(r.timestamp).toLocaleString()}"`,
      `"${r.source || 'N/A'}"`,
      r.faces_detected || 0,
      `"${r.mask_result || 'N/A'}"`,
      r.mask_confidence ? (r.mask_confidence * 100).toFixed(1) + '%' : 'N/A',
      `"${r.emotion_result || 'N/A'}"`,
      r.emotion_confidence ? (r.emotion_confidence * 100).toFixed(1) + '%' : 'N/A',
      r.processing_time_ms ? r.processing_time_ms.toFixed(1) : '0',
    ])

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', `FaceGuard_Analytics_Logs_${Date.now()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Client-side search filtering on loaded logs
  const filteredHistory = history.filter(r =>
    !searchQuery ||
    (r.image_name && r.image_name.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (r.emotion_result && r.emotion_result.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (r.mask_result && r.mask_result.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const emotionCounts = summary?.emotion_counts || {}
  const maskCounts    = summary?.mask_counts || {}
  const sourceCounts  = summary?.source_counts || {}
  const maxEmotion    = Math.max(...Object.values(emotionCounts), 1)
  const totalMaskFaces = (maskCounts.with_mask || 0) + (maskCounts.without_mask || 0)

  return (
    <div style={{ maxWidth: '1240px', margin: '0 auto', paddingBottom: '60px' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }} className="animate-fade-in-up">
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
          color: 'var(--cyan)', letterSpacing: '0.12em',
          textTransform: 'uppercase', marginBottom: '8px',
        }}>
          // SYSTEM_ANALYTICS_DASHBOARD
        </div>
        <h1 style={{ fontSize: '2.2rem', fontWeight: 800, margin: '0 0 8px' }}>
          Data Analytics & <span style={{ color: 'var(--cyan)' }}>Statistics Dashboard</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, maxWidth: '680px' }}>
          Real-time analytical metrics, facial sentiment distributions, mask compliance ratios, and historical database query logs.
        </p>
      </div>

      {/* Executive Key Performance Statistics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '16px', marginBottom: '32px' }}>
        {/* Total Predictions */}
        <div className="card">
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
            TOTAL_PREDICTIONS
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--cyan)' }}>
            {summary?.total_predictions || 0}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Logged across all sessions
          </div>
        </div>

        {/* Mask Compliance Rate */}
        <div className="card">
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
            MASK_COMPLIANCE_RATE
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: (summary?.mask_compliance_rate || 0) >= 70 ? 'var(--green)' : 'var(--red)' }}>
            {summary?.mask_compliance_rate !== undefined ? `${summary.mask_compliance_rate}%` : '—'}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {maskCounts.with_mask || 0} with mask / {maskCounts.without_mask || 0} without
          </div>
        </div>

        {/* Dominant Sentiment */}
        <div className="card">
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
            DOMINANT_SENTIMENT
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, textTransform: 'capitalize', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>{EMOTION_EMOJIS[summary?.dominant_emotion] || '😐'}</span>
            <span>{summary?.dominant_emotion || 'N/A'}</span>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Top facial emotion overall
          </div>
        </div>

        {/* Average Latency */}
        <div className="card">
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
            AVG_INFERENCE_LATENCY
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#7B61FF' }}>
            {summary?.avg_processing_ms ? `${summary.avg_processing_ms}ms` : '—'}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            PyTorch GPU execution speed
          </div>
        </div>

        {/* Training Dataset Benchmark */}
        <div className="card">
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
            DATASET_BENCHMARK
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#E040FB' }}>
            {summary?.dataset_statistics?.total_images?.toLocaleString() || '18,681'}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            11.4k Emotion + 7.3k Mask images
          </div>
        </div>
      </div>

      {/* Visual Distribution Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        {/* Emotion Distribution */}
        <div className="card">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: '0 0 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Facial Sentiment Statistics</span>
            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              EMOTION_DISTRIBUTION
            </span>
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Object.entries(emotionCounts).length > 0 ? (
              Object.entries(emotionCounts).sort(([,a], [,b]) => b - a).map(([emotion, count]) => {
                const totalEmotions = Object.values(emotionCounts).reduce((a, b) => a + b, 0)
                const pct = totalEmotions > 0 ? Math.round((count / totalEmotions) * 100) : 0
                return (
                  <div key={emotion}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px', textTransform: 'capitalize' }}>
                        <span>{EMOTION_EMOJIS[emotion]}</span>
                        <strong style={{ color: EMOTION_COLORS[emotion] || 'var(--cyan)' }}>{emotion}</strong>
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {count} ({pct}%)
                      </span>
                    </div>
                    <div style={{ height: '8px', background: 'var(--surface-darker)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${(count / maxEmotion) * 100}%`,
                          background: EMOTION_COLORS[emotion] || 'var(--cyan)',
                          transition: 'width 0.4s ease',
                        }}
                      />
                    </div>
                  </div>
                )
              })
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '16px 0' }}>
                No emotion prediction records yet.
              </div>
            )}
          </div>
        </div>

        {/* Mask Compliance & Source Distribution */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Mask Compliance Card */}
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: '0 0 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Mask Compliance Ratio</span>
              <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                MASK_STATS
              </span>
            </h3>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span>😷 With Mask</span>
                <strong style={{ color: 'var(--green)' }}>
                  {maskCounts.with_mask || 0} ({totalMaskFaces > 0 ? Math.round(((maskCounts.with_mask || 0) / totalMaskFaces) * 100) : 0}%)
                </strong>
              </div>
              <div style={{ height: '8px', background: 'var(--surface-darker)', borderRadius: '4px', overflow: 'hidden', marginBottom: '14px' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${totalMaskFaces > 0 ? ((maskCounts.with_mask || 0) / totalMaskFaces) * 100 : 0}%`,
                    background: 'var(--green)',
                    transition: 'width 0.4s ease',
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span>🔴 Without Mask</span>
                <strong style={{ color: 'var(--red)' }}>
                  {maskCounts.without_mask || 0} ({totalMaskFaces > 0 ? Math.round(((maskCounts.without_mask || 0) / totalMaskFaces) * 100) : 0}%)
                </strong>
              </div>
              <div style={{ height: '8px', background: 'var(--surface-darker)', borderRadius: '4px', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${totalMaskFaces > 0 ? ((maskCounts.without_mask || 0) / totalMaskFaces) * 100 : 0}%`,
                    background: 'var(--red)',
                    transition: 'width 0.4s ease',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Detection Source Breakdown */}
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: '0 0 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Detection Source Ratios</span>
              <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                SOURCE_RATIO
              </span>
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {Object.entries(sourceCounts).map(([src, count]) => {
                const totalSrc = Object.values(sourceCounts).reduce((a, b) => a + b, 0)
                const pct = totalSrc > 0 ? Math.round((count / totalSrc) * 100) : 0
                return (
                  <div key={src}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                      <span style={{ textTransform: 'capitalize', fontWeight: 600, color: SOURCE_COLORS[src] || 'var(--cyan)' }}>
                        {src === 'upload' ? '🖼 Upload Image' : src === 'batch' ? '📁 Batch Folder' : '📹 Live Stream'}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {count} ({pct}%)
                      </span>
                    </div>
                    <div style={{ height: '6px', background: 'var(--surface-darker)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${pct}%`,
                          background: SOURCE_COLORS[src] || 'var(--cyan)',
                          transition: 'width 0.4s ease',
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Historical Detection Logs Table */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 4px' }}>System Detection History</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Query paginated logs stored in SQL Server ({totalLogs} entries total)
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <input
              type="text"
              placeholder="Search filename or emotion..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                padding: '6px 12px', borderRadius: 'var(--radius-md)',
                background: 'var(--surface-darker)', border: '1px solid var(--border)',
                color: 'var(--text)', fontSize: '0.8rem', width: '200px',
              }}
            />

            <select
              value={sourceFilter}
              onChange={(e) => { setSourceFilter(e.target.value); setPage(1) }}
              style={{ padding: '6px 12px', borderRadius: 'var(--radius-md)', background: 'var(--surface-darker)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: '0.8rem' }}
            >
              <option value="">All Sources</option>
              <option value="upload">Upload</option>
              <option value="batch">Batch Folder</option>
              <option value="live">Live Stream</option>
            </select>

            <select
              value={maskFilter}
              onChange={(e) => { setMaskFilter(e.target.value); setPage(1) }}
              style={{ padding: '6px 12px', borderRadius: 'var(--radius-md)', background: 'var(--surface-darker)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: '0.8rem' }}
            >
              <option value="">All Mask Status</option>
              <option value="with_mask">With Mask</option>
              <option value="without_mask">Without Mask</option>
            </select>

            <select
              value={emotionFilter}
              onChange={(e) => { setEmotionFilter(e.target.value); setPage(1) }}
              style={{ padding: '6px 12px', borderRadius: 'var(--radius-md)', background: 'var(--surface-darker)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: '0.8rem' }}
            >
              <option value="">All Emotions</option>
              <option value="happy">Happy 😊</option>
              <option value="sad">Sad 😢</option>
              <option value="angry">Angry 😠</option>
              <option value="neutral">Neutral 😐</option>
              <option value="surprise">Surprise 😮</option>
              <option value="fear">Fear 😱</option>
              <option value="disgust">Disgust 🤢</option>
            </select>

            <button
              className="btn btn--secondary"
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={handleExportCSV}
              id="export-analytics-csv-btn"
            >
              📥 Export CSV
            </button>
          </div>
        </div>

        {loadingLogs ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} className="skeleton" style={{ height: '44px' }} />
            ))}
          </div>
        ) : (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '10px' }}>#</th>
                    <th style={{ padding: '10px' }}>TIMESTAMP</th>
                    <th style={{ padding: '10px' }}>SRC</th>
                    <th style={{ padding: '10px' }}>FACES</th>
                    <th style={{ padding: '10px' }}>MASK RESULT</th>
                    <th style={{ padding: '10px' }}>CONFIDENCE</th>
                    <th style={{ padding: '10px' }}>EMOTION RESULT</th>
                    <th style={{ padding: '10px' }}>CONFIDENCE</th>
                    <th style={{ padding: '10px' }}>MS</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHistory.map(log => (
                    <tr key={log.id} style={{ borderBottom: '1px solid var(--border-subtle)', fontSize: '0.85rem' }}>
                      <td style={{ padding: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {log.id}
                      </td>
                      <td style={{ padding: '10px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                        {new Date(log.timestamp).toLocaleString()}
                      </td>
                      <td style={{ padding: '10px', textTransform: 'capitalize', fontWeight: 600, color: SOURCE_COLORS[log.source] || 'var(--cyan)' }}>
                        {log.source || '—'}
                      </td>
                      <td style={{ padding: '10px', fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                        {log.faces_detected}
                      </td>
                      <td style={{ padding: '10px' }}>
                        <span className={`badge ${log.mask_result === 'with_mask' ? 'badge--green' : 'badge--red'}`}>
                          {log.mask_result || '—'}
                        </span>
                      </td>
                      <td style={{ padding: '10px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {log.mask_confidence ? `${(log.mask_confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding: '10px' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', textTransform: 'capitalize' }}>
                          <span>{EMOTION_EMOJIS[log.emotion_result] || ''}</span>
                          <strong style={{ color: EMOTION_COLORS[log.emotion_result] || 'var(--cyan)' }}>
                            {log.emotion_result || '—'}
                          </strong>
                        </span>
                      </td>
                      <td style={{ padding: '10px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {log.emotion_confidence ? `${(log.emotion_confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding: '10px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {log.processing_time_ms ? `${log.processing_time_ms.toFixed(1)}ms` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '20px', flexWrap: 'wrap' }}>
              <button
                className="btn btn--ghost"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                style={{ minWidth: '44px' }}
              >
                ← Prev
              </button>
              <span style={{ display: 'flex', alignItems: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)', padding: '0 12px' }}>
                Page {page} of {Math.ceil(totalLogs / 15) || 1}
              </span>
              <button
                className="btn btn--ghost"
                onClick={() => setPage(p => p + 1)}
                disabled={page >= Math.ceil(totalLogs / 15)}
                style={{ minWidth: '44px' }}
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
