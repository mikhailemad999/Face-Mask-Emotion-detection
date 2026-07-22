/**
 * @file BatchAnalyzePage.jsx
 * @description Folder & Multi-File Batch Analytics Page.
 * Uploads an entire folder of image files to compute emotion breakdowns (happy, sad, angry, etc.) and mask compliance.
 */
import React, { useState, useRef } from 'react'

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
  happy: '#00FFB3',
  sad: '#7B61FF',
  angry: '#FF3366',
  neutral: '#94A3B8',
  surprise: '#E040FB',
  fear: '#FFB700',
  disgust: '#FF6B35',
}

export default function BatchAnalyzePage() {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isDragging, setIsDragging]       = useState(false)
  const [loading, setLoading]             = useState(false)
  const [result, setResult]               = useState(null)
  const [error, setError]                 = useState(null)
  const [searchFilter, setSearchFilter]   = useState('')

  const fileInputRef   = useRef(null)
  const folderInputRef = useRef(null)

  /**
   * Filter uploaded files to valid images only.
   */
  const handleFilesAdded = (filesList) => {
    const valid = Array.from(filesList).filter(file =>
      /\.(jpe?g|png|webp|bmp)$/i.test(file.name)
    )
    setSelectedFiles(valid)
    setError(null)
    setResult(null)
  }

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false) }
  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.length) {
      handleFilesAdded(e.dataTransfer.files)
    }
  }

  /**
   * Submit selected batch files to backend POST /api/detect/batch/
   */
  const handleAnalyzeBatch = async () => {
    if (!selectedFiles.length) {
      setError('Please select or drag a folder containing images first.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      selectedFiles.forEach(file => {
        formData.append('images', file)
      })

      const res = await fetch('/api/detect/batch/', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.error || `Server returned error status ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  /**
   * Download batch results summary as a CSV file.
   */
  const handleExportCSV = () => {
    if (!result?.file_results) return

    const headers = ['Filename', 'Faces Detected', 'Top Mask', 'Top Emotion', 'Processing Time (ms)']
    const rows = result.file_results.map(f => [
      `"${f.filename}"`,
      f.faces_detected,
      `"${f.top_mask || 'N/A'}"`,
      `"${f.top_emotion || 'N/A'}"`,
      f.processing_ms ? f.processing_ms.toFixed(1) : '0',
    ])

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', `FaceGuard_Batch_Report_${Date.now()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Filter file table results by search query
  const filteredFileResults = result?.file_results?.filter(f =>
    f.filename.toLowerCase().includes(searchFilter.toLowerCase()) ||
    (f.top_emotion && f.top_emotion.toLowerCase().includes(searchFilter.toLowerCase())) ||
    (f.top_mask && f.top_mask.toLowerCase().includes(searchFilter.toLowerCase()))
  ) || []

  // Compute percentages for emotion breakdown
  const totalFaces = result?.total_faces || 0
  const maskCompliancePct = totalFaces > 0
    ? Math.round(((result?.mask_counts?.with_mask || 0) / totalFaces) * 100)
    : 0

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
          color: 'var(--cyan)', letterSpacing: '0.12em',
          textTransform: 'uppercase', marginBottom: '8px',
        }}>
          // FOLDER_BATCH_ANALYZER
        </div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, margin: '0 0 8px' }}>
          Folder Analytics & <span style={{ color: 'var(--cyan)' }}>Batch Analysis</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, maxWidth: '650px' }}>
          Upload an entire folder or select multiple photos to analyze emotion distributions (Happy, Sad, Angry, Neutral, Surprise, Disgust, Fear) and face mask compliance in bulk.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="card" style={{ marginBottom: '32px' }}>
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          style={{
            border: `2px dashed ${isDragging ? 'var(--cyan)' : 'var(--border)'}`,
            borderRadius: 'var(--radius-lg)',
            padding: '40px 24px',
            textAlign: 'center',
            background: isDragging ? 'rgba(0, 240, 255, 0.05)' : 'var(--surface-darker)',
            transition: 'all 0.2s ease',
          }}
        >
          <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>📁</div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 8px' }}>
            Select or Drag & Drop Folder of Images
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '20px' }}>
            Upload folder of JPG, PNG, or WebP images to generate bulk sentiment reports.
          </p>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {/* Hidden Folder Input */}
            <input
              type="file"
              ref={folderInputRef}
              webkitdirectory="true"
              directory="true"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => e.target.files && handleFilesAdded(e.target.files)}
            />

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              multiple
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => e.target.files && handleFilesAdded(e.target.files)}
            />

            <button
              type="button"
              className="btn btn--primary"
              onClick={() => folderInputRef.current?.click()}
              id="select-folder-btn"
            >
              📂 Choose Folder
            </button>

            <button
              type="button"
              className="btn btn--secondary"
              onClick={() => fileInputRef.current?.click()}
              id="select-files-btn"
            >
              🖼 Select Multiple Images
            </button>
          </div>

          {selectedFiles.length > 0 && (
            <div style={{
              marginTop: '20px', padding: '12px 16px',
              background: 'var(--surface)', borderRadius: 'var(--radius-md)',
              display: 'inline-flex', alignItems: 'center', gap: '12px',
            }}>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)', fontSize: '0.85rem' }}>
                {selectedFiles.length} image files selected
              </span>
              <button
                className="btn btn--secondary"
                style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                onClick={handleAnalyzeBatch}
                disabled={loading}
                id="run-batch-analysis-btn"
              >
                {loading ? '⚡ Analyzing Batch...' : '▶ Run Batch Analysis'}
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="card card--red mt-4 animate-fade-in-up">
            <div style={{ color: 'var(--red)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              ⚠️ {error}
            </div>
          </div>
        )}
      </div>

      {/* Batch Results Output */}
      {result && (
        <div className="animate-fade-in-up">
          {/* Executive Summary Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <div className="card">
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '8px' }}>
                TOTAL_FILES_ANALYZED
              </div>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--cyan)' }}>
                {result.total_files}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                {result.total_faces} faces detected total
              </div>
            </div>

            <div className="card">
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '8px' }}>
                DOMINANT_EMOTION
              </div>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, textTransform: 'capitalize', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>{EMOTION_EMOJIS[result.dominant_emotion] || '😐'}</span>
                <span>{result.dominant_emotion}</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Most frequent facial sentiment
              </div>
            </div>

            <div className="card">
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginBottom: '8px' }}>
                MASK_COMPLIANCE_RATE
              </div>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: maskCompliancePct >= 70 ? 'var(--green)' : 'var(--red)' }}>
                {maskCompliancePct}%
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                {result.mask_counts?.with_mask || 0} with mask / {result.mask_counts?.without_mask || 0} without
              </div>
            </div>
          </div>

          {/* Emotion & Mask Distribution Breakdown */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '32px' }}>
            {/* Emotion Breakdown Card */}
            <div className="card">
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: '0 0 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Facial Sentiment Breakdown</span>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  EMOTION_DISTRIBUTION
                </span>
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {Object.entries(result.emotion_counts || {}).map(([emotion, count]) => {
                  const pct = totalFaces > 0 ? Math.round((count / totalFaces) * 100) : 0
                  return (
                    <div key={emotion}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', textTransform: 'capitalize' }}>
                          <span>{EMOTION_EMOJIS[emotion]}</span>
                          <strong style={{ color: EMOTION_COLORS[emotion] }}>{emotion}</strong>
                        </span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                          {count} ({pct}%)
                        </span>
                      </div>
                      <div style={{ height: '8px', background: 'var(--surface-darker)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div
                          style={{
                            height: '100%',
                            width: `${pct}%`,
                            background: EMOTION_COLORS[emotion] || 'var(--cyan)',
                            transition: 'width 0.4s ease',
                          }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Mask Compliance Breakdown Card */}
            <div className="card">
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: '0 0 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Mask Compliance Analysis</span>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  MASK_RATIO
                </span>
              </h3>

              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '8px' }}>
                  <span>😷 With Mask</span>
                  <strong style={{ color: 'var(--green)' }}>
                    {result.mask_counts?.with_mask || 0} ({totalFaces > 0 ? Math.round(((result.mask_counts?.with_mask || 0) / totalFaces) * 100) : 0}%)
                  </strong>
                </div>
                <div style={{ height: '10px', background: 'var(--surface-darker)', borderRadius: '5px', overflow: 'hidden', marginBottom: '16px' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${totalFaces > 0 ? ((result.mask_counts?.with_mask || 0) / totalFaces) * 100 : 0}%`,
                      background: 'var(--green)',
                      transition: 'width 0.4s ease',
                    }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '8px' }}>
                  <span>🔴 Without Mask</span>
                  <strong style={{ color: 'var(--red)' }}>
                    {result.mask_counts?.without_mask || 0} ({totalFaces > 0 ? Math.round(((result.mask_counts?.without_mask || 0) / totalFaces) * 100) : 0}%)
                  </strong>
                </div>
                <div style={{ height: '10px', background: 'var(--surface-darker)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${totalFaces > 0 ? ((result.mask_counts?.without_mask || 0) / totalFaces) * 100 : 0}%`,
                      background: 'var(--red)',
                      transition: 'width 0.4s ease',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* File Results Table */}
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 4px' }}>File-by-File Breakdown</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Detailed analysis results for all files in folder
                </span>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <input
                  type="text"
                  placeholder="Filter by filename or emotion..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  style={{
                    padding: '8px 14px', borderRadius: 'var(--radius-md)',
                    background: 'var(--surface-darker)', border: '1px solid var(--border)',
                    color: 'var(--text)', fontSize: '0.85rem', width: '240px',
                  }}
                />

                <button
                  className="btn btn--secondary"
                  onClick={handleExportCSV}
                  id="export-csv-btn"
                >
                  📥 Export CSV
                </button>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '12px' }}>FILENAME</th>
                    <th style={{ padding: '12px' }}>FACES</th>
                    <th style={{ padding: '12px' }}>TOP MASK</th>
                    <th style={{ padding: '12px' }}>TOP EMOTION</th>
                    <th style={{ padding: '12px' }}>LATENCY</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredFileResults.map((f, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)', fontSize: '0.85rem' }}>
                      <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                        {f.filename}
                      </td>
                      <td style={{ padding: '12px', fontFamily: 'var(--font-mono)' }}>
                        {f.faces_detected}
                      </td>
                      <td style={{ padding: '12px' }}>
                        {f.top_mask ? (
                          <span className={`badge ${f.top_mask === 'with_mask' ? 'badge--green' : 'badge--red'}`}>
                            {f.top_mask}
                          </span>
                        ) : '—'}
                      </td>
                      <td style={{ padding: '12px' }}>
                        {f.top_emotion ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', textTransform: 'capitalize' }}>
                            <span>{EMOTION_EMOJIS[f.top_emotion]}</span>
                            <strong style={{ color: EMOTION_COLORS[f.top_emotion] }}>{f.top_emotion}</strong>
                          </span>
                        ) : '—'}
                      </td>
                      <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {f.processing_ms ? `${f.processing_ms.toFixed(1)}ms` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
