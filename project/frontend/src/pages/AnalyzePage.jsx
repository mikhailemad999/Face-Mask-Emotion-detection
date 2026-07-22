import React, { useState, useCallback } from 'react'
import UploadZone from '../components/UploadZone'
import EmotionMeter from '../components/EmotionMeter'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

/**
 * AnalyzePage — Single image upload and deep inference analysis page component.
 *
 * @returns {JSX.Element} Rendered image analysis workspace page component.
 */
export default function AnalyzePage() {
  const [file,        setFile]        = useState(null)
  const [previewUrl,  setPreviewUrl]  = useState(null)
  const [isLoading,   setIsLoading]   = useState(false)
  const [result,      setResult]      = useState(null)
  const [error,       setError]       = useState(null)

  /**
   * Handle image selection callback from UploadZone component.
   * @param {File} f - Selected image file object.
   * @param {string} url - Browser object URL preview string.
   */
  const handleImageSelected = useCallback((f, url) => {
    setFile(f)
    setPreviewUrl(url)
    setResult(null)
    setError(null)
  }, [])

  /**
   * Post selected image file to backend DRF API (/api/detect/image/) for detection.
   */
  const handleAnalyze = async () => {
    if (!file) return
    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', file)

    try {
      const res = await fetch(`${API_BASE}/detect/image/`, {
        method: 'POST',
        body:   formData,
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult(data.result)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const topPred = result?.predictions?.[0]

  return (
    <div className="main">
      <div className="container" style={{ paddingBottom: '48px' }}>
        {/* Page header */}
        <div className="page-header animate-fade-in-up">
          <div className="page-header__eyebrow">// analyze_image.py</div>
          <h1 className="page-header__title">
            Analyze <span>Image</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '520px' }}>
            Upload a photo to run face mask and emotion detection.
            The AI pipeline will locate faces, classify mask status, and identify emotional expression.
          </p>
        </div>

        <div className="grid grid--2" style={{ gap: '24px', alignItems: 'start' }}>
          {/* Left: Upload + action */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="card">
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.12em', textTransform: 'uppercase',
                color: 'var(--text-muted)', marginBottom: '16px',
              }}>
                INPUT_IMAGE
              </div>
              <UploadZone
                onImageSelected={handleImageSelected}
                isLoading={isLoading}
              />
              <button
                className="btn btn--primary w-full"
                style={{ marginTop: '16px', justifyContent: 'center' }}
                onClick={handleAnalyze}
                disabled={!file || isLoading}
                id="analyze-button"
              >
                {isLoading ? (
                  <>
                    <span className="animate-spin" style={{ display: 'inline-block' }}>⟳</span>
                    Analyzing…
                  </>
                ) : '⚡ Run Detection'}
              </button>
            </div>

            {/* Processing metadata */}
            {result && (
              <div className="card card--glass animate-fade-in-up">
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                  letterSpacing: '0.12em', textTransform: 'uppercase',
                  color: 'var(--text-muted)', marginBottom: '12px',
                }}>
                  INFERENCE_METADATA
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px',
                  fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>faces_detected</span>
                    <span style={{ color: 'var(--accent)' }}>{result.faces_detected}</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>processing_time</span>
                    <span style={{ color: 'var(--accent)' }}>{result.processing_time_ms}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>device</span>
                    <span style={{ color: 'var(--purple)' }}>{result.device}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {error && (
              <div className="card card--red animate-fade-in-up">
                <div style={{ color: 'var(--red)', fontFamily: 'var(--font-mono)',
                  fontSize: '0.8rem' }}>
                  ERROR: {error}
                </div>
              </div>
            )}

            {!result && !error && (
              <div className="card animate-fade-in-up" style={{ textAlign: 'center', padding: '64px 32px' }}>
                <div style={{ fontSize: '3rem', marginBottom: '16px', opacity: 0.3 }}>🎯</div>
                <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem' }}>
                  Upload an image and click Run Detection
                </div>
              </div>
            )}

            {result && result.faces_detected === 0 && (
              <div className="card animate-fade-in-up" style={{ textAlign: 'center', padding: '48px 32px' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>😶</div>
                <div style={{ color: 'var(--yellow)', fontFamily: 'var(--font-mono)' }}>
                  No faces detected in this image.
                </div>
              </div>
            )}

            {topPred && (
              <>
                {/* Mask result */}
                <div className={`card ${topPred.mask.label === 'with_mask' ? 'card--accent' : 'card--red'} animate-fade-in-up`}>
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                    letterSpacing: '0.12em', textTransform: 'uppercase',
                    color: 'var(--text-muted)', marginBottom: '12px',
                  }}>
                    MASK_DETECTION
                  </div>
                  <div className="flex items-center gap-4" style={{ gap: '16px' }}>
                    <span
                      className={`detection-badge ${topPred.mask.label === 'with_mask' ? 'detection-badge--mask' : 'detection-badge--no-mask'}`}
                      style={{ fontSize: '1rem', padding: '8px 16px' }}
                    >
                      {topPred.mask.label === 'with_mask' ? '✓ MASK' : '✗ NO MASK'}
                    </span>
                    <div>
                      <div style={{
                        fontFamily: 'var(--font-mono)', fontSize: '1.5rem',
                        fontWeight: 700,
                        color: topPred.mask.label === 'with_mask' ? 'var(--accent)' : 'var(--red)',
                      }}>
                        {Math.round(topPred.mask.confidence * 100)}%
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)',
                        fontFamily: 'var(--font-mono)' }}>
                        confidence
                      </div>
                    </div>
                  </div>
                </div>

                {/* Emotion result */}
                <div className="card card--glass animate-fade-in-up">
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                    letterSpacing: '0.12em', textTransform: 'uppercase',
                    color: 'var(--text-muted)', marginBottom: '16px',
                  }}>
                    EMOTION_RECOGNITION
                  </div>
                  <EmotionMeter
                    allProbs={topPred.emotion.all_probs}
                    topLabel={topPred.emotion.label}
                  />
                </div>

                {/* All faces */}
                {result.predictions.length > 1 && (
                  <div className="card animate-fade-in-up">
                    <div style={{
                      fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                      letterSpacing: '0.12em', textTransform: 'uppercase',
                      color: 'var(--text-muted)', marginBottom: '12px',
                    }}>
                      ALL FACES ({result.predictions.length})
                    </div>
                    {result.predictions.map((pred, i) => (
                      <div key={i} style={{
                        display: 'flex', justifyContent: 'space-between',
                        alignItems: 'center', padding: '8px 0',
                        borderBottom: i < result.predictions.length - 1
                          ? '1px solid var(--border-subtle)' : 'none',
                      }}>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
                          color: 'var(--text-muted)' }}>
                          Face #{i + 1}
                        </span>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <span className={`detection-badge ${pred.mask.label === 'with_mask' ? 'detection-badge--mask' : 'detection-badge--no-mask'}`}>
                            {pred.mask.label === 'with_mask' ? 'mask' : 'no mask'}
                          </span>
                          <span className="detection-badge detection-badge--emotion">
                            {pred.emotion.label}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
