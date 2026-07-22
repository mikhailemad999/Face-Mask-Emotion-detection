import React, { useRef, useEffect, useState, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

/**
 * LiveCameraPage — Real-time webcam face mask & emotion detection page component.
 * Captures video frames from browser webcam, posts to API, and draws animated bounding box overlays.
 *
 * @returns {JSX.Element} Rendered webcam streaming page component.
 */
export default function LiveCameraPage() {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const intervalRef = useRef(null)

  const [isStreaming, setIsStreaming] = useState(false)
  const [detections,  setDetections]  = useState([])
  const [fps,         setFps]         = useState(0)
  const [error,       setError]       = useState(null)
  const [sessionId]                   = useState(() => crypto.randomUUID())

  const FRAME_INTERVAL_MS = 300   // ~3fps inference
  const EMOTION_COLORS = {
    angry:'#FF3366', disgust:'#FF6B35', fear:'#FFB700',
    happy:'#00FFB3', neutral:'#94A3B8', sad:'#7B61FF', surprise:'#E040FB',
  }

  /**
   * Request webcam stream access and start video element playback.
   * Includes progressive fallback constraints and clear error handling for device locks.
   */
  const [devices, setDevices]               = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

  // Enumerate video devices on mount or when permissions granted
  const loadVideoDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return
      const allDevices = await navigator.mediaDevices.enumerateDevices()
      const videoInputs = allDevices.filter(d => d.kind === 'videoinput')
      setDevices(videoInputs)
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId)
      }
    } catch (e) {
      console.warn('Could not enumerate devices:', e)
    }
  }, [selectedDeviceId])

  useEffect(() => {
    loadVideoDevices()
  }, [loadVideoDevices])

  /**
   * Request webcam stream access and start video element playback.
   * Includes device enumeration, specific device constraints, and multi-device fallback.
   */
  const startCamera = useCallback(async () => {
    setError(null)

    // Helper to stop any existing media stream tracks on element
    if (videoRef.current?.srcObject) {
      const existing = videoRef.current.srcObject
      if (existing && 'getTracks' in existing) {
        existing.getTracks().forEach(t => t.stop())
      }
      videoRef.current.srcObject = null
    }

    try {
      let stream = null

      // Strategy 1: Selected Device ID or Ideal 640x480
      try {
        const videoConstraint = selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId } }
          : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }

        stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraint,
          audio: false,
        })
      } catch (firstErr) {
        // Strategy 2: Basic video constraint
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        } catch (secondErr) {
          // Strategy 3: Iterate through all enumerated camera devices
          const allDevs = await navigator.mediaDevices.enumerateDevices()
          const vDevs = allDevs.filter(d => d.kind === 'videoinput')
          for (const dev of vDevs) {
            try {
              stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: dev.deviceId } },
                audio: false,
              })
              if (stream) {
                setSelectedDeviceId(dev.deviceId)
                break
              }
            } catch (devErr) {
              continue
            }
          }
          if (!stream) throw secondErr
        }
      }

      if (videoRef.current && stream) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setIsStreaming(true)
        loadVideoDevices()
        startInferenceLoop()
      }
    } catch (err) {
      if (err.name === 'NotReadableError' || err.name === 'TrackStartError' || err.message?.includes('in use') || err.message?.includes('Could not start')) {
        setError('CAMERA_DEVICE_IN_USE: Another application (Zoom, Teams, Skype, or another browser tab) is currently using your webcam. Please close the other tab/app or select a different camera below and click Start Camera.')
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('CAMERA_PERMISSION_DENIED: Access to webcam was blocked. Please enable camera permissions in your browser address bar.')
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('NO_CAMERA_FOUND: No camera device detected on your system. Please plug in a webcam.')
      } else {
        setError(`Camera error: ${err.message}`)
      }
    }
  }, [selectedDeviceId, loadVideoDevices])

  /**
   * Stop active webcam stream, clear interval timer, and reset overlay canvas.
   */
  const stopCamera = useCallback(() => {
    clearInterval(intervalRef.current)
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setDetections([])
    // Clear canvas
    const ctx = canvasRef.current?.getContext('2d')
    ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
  }, [])

  /**
   * Start synthetic demo video stream mode when physical webcam is locked by another OS application.
   */
  const startDemoStream = useCallback(() => {
    setError(null)
    stopCamera()

    const demoCanvas = document.createElement('canvas')
    demoCanvas.width = 640
    demoCanvas.height = 480
    const ctx = demoCanvas.getContext('2d')

    let angle = 0
    const animInterval = setInterval(() => {
      angle += 0.05
      // Render animated background & face oval
      ctx.fillStyle = '#0B1120'
      ctx.fillRect(0, 0, 640, 480)

      // Draw stylized face shape
      const faceX = 320 + Math.sin(angle) * 30
      const faceY = 220 + Math.cos(angle * 0.8) * 15

      ctx.fillStyle = '#F5D0C5'
      ctx.beginPath()
      ctx.ellipse(faceX, faceY, 90, 120, 0, 0, Math.PI * 2)
      ctx.fill()

      // Draw eyes
      ctx.fillStyle = '#1E293B'
      ctx.beginPath()
      ctx.arc(faceX - 35, faceY - 25, 10, 0, Math.PI * 2)
      ctx.arc(faceX + 35, faceY - 25, 10, 0, Math.PI * 2)
      ctx.fill()

      // Draw mask (cyan mask)
      ctx.fillStyle = '#00F0FF'
      ctx.beginPath()
      ctx.roundRect(faceX - 60, faceY + 5, 120, 75, 12)
      ctx.fill()

      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 3
      ctx.strokeRect(faceX - 60, faceY + 5, 120, 75)
    }, 50)

    const demoStream = demoCanvas.captureStream(30)
    if (videoRef.current) {
      videoRef.current.srcObject = demoStream
      videoRef.current.play().catch(() => {})
      setIsStreaming(true)
      startInferenceLoop()
    }
  }, [stopCamera])

  /**
   * Start recurring frame capture loop sending base64 frame data to /api/detect/frame/.
   */
  const startInferenceLoop = () => {
    let lastTime = Date.now()
    intervalRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return

      const video  = videoRef.current
      const canvas = document.createElement('canvas')
      canvas.width  = video.videoWidth  || 640
      canvas.height = video.videoHeight || 480
      const ctx2 = canvas.getContext('2d')
      ctx2.drawImage(video, 0, 0)
      const b64 = canvas.toDataURL('image/jpeg', 0.7)

      try {
        const res = await fetch(`${API_BASE}/detect/frame/`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ frame: b64, session_id: sessionId }),
        })
        if (!res.ok) return
        const data = await res.json()
        const preds = data.result?.predictions || []
        setDetections(preds)
        drawOverlay(preds, canvas.width, canvas.height)

        const now = Date.now()
        setFps(Math.round(1000 / (now - lastTime)))
        lastTime = now
      } catch (e) {
        // silently skip failed frames
      }
    }, FRAME_INTERVAL_MS)
  }

  /**
   * Render bounding box overlays and text labels on the HTML5 canvas.
   *
   * @param {Array<Object>} preds - Array of face prediction objects containing bboxes and classification scores.
   * @param {number} w - Canvas pixel width.
   * @param {number} h - Canvas pixel height.
   */
  const drawOverlay = (preds, w, h) => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width  = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, w, h)

    preds.forEach(pred => {
      const { x, y, w: bw, h: bh } = pred.bbox
      const isMask  = pred.mask.label === 'with_mask'
      const color   = isMask ? '#00FFB3' : '#FF3366'
      const emotion = pred.emotion.label

      // Bounding box
      ctx.strokeStyle = color
      ctx.lineWidth   = 2
      ctx.shadowColor = color
      ctx.shadowBlur  = 8
      ctx.strokeRect(x, y, bw, bh)
      ctx.shadowBlur  = 0

      // Corner accents (L-shapes)
      const cs = 14
      ctx.lineWidth = 3
      ;[
        [x,      y,      cs, 0,   0,  cs],
        [x+bw,   y,      -cs, 0,  0,  cs],
        [x,      y+bh,   cs, 0,   0, -cs],
        [x+bw,   y+bh,   -cs, 0,  0, -cs],
      ].forEach(([cx, cy, dx1, dy1, dx2, dy2]) => {
        ctx.beginPath()
        ctx.moveTo(cx + dx1, cy + dy1)
        ctx.lineTo(cx, cy)
        ctx.lineTo(cx + dx2, cy + dy2)
        ctx.stroke()
      })

      // Label background
      const label = `${isMask ? '✓ MASK' : '✗ NO MASK'} | ${emotion}`
      ctx.font = 'bold 11px JetBrains Mono, monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = color + 'CC'
      ctx.fillRect(x, y - 22, tw + 12, 20)
      ctx.fillStyle = '#060B18'
      ctx.fillText(label, x + 6, y - 7)
    })
  }

  useEffect(() => () => stopCamera(), [])

  return (
    <div className="main">
      <div className="container" style={{ paddingBottom: '48px' }}>
        {/* Header */}
        <div className="page-header animate-fade-in-up">
          <div className="page-header__eyebrow">// live_detection.stream</div>
          <h1 className="page-header__title">
            Live <span>Camera</span> Detection
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '520px' }}>
            Real-time face mask and emotion detection from your webcam.
            Inference runs every 300ms on the server GPU.
          </p>
        </div>

        <div className="grid grid--2" style={{ gap: '24px', alignItems: 'start' }}>
          {/* Camera feed */}
          <div>
            <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
              {/* Status bar */}
              <div style={{
                padding: '12px 16px',
                background: 'var(--bg-panel)',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    className={`status-dot ${isStreaming ? 'status-dot--online' : 'status-dot--offline'}`}
                  />
                  <span style={{ color: 'var(--text-muted)' }}>
                    {isStreaming ? 'STREAMING' : 'CAMERA OFF'}
                  </span>
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {isStreaming && <span style={{ color: 'var(--accent)' }}>{fps} FPS</span>}
                </div>
              </div>

              {/* Video + overlay */}
              <div style={{ position: 'relative', background: '#000', minHeight: '320px' }}>
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  style={{
                    width: '100%', display: 'block',
                    opacity: isStreaming ? 1 : 0.3,
                    transition: 'opacity 0.3s',
                  }}
                  aria-label="Webcam feed"
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    position: 'absolute', top: 0, left: 0,
                    width: '100%', height: '100%',
                    pointerEvents: 'none',
                  }}
                  aria-hidden="true"
                />
                {!isStreaming && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexDirection: 'column', gap: '12px',
                  }}>
                    <div style={{ fontSize: '3rem', opacity: 0.4 }}>📷</div>
                    <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
                      fontSize: '0.85rem' }}>
                      Camera not active
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {devices.length > 1 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      SELECT_CAMERA_DEVICE
                    </label>
                    <select
                      value={selectedDeviceId}
                      onChange={(e) => {
                        setSelectedDeviceId(e.target.value)
                        if (isStreaming) {
                          stopCamera()
                          setTimeout(() => startCamera(), 300)
                        }
                      }}
                      style={{
                        padding: '8px 12px', borderRadius: 'var(--radius-md)',
                        background: 'var(--surface-darker)', border: '1px solid var(--border)',
                        color: 'var(--text)', fontSize: '0.85rem', width: '100%',
                      }}
                    >
                      {devices.map((d, i) => (
                        <option key={d.deviceId || i} value={d.deviceId}>
                          {d.label || `Camera Device ${i + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <button
                  className={`btn ${isStreaming ? 'btn--danger' : 'btn--primary'} w-full`}
                  style={{ justifyContent: 'center' }}
                  onClick={isStreaming ? stopCamera : startCamera}
                  id={isStreaming ? 'stop-camera-button' : 'start-camera-button'}
                >
                  {isStreaming ? '⏹ Stop Camera' : '▶ Start Camera'}
                </button>
              </div>
            </div>

            {error && (
              <div className="card card--red mt-4 animate-fade-in-up">
                <div style={{ color: 'var(--red)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', marginBottom: '12px' }}>
                  ⚠️ {error}
                </div>

                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <button
                    className="btn btn--secondary"
                    style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                    onClick={() => {
                      stopCamera()
                      setTimeout(() => startCamera(), 400)
                    }}
                  >
                    🔄 Release & Retry Camera
                  </button>

                  <button
                    className="btn btn--primary"
                    style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                    onClick={startDemoStream}
                  >
                    🎬 Test Demo Stream Mode
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Live predictions panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="card animate-fade-in-up">
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.12em', textTransform: 'uppercase',
                color: 'var(--text-muted)', marginBottom: '16px',
              }}>
                LIVE_PREDICTIONS ({detections.length} face{detections.length !== 1 ? 's' : ''})
              </div>

              {detections.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem', textAlign: 'center', padding: '32px' }}>
                  {isStreaming ? 'Scanning for faces…' : 'Start camera to begin detection'}
                </div>
              ) : (
                detections.map((pred, i) => (
                  <div key={i} style={{
                    background: 'var(--bg-panel)',
                    borderRadius: '8px', padding: '16px',
                    marginBottom: '12px',
                    border: '1px solid var(--border-subtle)',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between',
                      marginBottom: '12px' }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                        color: 'var(--text-muted)' }}>
                        Face #{i + 1}
                      </span>
                      <span className={`detection-badge ${
                        pred.mask.label === 'with_mask'
                          ? 'detection-badge--mask' : 'detection-badge--no-mask'
                      }`} style={{ fontSize: '0.65rem' }}>
                        {pred.mask.label === 'with_mask' ? '✓ MASK' : '✗ NO MASK'}
                      </span>
                    </div>

                    {/* Emotion mini-bars */}
                    {Object.entries(pred.emotion.all_probs || {})
                      .sort(([,a],[,b]) => b-a)
                      .slice(0, 3)
                      .map(([emo, prob]) => (
                        <div key={emo} style={{ marginBottom: '6px' }}>
                          <div style={{
                            display: 'flex', justifyContent: 'space-between',
                            fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                            color: 'var(--text-secondary)', marginBottom: '3px',
                          }}>
                            <span style={{ textTransform: 'capitalize' }}>{emo}</span>
                            <span>{Math.round(prob * 100)}%</span>
                          </div>
                          <div style={{
                            height: '4px', background: 'var(--bg-deep)',
                            borderRadius: '2px', overflow: 'hidden',
                          }}>
                            <div style={{
                              height: '100%',
                              width: `${prob * 100}%`,
                              background: EMOTION_COLORS[emo] || 'var(--accent)',
                              borderRadius: '2px',
                              transition: 'width 200ms ease',
                            }} />
                          </div>
                        </div>
                      ))}
                  </div>
                ))
              )}
            </div>

            {/* Session info */}
            <div className="card card--glass animate-fade-in-up">
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.12em', textTransform: 'uppercase',
                color: 'var(--text-muted)', marginBottom: '12px' }}>
                SESSION_INFO
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                <div className="flex justify-between" style={{ marginBottom: '6px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>session_id</span>
                </div>
                <div style={{ color: 'var(--accent)', fontSize: '0.7rem' }}>{sessionId}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
