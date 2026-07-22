import React, { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

/**
 * ModelsPage — Model registry page detailing MobileNetV2 and EfficientNet-B0 architecture specs, metrics, and active version info.
 *
 * @returns {JSX.Element} Rendered model registry info page component.
 */
export default function ModelsPage() {
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API_BASE}/detect/models/`)
      .then(r => r.json())
      .then(d => setInfo(d))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const MODEL_SPECS = {
    mask: {
      name:        'MobileNetV2',
      task:        'Binary Classification',
      classes:     ['with_mask', 'without_mask'],
      input:       '224×224×3 RGB',
      params:      '3.4M',
      framework:   'PyTorch 2.2',
      export:      'PyTorch .pt + ONNX',
      technique:   'Fine-tuning (ImageNet pretrained)',
    },
    emotion: {
      name:        'EfficientNet-B0',
      task:        '7-class Classification',
      classes:     ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
      input:       '224×224×3 RGB',
      params:      '5.3M',
      framework:   'PyTorch 2.2',
      export:      'PyTorch .pt + ONNX',
      technique:   'Fine-tuning + Label Smoothing + Class Weights',
    },
  }

  return (
    <div className="main">
      <div className="container" style={{ paddingBottom: '48px' }}>
        <div className="page-header animate-fade-in-up">
          <div className="page-header__eyebrow">// model_registry.info</div>
          <h1 className="page-header__title">Model <span>Registry</span></h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '520px' }}>
            Architecture details, training configuration, and evaluation metrics for active models.
          </p>
        </div>

        {/* System info */}
        {!loading && info && (
          <div className="card card--glass mb-8 animate-fade-in-up"
            style={{ marginBottom: '24px' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
              letterSpacing: '0.12em', textTransform: 'uppercase',
              color: 'var(--text-muted)', marginBottom: '12px' }}>
              SYSTEM_INFO
            </div>
            <div className="grid grid--3">
              {[
                ['CUDA Available',  info.cuda_available ? '✓ YES' : '✗ NO', info.cuda_available ? 'accent' : 'red'],
                ['GPU Device',      'NVIDIA RTX 2060', 'purple'],
                ['Inference Device', info.device || 'N/A', 'accent'],
              ].map(([label, val, color]) => (
                <div key={label}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
                    color: 'var(--text-muted)', textTransform: 'uppercase',
                    letterSpacing: '0.1em', marginBottom: '4px' }}>
                    {label}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700,
                    color: `var(--${color})`, fontSize: '0.9rem' }}>
                    {val}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model cards */}
        <div className="grid grid--2 animate-fade-in-up">
          {Object.entries(MODEL_SPECS).map(([key, spec]) => {
            const dbModel = info?.models?.find(m => m.model_type === key)
            return (
              <div key={key} className="card card--neon-top">
                {/* Header */}
                <div className="flex justify-between items-center mb-6"
                  style={{ marginBottom: '20px' }}>
                  <div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
                      color: 'var(--text-muted)', textTransform: 'uppercase',
                      letterSpacing: '0.1em', marginBottom: '4px' }}>
                      {key.toUpperCase()} MODEL
                    </div>
                    <h3 style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent)' }}>
                      {spec.name}
                    </h3>
                  </div>
                  <span className={`detection-badge ${
                    dbModel?.is_active ? 'detection-badge--mask' : 'detection-badge--emotion'
                  }`} style={{ fontSize: '0.6rem' }}>
                    {dbModel?.is_active ? '● ACTIVE' : '○ INACTIVE'}
                  </span>
                </div>

                {/* Metrics from DB */}
                {dbModel && (
                  <div className="grid grid--3" style={{ marginBottom: '20px', gap: '12px' }}>
                    {[
                      ['ACCURACY', dbModel.accuracy ? `${(dbModel.accuracy * 100).toFixed(1)}%` : 'N/A'],
                      ['F1 SCORE', dbModel.f1_score ? `${(dbModel.f1_score * 100).toFixed(1)}%` : 'N/A'],
                      ['ROC-AUC',  dbModel.roc_auc  ? dbModel.roc_auc.toFixed(3) : 'N/A'],
                    ].map(([lbl, val]) => (
                      <div key={lbl} style={{
                        background: 'var(--bg-panel)', borderRadius: '8px',
                        padding: '12px', textAlign: 'center',
                      }}>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem',
                          color: 'var(--text-muted)', textTransform: 'uppercase',
                          marginBottom: '4px' }}>
                          {lbl}
                        </div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700,
                          fontSize: '1rem', color: 'var(--accent)' }}>
                          {val}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Spec table */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {[
                    ['Task',       spec.task],
                    ['Input Size', spec.input],
                    ['Parameters', spec.params],
                    ['Framework',  spec.framework],
                    ['Export',     spec.export],
                    ['Technique',  spec.technique],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between" style={{
                      padding: '6px 0',
                      borderBottom: '1px solid var(--border-subtle)',
                    }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                        color: 'var(--text-muted)' }}>
                        {k}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                        color: 'var(--text-secondary)', textAlign: 'right', maxWidth: '55%' }}>
                        {v}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Classes */}
                <div style={{ marginTop: '16px' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
                    color: 'var(--text-muted)', textTransform: 'uppercase',
                    letterSpacing: '0.1em', marginBottom: '8px' }}>
                    OUTPUT CLASSES
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {spec.classes.map(cls => (
                      <span key={cls} style={{
                        background: 'var(--bg-panel)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '4px',
                        padding: '3px 8px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.7rem',
                        color: 'var(--text-secondary)',
                        textTransform: 'capitalize',
                      }}>
                        {cls.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Training pipeline */}
        <div className="card mt-8 animate-fade-in-up" style={{ marginTop: '24px' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: 'var(--text-muted)', marginBottom: '16px' }}>
            TRAINING_PIPELINE
          </div>
          <div className="grid grid--4" style={{ gap: '12px' }}>
            {[
              { step: '01', label: 'Data Preprocessing', desc: '8-step pipeline: duplicates, outliers, missing values, SMOTE', color: 'accent' },
              { step: '02', label: 'Model Training', desc: 'GPU: RTX 2060 | CUDA 12.x | Early stopping + cosine LR', color: 'purple' },
              { step: '03', label: 'Evaluation', desc: 'Accuracy, F1, ROC-AUC, Confusion Matrix, Learning Curves', color: 'yellow' },
              { step: '04', label: 'ONNX Export', desc: 'Optimized for production inference, opset 17', color: 'red' },
            ].map(({ step, label, desc, color }) => (
              <div key={step} style={{
                background: 'var(--bg-panel)',
                borderRadius: '8px', padding: '16px',
                borderTop: `2px solid var(--${color})`,
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                  color: `var(--${color})`, marginBottom: '8px', fontWeight: 700 }}>
                  STEP_{step}
                </div>
                <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '0.9rem' }}>
                  {label}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)',
                  lineHeight: 1.5 }}>
                  {desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
