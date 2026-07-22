import React from 'react'

const EMOTION_COLORS = {
  angry:   '#FF3366',
  disgust: '#FF6B35',
  fear:    '#FFB700',
  happy:   '#00FFB3',
  neutral: '#94A3B8',
  sad:     '#7B61FF',
  surprise: '#E040FB',
}

const EMOTION_EMOJIS = {
  angry: '😠', disgust: '🤢', fear: '😨',
  happy: '😊', neutral: '😐', sad: '😢', surprise: '😲',
}

/**
 * EmotionMeter — Animated confidence bar visualizer for 7 facial emotion classes.
 *
 * @param {Object} props - Component properties.
 * @param {Object.<string, number>} [props.allProbs={}] - Map of emotion labels to confidence probabilities (0.0 to 1.0).
 * @param {string} [props.topLabel=''] - Top predicted emotion class label string.
 * @returns {JSX.Element} Emotion probability visualization bars component.
 */
export default function EmotionMeter({ allProbs = {}, topLabel = '' }) {
  const sorted = Object.entries(allProbs).sort(([, a], [, b]) => b - a)

  return (
    <div style={{ width: '100%' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        marginBottom: '16px',
      }}>
        <span style={{ fontSize: '1.5rem' }}>{EMOTION_EMOJIS[topLabel] || '🎭'}</span>
        <div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: 'var(--text-muted)', marginBottom: '2px',
          }}>
            Detected Emotion
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontWeight: 700,
            fontSize: '1.1rem', color: EMOTION_COLORS[topLabel] || 'var(--accent)',
            textTransform: 'capitalize',
          }}>
            {topLabel || 'Unknown'}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {sorted.map(([emotion, prob]) => {
          const pct   = Math.round(prob * 100)
          const color = EMOTION_COLORS[emotion] || 'var(--accent)'
          const isTop = emotion === topLabel

          return (
            <div key={emotion} className="confidence-bar"
              style={{ opacity: isTop ? 1 : 0.65 }}>
              <div className="confidence-bar__label">
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>{EMOTION_EMOJIS[emotion]}</span>
                  <span style={{ textTransform: 'capitalize',
                    color: isTop ? color : 'var(--text-secondary)',
                    fontWeight: isTop ? 700 : 400,
                  }}>
                    {emotion}
                  </span>
                </span>
                <span style={{ color: isTop ? color : 'var(--text-muted)' }}>
                  {pct}%
                </span>
              </div>
              <div className="confidence-bar__track">
                <div
                  className="confidence-bar__fill"
                  style={{
                    width: `${pct}%`,
                    background: color,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
