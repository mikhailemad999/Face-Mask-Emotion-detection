import React from 'react'

/**
 * StatsCard — a glassmorphism stat display card
 * Props: label, value, sub, color ('accent'|'red'|'purple'|'yellow')
 */
const COLOR_MAP = {
  accent: 'var(--accent)',
  red:    'var(--red)',
  purple: 'var(--purple)',
  yellow: 'var(--yellow)',
}

export default function StatsCard({ label, value, sub, color = 'accent', icon }) {
  const clr = COLOR_MAP[color] || COLOR_MAP.accent

  return (
    <div
      className="card card--glass card--neon-top"
      style={{ '--neon-color': clr }}
    >
      <style>{`
        .card--neon-top::before {
          background: linear-gradient(90deg, transparent, ${clr}, transparent) !important;
        }
      `}</style>

      {icon && (
        <div style={{
          fontSize: '1.5rem', marginBottom: '12px',
          color: clr, lineHeight: 1,
        }} aria-hidden="true">
          {icon}
        </div>
      )}

      <div className="stat-card__label">{label}</div>

      <div
        className="stat-card__value"
        style={{ color: clr }}
        aria-label={`${label}: ${value}`}
      >
        {value}
      </div>

      {sub && <div className="stat-card__sub">{sub}</div>}
    </div>
  )
}
