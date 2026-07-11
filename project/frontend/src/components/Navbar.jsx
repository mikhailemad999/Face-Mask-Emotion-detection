import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { path: '/',          label: 'Dashboard' },
  { path: '/live',      label: 'Live Camera' },
  { path: '/analyze',   label: 'Analyze Image' },
  { path: '/analytics', label: 'Analytics' },
  { path: '/models',    label: 'Models' },
]

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <nav className="nav" role="navigation" aria-label="Main navigation">
      {/* Brand */}
      <NavLink to="/" className="nav__brand" aria-label="Face Mask Detection Home">
        <span className="nav__brand-dot" aria-hidden="true" />
        FACEGUARD<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>.AI</span>
      </NavLink>

      {/* Desktop links */}
      <ul className="nav__links" role="list">
        {NAV_ITEMS.map(({ path, label }) => (
          <li key={path}>
            <NavLink
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `nav__link${isActive ? ' nav__link--active' : ''}`
              }
            >
              {label}
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Status indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span
          className="status-dot status-dot--online"
          aria-label="System online"
          title="System online"
        />
        <span
          className="font-mono"
          style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}
        >
          ONLINE
        </span>
      </div>
    </nav>
  )
}
