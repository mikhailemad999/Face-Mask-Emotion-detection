import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar        from './components/Navbar'
import DashboardPage from './pages/DashboardPage'
import LiveCameraPage from './pages/LiveCameraPage'
import AnalyzePage   from './pages/AnalyzePage'
import BatchAnalyzePage from './pages/BatchAnalyzePage'
import AnalyticsPage from './pages/AnalyticsPage'
import ModelsPage    from './pages/ModelsPage'
import './index.css'

/**
 * App — Root React Component configuring BrowserRouter and application routes.
 *
 * @returns {JSX.Element} Rendered application layout with navigation bar and page routes.
 */
function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"          element={<DashboardPage />} />
        <Route path="/live"      element={<LiveCameraPage />} />
        <Route path="/analyze"   element={<AnalyzePage />} />
        <Route path="/batch"     element={<BatchAnalyzePage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/models"    element={<ModelsPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
