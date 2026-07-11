import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar        from './components/Navbar'
import DashboardPage from './pages/DashboardPage'
import LiveCameraPage from './pages/LiveCameraPage'
import AnalyzePage   from './pages/AnalyzePage'
import AnalyticsPage from './pages/AnalyticsPage'
import ModelsPage    from './pages/ModelsPage'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"          element={<DashboardPage />} />
        <Route path="/live"      element={<LiveCameraPage />} />
        <Route path="/analyze"   element={<AnalyzePage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/models"    element={<ModelsPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
