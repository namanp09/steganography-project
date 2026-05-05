import React, { useState } from 'react'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import ImagePage from './pages/ImagePage'
import AudioPage from './pages/AudioPage'
import VideoPage from './pages/VideoPage'
import HomePage from './pages/HomePage'
import MethodComparison from './components/MethodComparison'
import AttackResults from './components/AttackResults'

export default function App() {
  const [activePage, setActivePage] = useState('home')

  const pages = {
    home: <HomePage onNavigate={setActivePage} />,
    image: <ImagePage />,
    audio: <AudioPage />,
    video: <VideoPage />,
    compare: (
      <div className="max-w-6xl mx-auto">
        <MethodComparison />
      </div>
    ),
    attacks: (
      <div className="max-w-6xl mx-auto">
        <AttackResults />
      </div>
    ),
  }

  return (
    <div className="min-h-screen bg-gray-950">
      <Toaster position="top-right" toastOptions={{ style: { background: '#1e293b', color: '#f1f5f9' } }} />
      <Navbar activePage={activePage} onNavigate={setActivePage} />
      <main className="max-w-7xl mx-auto px-4 py-8">
        {pages[activePage]}
      </main>
    </div>
  )
}
