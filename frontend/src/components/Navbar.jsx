import React from 'react'
import { Shield, Image, Music, Video, Home } from 'lucide-react'

const navItems = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'image', label: 'Image', icon: Image },
  { id: 'audio', label: 'Audio', icon: Music },
  { id: 'video', label: 'Video', icon: Video },
]

export default function Navbar({ activePage, onNavigate }) {
  return (
    <nav className="bg-gray-900/80 backdrop-blur-md border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('home')}>
            <Shield className="w-6 h-6 text-blue-400" />
            <span className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              SteganoAI
            </span>
          </div>
          <div className="flex gap-1">
            {navItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activePage === id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}
