import React from 'react'
import { Shield, Image, Music, Video, Lock, Brain, BarChart3, Zap } from 'lucide-react'

const features = [
  { icon: Lock, title: 'AES-256-GCM Encryption', desc: 'Military-grade authenticated encryption before embedding' },
  { icon: Brain, title: 'Deep Learning Models', desc: 'U-Net++, HiDDeN GAN, and Invertible Neural Networks' },
  { icon: Zap, title: 'Codec-Robust Training', desc: 'Survives H.264/H.265 compression and noise attacks' },
  { icon: BarChart3, title: 'Comprehensive Metrics', desc: 'PSNR, SSIM, MS-SSIM, LPIPS, BER evaluation' },
]

const mediaCards = [
  { id: 'image', icon: Image, title: 'Image Steganography', desc: 'LSB, DCT, DWT, and deep learning methods', color: 'from-blue-500 to-cyan-500' },
  { id: 'audio', icon: Music, title: 'Audio Steganography', desc: 'LSB and DWT-based audio embedding', color: 'from-purple-500 to-pink-500' },
  { id: 'video', icon: Video, title: 'Video Steganography', desc: 'INN, 3D CNN, motion-compensated (Primary Focus)', color: 'from-orange-500 to-red-500' },
]

export default function HomePage({ onNavigate }) {
  return (
    <div className="space-y-16">
      {/* Hero */}
      <section className="text-center py-16">
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-blue-500/10 rounded-2xl border border-blue-500/20">
            <Shield className="w-12 h-12 text-blue-400" />
          </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            AI-Enhanced Steganography
          </span>
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto mb-8">
          Multi-modal secure steganography system using state-of-the-art deep learning,
          transform-domain techniques, and authenticated encryption.
        </p>
        <div className="flex gap-4 justify-center">
          <button onClick={() => onNavigate('image')} className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-medium transition-all">
            Get Started
          </button>
        </div>
      </section>

      {/* Media Cards */}
      <section className="grid md:grid-cols-3 gap-6">
        {mediaCards.map(({ id, icon: Icon, title, desc, color }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            className="text-left p-6 bg-gray-900/50 rounded-2xl border border-gray-800 hover:border-gray-600 transition-all group"
          >
            <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
              <Icon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">{title}</h3>
            <p className="text-sm text-gray-400">{desc}</p>
          </button>
        ))}
      </section>

      {/* Features */}
      <section>
        <h2 className="text-2xl font-bold text-center mb-8">Advanced Techniques</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {features.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="p-5 bg-gray-900/30 rounded-xl border border-gray-800">
              <Icon className="w-8 h-8 text-blue-400 mb-3" />
              <h4 className="font-semibold mb-1">{title}</h4>
              <p className="text-xs text-gray-400">{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
