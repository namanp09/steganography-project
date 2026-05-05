import React, { useState } from 'react'
import { ShieldAlert, Trophy, Info } from 'lucide-react'

// ─── Data ────────────────────────────────────────────────────────────────────

const IMAGE_ATTACKS = [
  'Baseline', 'JPEG Q90', 'JPEG Q75', 'JPEG Q50', 'Gauss σ5', 'Gauss σ15',
  'Salt & Pepper', 'Gauss Blur', 'Median', 'Resize 50%', 'Brightness', 'Rotation 1°', 'Crop 5%',
]
const IMAGE_DATA = {
  LSB: [0.00, 0.52, 0.49, 0.51, 0.52, 0.51, 0.00, 0.54, 0.45, 0.48, 0.30, 0.49, 0.48],
  DCT: [0.00, 0.00, 0.10, 0.34, 0.13, 0.47, 0.24, 0.33, 0.45, 0.43, 0.08, 0.44, 0.47],
  DWT: [0.00, 0.18, 0.43, 0.49, 0.44, 0.51, 0.07, 0.44, 0.50, 0.44, 0.15, 0.50, 0.51],
  GAN: [0.375, 0.44, 0.375, 0.50, 0.375, 0.44, 0.375, 0.50, 0.31, 0.50, 0.375, 0.50, 0.44],
}
const IMAGE_SUMMARY = [
  { method: 'LSB', embedPSNR: 70.27, avgBER: 0.4403, note: 'Invisible but fragile to any filtering' },
  { method: 'DCT', embedPSNR: 46.87, avgBER: 0.2895, note: 'Best image robustness — survives JPEG & noise' },
  { method: 'DWT', embedPSNR: 50.29, avgBER: 0.3888, note: 'Good imperceptibility, moderate robustness' },
  { method: 'GAN', embedPSNR: 45.67, avgBER: 0.4271, note: 'Adversarial training — better on video' },
]

const AUDIO_ATTACKS = [
  'Baseline', 'AWGN 30dB', 'AWGN 20dB', 'LP 4kHz', 'LP 8kHz',
  'Vol ×0.8', 'Vol ×1.2', 'Resample', 'Time Shift', 'Echo',
]
const AUDIO_DATA = {
  LSB: [0.00, 0.49, 0.46, 0.52, 0.48, 0.49, 0.51, 0.46, 0.52, 0.46],
  DWT: [0.00, 0.39, 0.49, 0.52, 0.00, 0.53, 0.50, 0.00, 0.54, 0.50],
}
const AUDIO_SUMMARY = [
  { method: 'LSB', embedPSNR: 86.74, avgBER: 0.4874, note: 'Nearly inaudible but fails all attacks' },
  { method: 'DWT', embedPSNR: 49.92, avgBER: 0.3837, note: 'Best audio robustness — survives LP filter & resampling' },
]

const VIDEO_ATTACKS = [
  'Baseline', 'Gauss Noise', 'JPEG Q75', 'JPEG Q50', 'Resize 50%',
  'Frame Drop', 'Brightness', 'Gauss Blur', 'Salt & Pepper',
]
const VIDEO_DATA = {
  LSB: [0.00, 0.52, 0.50, 0.51, 0.49, 0.00, 0.32, 0.48, 0.01],
  DCT: [0.00, 0.02, 0.00, 0.00, 0.29, 0.00, 0.00, 1.00, 0.21],
  DWT: [0.00, 0.49, 0.09, 0.29, 0.25, 0.00, 0.00, 0.25, 0.06],
  GAN: [0.125, 0.125, 0.188, 0.188, 0.188, 0.125, 0.125, 0.125, 0.125],
}
const VIDEO_SUMMARY = [
  { method: 'LSB', embedPSNR: 98.79, avgBER: 0.3538, note: 'Survives frame-drop only' },
  { method: 'DCT', embedPSNR: 97.76, avgBER: 0.1902, note: 'Survives JPEG, noise, brightness & frame-drop' },
  { method: 'DWT', embedPSNR: 98.05, avgBER: 0.1794, note: 'Very robust — survives most attacks' },
  { method: 'GAN', embedPSNR: 34.03, avgBER: 0.1484, note: 'Most robust — trained with H.264 noise layer' },
]

// ─── Best method callouts ─────────────────────────────────────────────────────

const BEST = {
  image: {
    method: 'DCT',
    color: '#818cf8',
    title: 'DCT is the most robust image method',
    detail: 'Avg BER of 0.29 — it fully survives JPEG Q=90 and mild noise because it embeds data in the same frequency domain JPEG uses.',
  },
  audio: {
    method: 'DWT',
    color: '#c084fc',
    title: 'DWT is the most robust audio method',
    detail: 'Avg BER of 0.38 — survives low-pass filtering and resampling perfectly (BER = 0.0) because DWT operates in frequency sub-bands.',
  },
  video: {
    method: 'GAN',
    color: '#34d399',
    title: 'GAN is the most robust video method',
    detail: 'Avg BER of 0.15 — the lowest of all methods across all tabs. The model was adversarially trained with a simulated H.264 video compression noise layer.',
  },
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Softer thresholds: <0.15 green, 0.15–0.35 yellow, 0.35–0.49 orange, ≥0.49 red
function berColor(ber) {
  if (ber < 0.15) return 'text-emerald-400 bg-emerald-950/50'
  if (ber < 0.35) return 'text-yellow-300 bg-yellow-950/40'
  if (ber < 0.49) return 'text-orange-400 bg-orange-950/30'
  return 'text-red-400 bg-red-950/40'
}
function berLabel(ber) {
  if (ber < 0.15) return 'Robust'
  if (ber < 0.35) return 'Moderate'
  if (ber < 0.49) return 'Fragile'
  return 'Lost'
}

const METHOD_COLORS = { LSB: '#94a3b8', DCT: '#818cf8', DWT: '#c084fc', GAN: '#34d399' }

// ─── Components ──────────────────────────────────────────────────────────────

function BestMethodCard({ tab }) {
  const b = BEST[tab]
  return (
    <div
      className="flex items-start gap-4 p-4 rounded-xl border"
      style={{ borderColor: `${b.color}40`, backgroundColor: `${b.color}08` }}
    >
      <Trophy className="w-5 h-5 mt-0.5 shrink-0" style={{ color: b.color }} />
      <div>
        <p className="text-sm font-bold mb-1" style={{ color: b.color }}>
          {b.title}
        </p>
        <p className="text-xs text-gray-400 leading-relaxed">{b.detail}</p>
      </div>
    </div>
  )
}

function BerCell({ ber }) {
  return (
    <td className="px-3 py-2 text-center">
      <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold ${berColor(ber)}`}>
        {ber.toFixed(3)}
      </span>
    </td>
  )
}

function AttackTable({ attacks, data, noteGANskipped }) {
  const methods = Object.keys(data)
  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/70">
            <th className="px-3 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide">Attack</th>
            {methods.map((m) => (
              <th key={m} className="px-3 py-3 text-center text-xs font-bold uppercase tracking-wide whitespace-nowrap"
                style={{ color: METHOD_COLORS[m] }}>
                {m} BER
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {attacks.map((atk, i) => (
            <tr key={atk}
              className={`border-b border-gray-800/60 last:border-0 ${i % 2 === 0 ? 'bg-gray-900/20' : 'bg-gray-900/40'}`}>
              <td className="px-3 py-2 text-xs text-gray-300 font-medium whitespace-nowrap">{atk}</td>
              {methods.map((m) => <BerCell key={m} ber={data[m][i]} />)}
            </tr>
          ))}
        </tbody>
      </table>
      {noteGANskipped && (
        <p className="px-4 py-2 text-xs text-gray-500 border-t border-gray-800 flex items-center gap-1.5">
          <Info className="w-3 h-3" />
          GAN audio was not evaluated — the saved checkpoint was trained with different parameters and needs retraining.
        </p>
      )}
    </div>
  )
}

function SummaryTable({ rows }) {
  const bestBER = Math.min(...rows.filter(r => r.avgBER > 0).map(r => r.avgBER))
  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800 mt-4">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/70">
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide">Method</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Embed PSNR</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Avg BER</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Rating</th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide">Key Finding</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ method, embedPSNR, avgBER, note }, i) => {
            const color = METHOD_COLORS[method] ?? '#94a3b8'
            const isBest = avgBER === bestBER
            return (
              <tr key={method}
                className={`border-b border-gray-800/60 last:border-0 ${
                  isBest ? 'bg-emerald-950/20' : i % 2 === 0 ? 'bg-gray-900/20' : 'bg-gray-900/40'
                }`}>
                <td className="px-4 py-3 flex items-center gap-2">
                  <span className="text-xs font-bold px-2 py-0.5 rounded"
                    style={{ color, backgroundColor: `${color}20` }}>
                    {method}
                  </span>
                  {isBest && <Trophy className="w-3.5 h-3.5 text-yellow-400" />}
                </td>
                <td className="px-4 py-3 text-center text-xs font-mono text-gray-300">{embedPSNR.toFixed(2)} dB</td>
                <BerCell ber={avgBER} />
                <td className="px-4 py-3 text-center">
                  <span className={`text-xs font-semibold ${berColor(avgBER).split(' ')[0]}`}>
                    {berLabel(avgBER)}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-gray-400">{note}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ─── Tab panels ───────────────────────────────────────────────────────────────

function ImagePanel() {
  return (
    <div className="space-y-6">
      <BestMethodCard tab="image" />
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img src="/charts/chart_image.png" alt="Image attack simulation chart" className="w-full object-contain" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">BER per attack — 13 attacks × 4 methods</h3>
        <AttackTable attacks={IMAGE_ATTACKS} data={IMAGE_DATA} />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Method comparison summary</h3>
        <SummaryTable rows={IMAGE_SUMMARY} />
      </div>
    </div>
  )
}

function AudioPanel() {
  return (
    <div className="space-y-6">
      <BestMethodCard tab="audio" />
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img src="/charts/chart_audio.png" alt="Audio attack simulation chart" className="w-full object-contain" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">BER per attack — 10 attacks × 2 methods</h3>
        <AttackTable attacks={AUDIO_ATTACKS} data={AUDIO_DATA} noteGANskipped />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Method comparison summary</h3>
        <SummaryTable rows={AUDIO_SUMMARY} />
      </div>
    </div>
  )
}

function VideoPanel() {
  return (
    <div className="space-y-6">
      <BestMethodCard tab="video" />
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img src="/charts/chart_video.png" alt="Video attack simulation chart" className="w-full object-contain" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">BER per attack — 9 attacks × 4 methods</h3>
        <AttackTable attacks={VIDEO_ATTACKS} data={VIDEO_DATA} />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Method comparison summary</h3>
        <SummaryTable rows={VIDEO_SUMMARY} />
      </div>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'image', label: '🖼 Image  (13 attacks)' },
  { id: 'audio', label: '🎵 Audio  (10 attacks)' },
  { id: 'video', label: '🎬 Video  (9 attacks)' },
]
const PANELS = { image: ImagePanel, audio: AudioPanel, video: VideoPanel }

export default function AttackResults() {
  const [activeTab, setActiveTab] = useState('image')
  const ActivePanel = PANELS[activeTab]

  return (
    <section className="space-y-8 max-w-5xl mx-auto">

      {/* Header */}
      <div className="text-center space-y-3">
        <div className="flex items-center justify-center gap-3">
          <ShieldAlert className="w-7 h-7 text-orange-400" />
          <h2 className="text-2xl font-bold">Attack Simulation</h2>
        </div>
        <p className="text-gray-400 text-sm max-w-2xl mx-auto leading-relaxed">
          Each steganography method was tested against real-world attacks — JPEG compression,
          noise, blurring, cropping, and more. We measure two things:
        </p>
        <div className="flex flex-wrap justify-center gap-4 text-xs">
          <span className="px-3 py-1.5 rounded-lg bg-blue-950/40 border border-blue-800/50 text-blue-300">
            <strong>PSNR</strong> — image quality after attack (higher = less damage)
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-purple-950/40 border border-purple-800/50 text-purple-300">
            <strong>BER</strong> — how many hidden bits survive (lower = more robust)
          </span>
        </div>
      </div>

      {/* BER legend */}
      <div className="flex flex-wrap items-center gap-2 justify-center">
        <span className="text-xs text-gray-500 mr-1">BER scale:</span>
        {[
          { label: '< 0.15 — Robust', cls: 'bg-emerald-950/60 text-emerald-400 border border-emerald-700/50' },
          { label: '0.15–0.35 — Moderate', cls: 'bg-yellow-950/60 text-yellow-300 border border-yellow-700/50' },
          { label: '0.35–0.49 — Fragile', cls: 'bg-orange-950/60 text-orange-400 border border-orange-700/50' },
          { label: '≥ 0.5 — Data lost', cls: 'bg-red-950/60 text-red-400 border border-red-700/50' },
        ].map(({ label, cls }) => (
          <span key={label} className={`text-xs font-medium px-3 py-1 rounded-full ${cls}`}>{label}</span>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {TABS.map(({ id, label }) => (
          <button key={id} onClick={() => setActiveTab(id)}
            className={`px-5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-all -mb-px ${
              activeTab === id
                ? 'border-blue-500 text-blue-400 bg-blue-950/30'
                : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/40'
            }`}>
            {label}
          </button>
        ))}
      </div>

      <div className="pt-2">
        <ActivePanel />
      </div>
    </section>
  )
}
