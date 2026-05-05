import React, { useState } from 'react'
import { ShieldAlert } from 'lucide-react'

// ─── Data ────────────────────────────────────────────────────────────────────

const IMAGE_ATTACKS = [
  'Baseline', 'JPEG Q90', 'JPEG Q75', 'JPEG Q50', 'Gauss σ5', 'Gauss σ15',
  'Salt & Pepper', 'Gauss Blur', 'Median', 'Resize', 'Brightness', 'Rotation', 'Crop',
]

const IMAGE_DATA = {
  LSB: [0.0, 0.52, 0.49, 0.51, 0.52, 0.51, 0.0, 0.54, 0.45, 0.48, 0.30, 0.49, 0.48],
  DCT: [0.0, 0.0, 0.10, 0.34, 0.13, 0.47, 0.24, 0.33, 0.45, 0.43, 0.08, 0.44, 0.47],
  DWT: [0.0, 0.18, 0.43, 0.49, 0.44, 0.51, 0.07, 0.44, 0.50, 0.44, 0.15, 0.50, 0.51],
  GAN: [0.375, 0.44, 0.375, 0.50, 0.375, 0.44, 0.375, 0.50, 0.31, 0.50, 0.375, 0.50, 0.44],
}

const IMAGE_SUMMARY = [
  { method: 'LSB', embedPSNR: 70.27, avgBER: 0.4403 },
  { method: 'DCT', embedPSNR: 46.87, avgBER: 0.2895 },
  { method: 'DWT', embedPSNR: 50.29, avgBER: 0.3888 },
  { method: 'GAN', embedPSNR: 45.67, avgBER: 0.4271 },
]

const AUDIO_ATTACKS = [
  'Baseline', 'AWGN 30dB', 'AWGN 20dB', 'LP 4kHz', 'LP 8kHz',
  'Vol ×0.8', 'Vol ×1.2', 'Resample', 'Time Shift', 'Echo',
]

const AUDIO_DATA = {
  LSB: [0.0, 0.49, 0.46, 0.52, 0.48, 0.49, 0.51, 0.46, 0.52, 0.46],
  DWT: [0.0, 0.39, 0.49, 0.52, 0.0, 0.53, 0.50, 0.0, 0.54, 0.50],
}

const AUDIO_SUMMARY = [
  { method: 'LSB', embedPSNR: 86.74, avgBER: 0.4874 },
  { method: 'DWT', embedPSNR: 49.92, avgBER: 0.3837 },
]

const VIDEO_ATTACKS = [
  'Baseline', 'Gauss Noise', 'JPEG Q75', 'JPEG Q50', 'Resize',
  'Frame Drop', 'Brightness', 'Gauss Blur', 'Salt & Pepper',
]

const VIDEO_DATA = {
  LSB: [0.0, 0.52, 0.50, 0.51, 0.49, 0.0, 0.32, 0.48, 0.01],
  DCT: [0.0, 0.02, 0.0, 0.0, 0.29, 0.0, 0.0, 1.0, 0.21],
  DWT: [0.0, 0.49, 0.09, 0.29, 0.25, 0.0, 0.0, 0.25, 0.06],
  GAN: [0.125, 0.125, 0.1875, 0.1875, 0.1875, 0.125, 0.125, 0.125, 0.125],
}

const VIDEO_SUMMARY = [
  { method: 'LSB', embedPSNR: 98.79, avgBER: 0.3538 },
  { method: 'DCT', embedPSNR: 97.76, avgBER: 0.1902 },
  { method: 'DWT', embedPSNR: 98.05, avgBER: 0.1794 },
  { method: 'GAN', embedPSNR: 34.03, avgBER: 0.1484 },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function berColor(ber) {
  if (ber < 0.1) return 'text-emerald-400 bg-emerald-950/40'
  if (ber < 0.3) return 'text-yellow-300 bg-yellow-950/30'
  if (ber < 0.5) return 'text-orange-400 bg-orange-950/30'
  return 'text-red-400 bg-red-950/40'
}

function berLabel(ber) {
  if (ber < 0.1) return 'robust'
  if (ber < 0.3) return 'moderate'
  if (ber < 0.5) return 'fragile'
  return 'random'
}

const METHOD_COLORS = {
  LSB: '#94a3b8',
  DCT: '#60a5fa',
  DWT: '#c084fc',
  GAN: '#34d399',
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function BerCell({ ber }) {
  const cls = berColor(ber)
  return (
    <td className="px-3 py-2 text-center">
      <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold ${cls}`}>
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
            <th className="px-3 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide whitespace-nowrap">
              Attack
            </th>
            {methods.map((m) => (
              <th
                key={m}
                className="px-3 py-3 text-center text-xs font-bold uppercase tracking-wide whitespace-nowrap"
                style={{ color: METHOD_COLORS[m] }}
              >
                {m} BER
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {attacks.map((atk, i) => (
            <tr
              key={atk}
              className={`border-b border-gray-800/60 last:border-0 ${
                i % 2 === 0 ? 'bg-gray-900/20' : 'bg-gray-900/40'
              }`}
            >
              <td className="px-3 py-2 text-xs text-gray-300 font-medium whitespace-nowrap">
                {atk}
              </td>
              {methods.map((m) => (
                <BerCell key={m} ber={data[m][i]} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {noteGANskipped && (
        <p className="px-4 py-2 text-xs text-gray-500 border-t border-gray-800">
          GAN skipped — checkpoint shape mismatch prevented inference during simulation.
        </p>
      )}
    </div>
  )
}

function SummaryTable({ rows }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800 mt-6">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 bg-gray-900/70">
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide">Method</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Embed PSNR (dB)</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Avg BER (all attacks)</th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wide">Robustness</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ method, embedPSNR, avgBER }, i) => {
            const color = METHOD_COLORS[method] ?? '#94a3b8'
            return (
              <tr
                key={method}
                className={`border-b border-gray-800/60 last:border-0 ${
                  i % 2 === 0 ? 'bg-gray-900/20' : 'bg-gray-900/40'
                }`}
              >
                <td className="px-4 py-3">
                  <span
                    className="text-xs font-bold px-2 py-0.5 rounded"
                    style={{ color, backgroundColor: `${color}18` }}
                  >
                    {method}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-xs font-mono text-gray-300">
                  {embedPSNR.toFixed(2)}
                </td>
                <BerCell ber={avgBER} />
                <td className="px-4 py-3 text-center">
                  <span className={`text-xs capitalize font-medium ${berColor(avgBER).split(' ')[0]}`}>
                    {berLabel(avgBER)}
                  </span>
                </td>
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
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img
          src="/charts/chart_image.png"
          alt="Image attack simulation BER chart"
          className="w-full object-contain"
        />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Per-attack BER — 13 attacks across 4 methods
        </h3>
        <AttackTable attacks={IMAGE_ATTACKS} data={IMAGE_DATA} />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Per-method averages</h3>
        <SummaryTable rows={IMAGE_SUMMARY} />
      </div>
    </div>
  )
}

function AudioPanel() {
  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img
          src="/charts/chart_audio.png"
          alt="Audio attack simulation BER chart"
          className="w-full object-contain"
        />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Per-attack BER — 10 attacks across 2 methods
        </h3>
        <AttackTable attacks={AUDIO_ATTACKS} data={AUDIO_DATA} noteGANskipped />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Per-method averages</h3>
        <SummaryTable rows={AUDIO_SUMMARY} />
      </div>
    </div>
  )
}

function VideoPanel() {
  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <img
          src="/charts/chart_video.png"
          alt="Video attack simulation BER chart"
          className="w-full object-contain"
        />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Per-attack BER — 9 attacks across 4 methods
        </h3>
        <AttackTable attacks={VIDEO_ATTACKS} data={VIDEO_DATA} />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Per-method averages</h3>
        <SummaryTable rows={VIDEO_SUMMARY} />
      </div>
    </div>
  )
}

// ─── Main export ──────────────────────────────────────────────────────────────

const TABS = [
  { id: 'image', label: 'Image', panel: ImagePanel },
  { id: 'audio', label: 'Audio', panel: AudioPanel },
  { id: 'video', label: 'Video', panel: VideoPanel },
]

export default function AttackResults() {
  const [activeTab, setActiveTab] = useState('image')
  const ActivePanel = TABS.find((t) => t.id === activeTab)?.panel ?? ImagePanel

  return (
    <section className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <ShieldAlert className="w-7 h-7 text-orange-400" />
          <h2 className="text-2xl font-bold">Attack Simulation — Robustness Evaluation</h2>
        </div>
        <p className="text-gray-400 text-sm max-w-2xl mx-auto leading-relaxed">
          We simulated 13 image attacks, 10 audio attacks, and 9 video attacks on each steganography
          method. Two metrics are reported: PSNR (higher = better signal quality) and BER — Bit Error
          Rate (lower = more robust; BER=0.5 means random noise).
        </p>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 justify-center">
        {[
          { label: 'BER < 0.1 — robust', cls: 'bg-emerald-950/60 text-emerald-400 border border-emerald-700/50' },
          { label: 'BER 0.1–0.3 — moderate', cls: 'bg-yellow-950/60 text-yellow-300 border border-yellow-700/50' },
          { label: 'BER 0.3–0.5 — fragile', cls: 'bg-orange-950/60 text-orange-400 border border-orange-700/50' },
          { label: 'BER ≥ 0.5 — random noise', cls: 'bg-red-950/60 text-red-400 border border-red-700/50' },
        ].map(({ label, cls }) => (
          <span key={label} className={`text-xs font-medium px-3 py-1 rounded-full ${cls}`}>
            {label}
          </span>
        ))}
        <span className="text-xs text-gray-500 border border-dashed border-red-600/50 text-red-500/80 px-3 py-1 rounded-full">
          BER = 0.5 → random noise (embed destroyed)
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 border-b border-gray-800 pb-0">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`px-5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-all -mb-px ${
              activeTab === id
                ? 'border-blue-500 text-blue-400 bg-blue-950/30'
                : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/40'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Active panel */}
      <div className="pt-2">
        <ActivePanel />
      </div>
    </section>
  )
}
