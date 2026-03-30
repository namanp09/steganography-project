import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function MetricsDisplay({ metrics }) {
  if (!metrics) return null

  const data = Object.entries(metrics).map(([key, value]) => ({
    name: key,
    value: typeof value === 'number' ? value : 0,
  }))

  const getColor = (name, value) => {
    if (name.includes('PSNR')) return value > 35 ? '#22c55e' : value > 30 ? '#eab308' : '#ef4444'
    if (name.includes('SSIM')) return value > 0.97 ? '#22c55e' : value > 0.95 ? '#eab308' : '#ef4444'
    if (name.includes('BER')) return value === 0 ? '#22c55e' : '#ef4444'
    return '#3b82f6'
  }

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">Quality Metrics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} className="bg-gray-900/50 rounded-lg p-3 text-center">
            <p className="text-xs text-gray-400 uppercase tracking-wide">{key}</p>
            <p className={`text-xl font-bold mt-1`} style={{ color: getColor(key, value) }}>
              {typeof value === 'number' ? (value < 0.01 ? value.toExponential(2) : value.toFixed(4)) : value}
            </p>
          </div>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
          <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
