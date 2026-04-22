import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { Lock, Unlock, Loader2 } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import MetricsDisplay from '../components/MetricsDisplay'
import { encodeSteganography, decodeSteganography } from '../utils/api'

const METHODS = [
  { id: 'lsb', label: 'LSB', desc: 'Least Significant Bit (baseline)' },
  { id: 'dct', label: 'DCT', desc: 'Discrete Cosine Transform + QIM' },
  { id: 'dwt', label: 'DWT', desc: 'Discrete Wavelet Transform' },
  { id: 'gan', label: 'GAN', desc: 'Adaptive Cost Learning GAN (modern)' },
]

export default function ImagePage() {
  const [mode, setMode] = useState('encode')
  const [file, setFile] = useState(null)
  const [message, setMessage] = useState('')
  const [password, setPassword] = useState('')
  const [method, setMethod] = useState('lsb')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleSubmit = async () => {
    if (!file || !password) return toast.error('Upload a file and enter a password')
    if (mode === 'encode' && !message) return toast.error('Enter a secret message')

    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append(mode === 'encode' ? 'cover' : 'stego', file)
    formData.append('password', password)
    formData.append('method', method)
    if (mode === 'encode') formData.append('message', message)

    try {
      const fn = mode === 'encode' ? encodeSteganography : decodeSteganography
      const data = await fn('image', formData)
      setResult(data)
      toast.success(mode === 'encode' ? 'Message hidden successfully!' : 'Message extracted!')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Operation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Image Steganography</h1>
        <p className="text-gray-400">Hide encrypted messages within images</p>
      </div>

      {/* Mode Toggle */}
      <div className="flex justify-center gap-2 p-1 bg-gray-900 rounded-xl w-fit mx-auto">
        {['encode', 'decode'].map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); setResult(null) }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
              mode === m ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            {m === 'encode' ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Left: Inputs */}
        <div className="space-y-4">
          <FileUpload
            file={file}
            onFileSelect={setFile}
            accept={mode === 'encode'
              ? { 'image/*': ['.png', '.bmp', '.tiff', '.jpg'] }
              : { 'image/png': ['.png'], 'image/bmp': ['.bmp'], 'image/tiff': ['.tiff'] }
            }
            label={mode === 'encode' ? 'Cover Image' : 'Stego Image (PNG only — JPEG destroys hidden data)'}
          />

          {mode === 'encode' && (
            <div>
              <label className="text-sm font-medium text-gray-300">Secret Message</label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Enter your secret message..."
                className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-blue-500 focus:outline-none resize-none h-28"
              />
            </div>
          )}

          <div>
            <label className="text-sm font-medium text-gray-300">Password (AES-256-GCM)</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Encryption password..."
              className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>

          {/* Method Selection */}
          <div>
            <label className="text-sm font-medium text-gray-300">Method</label>
            <div className="grid grid-cols-2 gap-2 mt-1">
              {METHODS.map(({ id, label, desc }) => (
                <button
                  key={id}
                  onClick={() => setMethod(id)}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    method === id
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                  }`}
                >
                  <p className="text-sm font-semibold">{label}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-xl font-medium transition-all flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : mode === 'encode' ? <Lock className="w-5 h-5" /> : <Unlock className="w-5 h-5" />}
            {loading ? 'Processing...' : mode === 'encode' ? 'Hide Message' : 'Extract Message'}
          </button>
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          {result && mode === 'encode' && (
            <>
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <p className="text-sm text-gray-400 mb-2">Stego Image</p>
                <img src={result.output_file} alt="Stego" className="w-full rounded-lg" />
                <a href={result.output_file} download className="mt-3 block text-center text-sm text-blue-400 hover:underline">
                  Download
                </a>
              </div>
              <MetricsDisplay metrics={result.metrics} />
              <div className="text-xs text-gray-500">
                Time: {result.time_ms}ms | Hash: {result.data_hash?.slice(0, 16)}...
              </div>
            </>
          )}

          {result && mode === 'decode' && (
            <div className="bg-gray-800/50 rounded-xl p-6 border border-green-500/30">
              <p className="text-sm text-gray-400 mb-2">Extracted Message</p>
              <p className="text-lg font-medium text-green-400 break-words">{result.message}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
