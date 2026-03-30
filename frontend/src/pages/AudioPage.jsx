import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { Lock, Unlock, Loader2 } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import { encodeSteganography, decodeSteganography } from '../utils/api'

const METHODS = [
  { id: 'lsb', label: 'LSB', desc: 'Audio sample LSB' },
  { id: 'dwt', label: 'DWT', desc: 'Wavelet transform (Daubechies-4)' },
]

export default function AudioPage() {
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
      const data = await fn('audio', formData)
      setResult(data)
      toast.success(mode === 'encode' ? 'Message hidden in audio!' : 'Message extracted!')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Operation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Audio Steganography</h1>
        <p className="text-gray-400">Hide encrypted messages within audio files</p>
      </div>

      <div className="flex justify-center gap-2 p-1 bg-gray-900 rounded-xl w-fit mx-auto">
        {['encode', 'decode'].map((m) => (
          <button key={m} onClick={() => { setMode(m); setResult(null) }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
              mode === m ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
            }`}>
            {m === 'encode' ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        <FileUpload file={file} onFileSelect={setFile} accept={{ 'audio/*': ['.wav', '.flac'] }}
          label={mode === 'encode' ? 'Cover Audio' : 'Stego Audio'} />

        {mode === 'encode' && (
          <div>
            <label className="text-sm font-medium text-gray-300">Secret Message</label>
            <textarea value={message} onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your secret message..."
              className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-purple-500 focus:outline-none resize-none h-28" />
          </div>
        )}

        <div>
          <label className="text-sm font-medium text-gray-300">Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            placeholder="Encryption password..."
            className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-purple-500 focus:outline-none" />
        </div>

        <div>
          <label className="text-sm font-medium text-gray-300">Method</label>
          <div className="grid grid-cols-2 gap-2 mt-1">
            {METHODS.map(({ id, label, desc }) => (
              <button key={id} onClick={() => setMethod(id)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  method === id ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/50'
                }`}>
                <p className="text-sm font-semibold">{label}</p>
                <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
              </button>
            ))}
          </div>
        </div>

        <button onClick={handleSubmit} disabled={loading}
          className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded-xl font-medium transition-all flex items-center justify-center gap-2">
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : null}
          {loading ? 'Processing...' : mode === 'encode' ? 'Hide Message' : 'Extract Message'}
        </button>

        {result && mode === 'encode' && (
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <p className="text-green-400 font-medium">Message hidden successfully!</p>
            <a href={result.output_file} download className="text-sm text-purple-400 hover:underline">
              Download stego audio
            </a>
            <p className="text-xs text-gray-500 mt-1">Time: {result.time_ms}ms</p>
          </div>
        )}

        {result && mode === 'decode' && (
          <div className="bg-gray-800/50 rounded-xl p-6 border border-green-500/30">
            <p className="text-sm text-gray-400 mb-2">Extracted Message</p>
            <p className="text-lg font-medium text-green-400 break-words">{result.message}</p>
          </div>
        )}
      </div>
    </div>
  )
}
