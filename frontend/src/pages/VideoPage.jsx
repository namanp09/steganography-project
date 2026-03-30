import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { Lock, Unlock, Loader2, Star } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import { encodeSteganography, decodeSteganography } from '../utils/api'

const METHODS = [
  { id: 'lsb', label: 'LSB', desc: 'Frame-level LSB + motion compensation' },
  { id: 'dct', label: 'DCT', desc: 'DCT on frames with temporal awareness' },
  { id: 'dwt', label: 'DWT', desc: 'DWT with frame selection' },
]

export default function VideoPage() {
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
      const data = await fn('video', formData)
      setResult(data)
      toast.success(mode === 'encode' ? 'Message hidden in video!' : 'Message extracted!')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Operation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <h1 className="text-3xl font-bold">Video Steganography</h1>
          <span className="flex items-center gap-1 px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs font-medium rounded-full">
            <Star className="w-3 h-3" /> Primary Focus
          </span>
        </div>
        <p className="text-gray-400">Hide encrypted messages within video files with motion-compensated embedding</p>
      </div>

      <div className="flex justify-center gap-2 p-1 bg-gray-900 rounded-xl w-fit mx-auto">
        {['encode', 'decode'].map((m) => (
          <button key={m} onClick={() => { setMode(m); setResult(null) }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
              mode === m ? 'bg-orange-600 text-white' : 'text-gray-400 hover:text-white'
            }`}>
            {m === 'encode' ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        <FileUpload file={file} onFileSelect={setFile} accept={{ 'video/*': ['.mp4', '.avi', '.mkv'] }}
          label={mode === 'encode' ? 'Cover Video' : 'Stego Video'} />

        {mode === 'encode' && (
          <div>
            <label className="text-sm font-medium text-gray-300">Secret Message</label>
            <textarea value={message} onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your secret message..."
              className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-orange-500 focus:outline-none resize-none h-28" />
          </div>
        )}

        <div>
          <label className="text-sm font-medium text-gray-300">Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            placeholder="Encryption password..."
            className="w-full mt-1 p-3 bg-gray-800 border border-gray-700 rounded-xl text-sm focus:border-orange-500 focus:outline-none" />
        </div>

        <div>
          <label className="text-sm font-medium text-gray-300">Method</label>
          <div className="grid grid-cols-3 gap-2 mt-1">
            {METHODS.map(({ id, label, desc }) => (
              <button key={id} onClick={() => setMethod(id)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  method === id ? 'border-orange-500 bg-orange-500/10' : 'border-gray-700 bg-gray-800/50'
                }`}>
                <p className="text-sm font-semibold">{label}</p>
                <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
              </button>
            ))}
          </div>
        </div>

        <button onClick={handleSubmit} disabled={loading}
          className="w-full py-3 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 rounded-xl font-medium transition-all flex items-center justify-center gap-2">
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : null}
          {loading ? 'Processing...' : mode === 'encode' ? 'Hide Message' : 'Extract Message'}
        </button>

        {result && mode === 'encode' && (
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700 space-y-3">
            <p className="text-green-400 font-medium">Message hidden in video!</p>
            <a href={result.output_file} download className="text-sm text-orange-400 hover:underline block">
              Download stego video
            </a>
            {result.info && (
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                <p>Frames used: {result.info.frames_used}</p>
                <p>Total frames: {result.info.total_frames}</p>
                <p>Capacity: {result.info.capacity_bytes} bytes</p>
                <p>Data size: {result.info.data_size_bytes} bytes</p>
              </div>
            )}
            <p className="text-xs text-gray-500">Time: {result.time_ms}ms</p>
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
