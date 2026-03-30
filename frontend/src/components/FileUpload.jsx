import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X } from 'lucide-react'

export default function FileUpload({ file, onFileSelect, accept, label }) {
  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) onFileSelect(accepted[0])
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
  })

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-300">{label}</label>
      {file ? (
        <div className="flex items-center gap-3 p-4 bg-gray-800 rounded-xl border border-gray-700">
          <div className="flex-1 truncate">
            <p className="text-sm font-medium text-white truncate">{file.name}</p>
            <p className="text-xs text-gray-400">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
          <button onClick={() => onFileSelect(null)} className="p-1 hover:bg-gray-700 rounded">
            <X className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={`p-8 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${
            isDragActive
              ? 'border-blue-400 bg-blue-500/10'
              : 'border-gray-700 hover:border-gray-500 bg-gray-800/50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-8 h-8 mx-auto mb-2 text-gray-500" />
          <p className="text-sm text-gray-400">
            {isDragActive ? 'Drop file here...' : 'Drag & drop or click to select'}
          </p>
        </div>
      )}
    </div>
  )
}
