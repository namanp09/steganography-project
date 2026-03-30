import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export async function encodeSteganography(mediaType, formData) {
  const res = await api.post(`/${mediaType}/encode`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function decodeSteganography(mediaType, formData) {
  const res = await api.post(`/${mediaType}/decode`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function getMethods() {
  const res = await api.get('/methods')
  return res.data
}

export default api
