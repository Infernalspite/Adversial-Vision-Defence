import { useState } from 'react'
import { analyzeImage } from '../services/api'
import type { ArgusAnalysisResult } from '../types'

export function useAnalysis() {
  const [result, setResult] = useState<ArgusAnalysisResult | null>(null)
  const [image, setImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function selectImage(file: File | null) {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImage(file)
    setPreviewUrl(file ? URL.createObjectURL(file) : null)
    setResult(null)
    setError(null)
  }

  async function runAnalysis() {
    if (!image) return
    setLoading(true)
    setError(null)
    try { setResult(await analyzeImage(image)) } catch (cause) { setError(cause instanceof Error ? cause.message : 'The backend could not complete analysis.') } finally { setLoading(false) }
  }

  function reset() { selectImage(null) }
  function adoptResult(nextResult: ArgusAnalysisResult) { setResult(nextResult); setError(null) }
  return { image, previewUrl, result, loading, error, selectImage, runAnalysis, reset, adoptResult }
}
