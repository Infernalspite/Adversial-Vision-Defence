import type { ArgusAnalysisResult, AttackResult, DetectionResult, DefenseResult, EvaluationSummary, ExperimentSummary, HardNegative, InferenceResult, TrustMapResult, VerificationResult, AuditRecord } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'

export async function submitInference(image: File): Promise<InferenceResult> {
  const formData = new FormData()
  formData.append('image', image)
  const response = await fetch(`${API_BASE_URL}/inference`, { method: 'POST', body: formData })
  if (!response.ok) throw new Error(`Inference request failed: ${response.status}`)
  return response.json() as Promise<InferenceResult>
}

async function postImage<T>(path: string, image: File, fields: Record<string, string | undefined> = {}): Promise<T> {
  const formData = new FormData()
  formData.append('image', image)
  Object.entries(fields).forEach(([key, value]) => { if (value !== undefined) formData.append(key, value) })
  const response = await fetch(`${API_BASE_URL}${path}`, { method: 'POST', body: formData })
  if (!response.ok) throw new Error(`Request failed: ${response.status}`)
  return response.json() as Promise<T>
}

export function analyzeImage(image: File, mode = 'standard'): Promise<ArgusAnalysisResult> {
  return postImage<ArgusAnalysisResult>('/analyze', image, { mode })
}

export function generateAttack(image: File, fields: Record<string, string | undefined>): Promise<AttackResult> {
  return postImage<AttackResult>('/attacks/generate', image, fields)
}

export function evaluateAttack(image: File, fields: Record<string, string | undefined>): Promise<AttackResult> {
  return postImage<AttackResult>('/attacks/evaluate', image, fields)
}

export function analyzeDetection(image: File): Promise<DetectionResult> {
  return postImage<DetectionResult>('/detection/analyze', image)
}

export function analyzeTrustMap(image: File): Promise<TrustMapResult> {
  return postImage<TrustMapResult>('/trust-map/analyze', image)
}

export function applyDefense(image: File): Promise<DefenseResult> {
  return postImage<DefenseResult>('/defense/apply', image)
}

export function analyzeVerification(originalImage: File, defendedImage: File): Promise<VerificationResult> {
  const formData = new FormData()
  formData.append('original_image', originalImage)
  formData.append('defended_image', defendedImage)
  return fetch(`${API_BASE_URL}/verification/analyze`, { method: 'POST', body: formData }).then(async response => {
    if (!response.ok) throw new Error(`Request failed: ${response.status}`)
    return response.json() as Promise<VerificationResult>
  })
}

export async function getAuditRecords(limit = 20): Promise<AuditRecord[]> {
  const response = await fetch(`${API_BASE_URL}/audit?limit=${limit}`)
  if (!response.ok) throw new Error(`Audit request failed: ${response.status}`)
  return response.json() as Promise<AuditRecord[]>
}

export async function getAuditRecord(requestId: string): Promise<AuditRecord[]> {
  const response = await fetch(`${API_BASE_URL}/audit/${encodeURIComponent(requestId)}`)
  if (!response.ok) throw new Error(`Audit request failed: ${response.status}`)
  return response.json() as Promise<AuditRecord[]>
}

export async function getEvaluationSummary(): Promise<EvaluationSummary> {
  const response = await fetch(`${API_BASE_URL}/evaluation/summary`)
  if (!response.ok) throw new Error(`Evaluation summary unavailable: ${response.status}`)
  return response.json() as Promise<EvaluationSummary>
}

export async function getExperiments(): Promise<ExperimentSummary[]> { const response = await fetch(`${API_BASE_URL}/learning/experiments`); if (!response.ok) throw new Error(`Experiments unavailable: ${response.status}`); return response.json() as Promise<ExperimentSummary[]> }
export async function getHardNegatives(): Promise<{ count: number; items: HardNegative[] }> { const response = await fetch(`${API_BASE_URL}/learning/hard-negatives`); if (!response.ok) throw new Error(`Hard negatives unavailable: ${response.status}`); return response.json() as Promise<{ count: number; items: HardNegative[] }> }

export async function getHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new Error(`Health request failed: ${response.status}`)
  return response.json() as Promise<{ status: string }>
}
