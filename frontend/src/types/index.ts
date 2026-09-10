export type FinalState = 'TRUSTED' | 'DEFENDED' | 'ABSTAIN'

export interface InferenceResult {
  request_id: string
  original_prediction: string | null
  original_confidence: number | null
  attack_detected: boolean
  attack_score: number
  attack_type: string | null
  trust_map: Record<string, unknown> | null
  defense_applied: boolean
  defense_method: string | null
  defended_prediction: string | null
  defended_confidence: number | null
  semantic_verification_score: number | null
  final_state: FinalState
  explanation: string
  processing_time: number | null
}

export interface AuditRecord {
  request_id: string
  stage: string
  message: string
  timestamp: string
  details: Record<string, unknown>
}

export type AuditEvent = AuditRecord

export interface DetectorResult {
  detector_name: string
  score: number
  detected: boolean
  confidence: number
  evidence: Record<string, unknown>
  processing_time_ms: number
  metadata: Record<string, unknown>
}

export interface DetectionResult {
  attack_score: number
  attack_detected: boolean
  detection_threshold: number
  detectors: Record<string, DetectorResult>
}

export interface AttackResult {
  request_id: string
  attack_type: string
  attack_parameters: Record<string, unknown>
  original_prediction: { class_id: number; class_name: string; confidence: number }
  original_confidence: number
  adversarial_prediction: { class_id: number; class_name: string; confidence: number }
  adversarial_confidence: number
  attack_success: boolean
  perturbation_magnitude: number
  generation_time_ms: number
  adversarial_image: string
  patch_mask: string | null
}

export interface SuspiciousRegion {
  region_id: number
  x: number
  y: number
  width: number
  height: number
  area_pixels: number
  area_percentage: number
  anomaly_score: number
  trust_score: number
}

export interface TrustMapResult {
  request_id: string
  trust_map: {
    width: number
    height: number
    map_reference: string
    overlay_reference: string
    resolution: [number, number]
    normalization: Record<string, unknown>
    contributing_detectors: string[]
  }
  global_trust_score: number
  suspicious_regions: SuspiciousRegion[]
  suspicious_area_percentage: number
  spatial_evidence: {
    saliency_available: boolean
    frequency_available: boolean
    local_anomaly_available: boolean
    patch_mask_available: boolean
  }
  processing_time_ms: number
}

export interface DefenseTrace {
  timestamp: string
  attack_score: number
  selected_defense: string
  reason: string
  input_trust_score: number
  suspicious_regions: number
  defense_parameters: Record<string, unknown>
  processing_time_ms: number
  success: boolean
  output_image_reference: string | null
}

export interface DefenseResult {
  request_id: string
  original_prediction: { class_id: number; class_name: string; confidence: number }
  original_confidence: number
  attack_score: number
  attack_detected: boolean
  global_trust_score: number
  suspicious_regions: SuspiciousRegion[]
  selected_defense: string
  defense_reason: string
  defense_parameters: Record<string, unknown>
  defense_applied: boolean
  defended_prediction: { class_id: number; class_name: string; confidence: number }
  defended_confidence: number
  prediction_changed: boolean
  defended_image_reference: string
  defense_trace: DefenseTrace
  processing_time_ms: number
}

export interface ConsistencyResult {
  score: number
  evidence: Record<string, unknown>
}

export interface VerificationResult {
  request_id: string
  original_prediction: { class_id: number; class_name: string; confidence: number }
  original_confidence: number
  defended_prediction: { class_id: number; class_name: string; confidence: number }
  defended_confidence: number
  prediction_changed: boolean
  confidence_change: number
  top_k_overlap: number
  object_consistency: ConsistencyResult
  geometry_consistency: ConsistencyResult
  scene_consistency: ConsistencyResult
  verification_score: number
  explanation: string
  processing_time_ms: number
}

export type DecisionState = 'TRUSTED' | 'DEFENDED' | 'ABSTAIN'

export interface DecisionReasons {
  state: DecisionState
  reasons: string[]
}

export interface PixelTraceSummary {
  suspicion_mean: number
  suspicion_max: number
  suspicious_pixel_fraction: number
  num_suspicious_regions: number
  largest_region_area_fraction: number
  trust_resolution: Record<string, number>
}

export interface PersonBox {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
}

export interface PersonDetectionSummary {
  detector_available: boolean
  present: boolean
  count: number
  max_confidence: number
  boxes: PersonBox[]
  inference_time_ms: number
}

export interface ArgusAnalysisResult {
  request_id: string
  tier: 'baseline' | 'robust'
  tier_reason: string
  matched_defended_class: string | null
  person_detection: PersonDetectionSummary | null
  suspicion_heatmap_reference: string
  pixel_trace: PixelTraceSummary
  attack_score: number
  attack_detected: boolean
  attack_detection_threshold: number
  detectors: Record<string, DetectorResult>
  global_trust_score: number
  trust_map_reference: string
  trust_map_overlay_reference: string
  suspicious_regions: SuspiciousRegion[]
  defense_applied: boolean
  defense_method: string
  defended_image_reference: string
  original_prediction: { class_id: number; class_name: string; confidence: number }
  defended_prediction: { class_id: number; class_name: string; confidence: number }
  original_confidence: number
  defended_confidence: number
  prediction_changed: boolean
  object_consistency: ConsistencyResult
  geometry_consistency: ConsistencyResult
  scene_consistency: ConsistencyResult
  verification_score: number
  final_state: DecisionState
  final_prediction: string | null
  final_confidence: number | null
  decision_score: number
  decision_reasons: string[]
  explanation: string
  fallback_action: string | null
  fallback_reason: string | null
  defense_trace: DefenseTrace
  audit: AuditEvent[]
  processing_time_ms: number
}

export interface EvaluationSummary {
  dataset: Record<string, number>
  categories: Record<string, {
    attack: Record<string, number>
    detection: Record<string, number>
    defense: Record<string, number>
    decisions: Record<string, number>
    localization: Record<string, number>
    latency: Record<string, number>
  }>
  combined: Record<string, Record<string, number>>
}

export interface HardNegative { request_id: string; category: string; explanation: string; timestamp?: string; attack_type?: string | null; attack_score?: number | null; trust_score?: number | null; verification_score?: number | null; final_state?: DecisionState | null; expected_behavior: string }
export interface ExperimentSummary { experiment_id: string; timestamp: string; attack_type: string; generations: number; population_size: number; seed: number }
