# Architecture

ARGUS-AEGIS is organized as ports and adapters around a staged visual trust pipeline. API routes accept transport-level input and return Pydantic contracts. Domain packages own model, attack, detector, trust, defense, verification, pipeline, and audit behavior.

```mermaid
flowchart TB
    Client[React Dashboard / Attack Lab] --> API[FastAPI API]
    API --> Ingest[Zero-Trust Ingest + Sanitization]
    Ingest --> Model[BaseVisionModel]
    Model --> Detectors[BaseDetector implementations]
    Detectors --> Score[UnifiedAttackScorer]
    Score --> Map[TrustMapBuilder]
    Map --> Locate[AttackLocalizer]
    Locate --> Orchestrator[DefenseOrchestrator]
    Orchestrator --> Strategy[DefenseStrategy]
    Strategy --> ReModel[BaseVisionModel re-inference]
    ReModel --> Anchor[SemanticRealityAnchor]
    Anchor --> Fusion[TrustFusion]
    Fusion --> State{FinalState}
    State --> Audit[AuditLogger]
```

## Contracts

- `BaseVisionModel.predict(image) -> ModelPrediction`
- `BaseAttack.generate(image, target) -> AttackResult`
- `BaseDetector.detect(image, context) -> DetectorResult`
- `DefenseStrategy.defend(image, trust_map, context) -> DefenseResult`
- `SemanticRealityAnchor.verify(original, defended, context) -> VerificationResult`
- `TrustFusion.fuse(signals) -> TrustFusionResult`
- `AuditLogger.record(event) -> None`

## State Model

Externally visible states are exactly `TRUSTED`, `DEFENDED`, and `ABSTAIN`. Internal stages are represented by `PipelineStage` for traceability and audit correlation.

## Security Boundaries

Uploaded images must be treated as untrusted bytes. Future ingest work must enforce size, MIME, decoding, dimension, and resource limits before model execution. Audit records must be structured, redactable, and append-only before production deployment.

Detector, defense, trust-map, and semantic-verification algorithms remain deferred to later phases.

## Phase 2 Attack Lab

Phase 2 adds a separate attack engine around the baseline model. Attacks operate on RGB pixel tensors and use the same differentiable resize, center crop, and ImageNet normalization path as inference.

```mermaid
flowchart TB
    Original[Original Image] --> Base[Base Model]
    Base --> Clean[Clean Result]
    Clean --> Engine[Attack Engine]
    Engine --> FGSM[FGSM]
    Engine --> PGD[PGD]
    Engine --> Patch[Adversarial Patch]
    FGSM --> Adv[Adversarial Image]
    PGD --> Adv
    Patch --> Adv
    Adv --> AdvModel[Base Model]
    AdvModel --> AdvResult[Adversarial Result]
```

The attack registry creates `fgsm`, `pgd`, and `adversarial_patch` adapters. Generated images are returned as temporary data references by the API and are not persisted. The CLI demo may save a local generated artifact under `data/adversarial/`.

## Phase 3 Detection

```mermaid
flowchart TB
    Input[Input Image] --> Base[Base Model]
    Base --> FS[Feature Squeezing]
    Base --> FA[Frequency Analysis]
    Base --> CI[Confidence Instability]
    Base --> SA[Saliency Analysis]
    FS --> Results[Detector Results]
    FA --> Results
    CI --> Results
    SA --> Results
    Results --> Score[Unified Attack Score]
    Score --> Decision{Detection Threshold}
    Decision -->|below| Normal[Evidence below MVP threshold]
    Decision -->|above| Suspicious[Evidence above MVP threshold]
```

Each detector provides evidence independently. `UnifiedAttackScorer` performs transparent configurable weighted fusion; the default weights are feature squeezing `0.30`, frequency `0.25`, confidence instability `0.25`, and saliency `0.20`, with a configurable threshold of `0.70`. These are MVP calibration parameters, not mathematically guaranteed detection limits. Phase 3 does not choose defenses or final trust states.

## Phase 4 Spatial Trust Map

```mermaid
flowchart TB
    Evidence[Detector Evidence] --> Saliency[Saliency]
    Evidence --> Frequency[Frequency]
    Evidence --> Local[Local Anomaly]
    Saliency --> Fusion[Spatial Fusion]
    Frequency --> Fusion
    Local --> Fusion
    Patch[Optional Patch Evidence] --> Fusion
    Fusion --> Anomaly[Spatial Anomaly]
    Anomaly --> Trust[1 - Anomaly]
    Trust --> Map[Trust Map]
    Map --> Regions[Attack Localization]
    Regions --> Suspicious[Suspicious Regions]
    Map --> Global[Global Trust Score]
```

Attack Score and Trust Map are different outputs. Attack Score is global evidence of adversarial manipulation from Phase 3. Trust Map is a spatial representation of relative perception confidence, where higher values are more trusted. Localization extracts connected low-trust regions from that map. None of these values are mathematically certified probabilities or final trust states.

## Phase 7 Final Decision

```mermaid
flowchart TB
    Input[Input] --> Model[Base Vision Model]
    Model --> Detection[Attack Detection]
    Detection --> Score[Attack Score]
    Score --> Trust[Spatial Trust Map]
    Trust --> Defense[Defense]
    Defense --> ReInference[Re-Inference]
    ReInference --> Verification[Semantic Verification]
    Verification --> Decision[Final Decision Engine]
    Decision --> Trusted[TRUSTED]
    Decision --> Defended[DEFENDED]
    Decision --> Abstain[ABSTAIN]
```

`TRUSTED` requires low attack risk, high global trust, strong verification, and sufficient original confidence. `DEFENDED` requires detected attack evidence, an applied defense, strong verification, and sufficient defended confidence. All contradictory, uncertain, or insufficient-evidence cases map to `ABSTAIN`, which is a deliberate safety mechanism and returns a `NO_ACTION` fallback.

| Condition | Result |
| --- | --- |
| Low attack risk + high trust + strong verification | TRUSTED |
| Attack detected + defense applied + strong verification | DEFENDED |
| Attack detected + weak defense result | ABSTAIN |
| High uncertainty | ABSTAIN |
| Conflicting evidence | ABSTAIN |
| Insufficient confidence | ABSTAIN |

The decision score is a transparent average of inverse attack risk, global trust, verification score, and selected prediction confidence. It is distinct from both Attack Score and Verification Score.

## Phase 10 Offline Learning Loop

```mermaid
flowchart LR
    Audit[Audit Records] --> Failures[Failure Analysis]
    Failures --> Hard[Hard Negatives]
    Hard --> Dataset[Candidate Dataset]
    Dataset --> Evolution[Bounded Attack Evolution]
    Evolution --> Evaluation[Offline Evaluation]
    Evaluation --> Gate[Validation Gate]
    Gate --> Candidate[Candidate Artifact]
    Candidate -->|explicit future promotion| Production[Production]
```

This loop is offline and candidate-only. It does not run during production inference, does not silently tune thresholds, and does not overwrite production models.

## Final Integrated Pipeline

```mermaid
flowchart TB
    Input[Image] --> Zero[Zero-Trust Ingest]
    Zero --> Sanitize[Sanitized RGB Representation]
    Sanitize --> Model[Base Vision Model]
    Model --> Triage[Adversarial Triage]
    Triage --> Score[Unified Attack Score]
    Score --> Trust[Spatial Trust Map]
    Trust --> Localize[Attack Localization]
    Localize --> Orchestrator[Defense Orchestrator]
    Orchestrator --> Defense[Defense Transformation]
    Defense --> Reinfer[Re-Inference]
    Reinfer --> Anchor[Semantic Reality Anchor]
    Anchor --> Verify[Final Verification]
    Verify --> Decision[Decision Engine]
    Decision --> Trusted[TRUSTED]
    Decision --> Defended[DEFENDED]
    Decision --> Abstain[ABSTAIN]
    Decision --> Audit[Audit Record]
    Audit --> Learning[Offline Learning Loop]
    Learning --> Evolution[Bounded Attack Evolution]
    Evolution --> Validation[Offline Validation]
```

Spatial fusion defaults to saliency `0.30`, frequency `0.25`, local anomaly `0.20`, and patch `0.25`. If a source is unavailable, its weight is removed and the remaining weights are renormalized. Patch masks are optional evaluation metadata and are not required for normal analysis.

## Phase 5 Autonomous Defense

```mermaid
flowchart TB
    Detection[Phase 3 Attack Detection] --> Score[Attack Score]
    Detection --> Evidence[Attack Evidence]
    Score --> Policy[Defense Policy]
    Evidence --> Policy
    Trust[Phase 4 Trust Map] --> Regions[Suspicious Regions]
    Regions --> Policy
    Policy --> Transform[Transform]
    Policy --> Mask[Mask]
    Policy --> Purify[Purification]
    Transform --> Defended[Defended Image]
    Mask --> Defended
    Purify --> Defended
    Defended --> ReInference[Base Model Re-Inference]
```

Detection identifies global evidence, the Trust Map identifies spatial risk, policy chooses the least destructive appropriate response, the defense modifies the input, and re-inference measures the result. The policy is transparent and configurable, not guaranteed optimal. Certified inference is represented only as an inactive roadmap placeholder.

## Phase 6 Semantic Reality Anchor

```mermaid
flowchart TB
    Defended[Defended Image] --> Anchor[Semantic Reality Anchor]
    Anchor --> Object[Object Consistency]
    Anchor --> Geometry[Geometry Consistency]
    Anchor --> Scene[Scene Consistency]
    Object --> Fusion[Verification Fusion]
    Geometry --> Fusion
    Scene --> Fusion
    Fusion --> Score[Verification Score]
    Score --> Final[Phase 7 Final Decision Layer]
```

Attack Score asks how much global evidence suggests adversarial manipulation. The Spatial Trust Map asks where perception may be unreliable. Verification Score asks how consistent the defended interpretation remains with available classification, structural, and coarse scene evidence. These signals remain separate, and none is mathematically guaranteed semantic truth.
