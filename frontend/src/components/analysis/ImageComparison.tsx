import type { ArgusAnalysisResult } from '../../types'

export function ImageComparison({ originalUrl, result }: { originalUrl: string | null; result: ArgusAnalysisResult | null }) {
  return <section className="panel comparison-panel"><div className="panel-heading"><div><span className="section-kicker">02 / VISUAL TRACE</span><h2>Input versus response</h2></div><span className="panel-note">{result ? 'DEFENDED OUTPUT' : 'AWAITING ANALYSIS'}</span></div><div className="comparison-grid"><div className="image-frame"><span className="image-label">ORIGINAL INPUT</span>{originalUrl ? <img src={originalUrl} alt="Original uploaded input" /> : <div className="image-empty">No image selected</div>}</div><div className="image-frame"><span className="image-label">DEFENDED IMAGE</span>{result ? <img src={result.defended_image_reference} alt="Defended output" /> : <div className="image-empty">Run analysis to generate</div>}</div></div></section>
}
