import { ImagePlus, RotateCcw, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export function ImageUploader({ previewUrl, onSelect, onReset, disabled }: { previewUrl: string | null; onSelect: (file: File) => void; onReset: () => void; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  function validate(file: File | undefined) { if (!file) return; if (!ACCEPTED.includes(file.type)) return setError('Use JPG, JPEG, PNG, or WEBP images.'); if (file.size > MAX_BYTES) return setError('Images must be smaller than 10 MB.'); setError(null); onSelect(file) }
  return <section className="uploader"><div className="section-kicker">01 / INPUT</div>{previewUrl ? <div className="preview-wrap"><img src={previewUrl} alt="Selected input preview" /><button className="icon-button preview-reset" onClick={onReset} title="Remove selected image"><RotateCcw size={17} /></button></div> : <button className="dropzone" disabled={disabled} onClick={() => inputRef.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); validate(event.dataTransfer.files[0]) }}><UploadCloud size={30} /><strong>Drop a visual input here</strong><span>or choose a JPG, PNG, or WEBP file</span></button>}<input ref={inputRef} hidden type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={event => validate(event.target.files?.[0])} />{error && <p className="field-error">{error}</p>}<div className="uploader-footer"><span><ImagePlus size={14} /> Zero-trust ingest</span>{previewUrl && <span className="file-ready">IMAGE READY</span>}</div></section>
}
