import { UploadCloud } from 'lucide-react'

export function ImageUploader() {
  return <section className="rounded-lg border border-dashed border-cyan-400/40 bg-slate-900/70 p-8 text-center"><UploadCloud className="mx-auto text-cyan-300" size={32} /><h2 className="mt-3 text-lg font-bold text-white">Awaiting visual input</h2><p className="mt-1 text-sm text-slate-400">Upload flow is scaffolded; ingest validation will be connected next.</p><button type="button" disabled className="mt-5 rounded-md bg-cyan-400/20 px-4 py-2 text-sm font-semibold text-cyan-200">Choose image</button></section>
}
