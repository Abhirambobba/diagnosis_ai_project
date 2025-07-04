'use client'

import { useState } from 'react'

// Helper to strip markdown (you can replace this with react-markdown later if needed)
function stripMarkdown(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, '$1')           // bold
    .replace(/\*([^*]+)\*/g, '$1')               // italic
    .replace(/`([^`]+)`/g, '$1')                 // inline code
    .replace(/#+\s/g, '')                        // headings
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')     // links
    .replace(/>\s+/g, '')                        // blockquotes
    .replace(/- /g, '')                          // list dashes
    .trim()
}

export default function Home() {
  const [prompt, setPrompt] = useState('')
  const [file, setFile] = useState(null)
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    const formData = new FormData()
    formData.append('prompt', prompt)
    if (file) formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/diagnose', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()

      if (data.summary) {
        data.summary = stripMarkdown(data.summary)
      }

      setResponse(data)
    } catch (err) {
      console.error('❌ API error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-white p-8 grid grid-cols-3 gap-6">
      {/* Left Panel */}
      <div className="flex flex-col col-span-1">
        <h1 className="text-3xl font-bold text-blue-800 mb-6">🩺 Diagnosis Assistant</h1>
        <form onSubmit={handleSubmit} className="space-y-4 bg-blue-50 p-4 rounded-xl shadow-md">
          <label className="block text-sm font-semibold text-gray-700">Upload Medical Report (PDF)</label>
          <input
            type="file"
            accept="application/pdf"
            className="cursor-pointer w-full p-2 border rounded bg-white"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <label className="block text-sm font-semibold text-gray-700">Describe Patient Symptoms</label>
          <textarea
            className="w-full p-3 border rounded bg-white resize-none focus:outline-blue-500"
            rows={6}
            placeholder="e.g. 38-year-old male with fever, dry cough, shortness of breath..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            required
          />

          <button
            type="submit"
            className="w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition duration-200"
          >
            {loading ? 'Analyzing...' : 'Submit'}
          </button>
        </form>
      </div>

      {/* Right Panel */}
      <div className="col-span-2">
        {response && (
          <div className="p-6 border rounded-lg bg-gray-50 shadow">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">🔍 Diagnosis Result</h2>
            <div className="space-y-3 text-sm text-gray-700 leading-relaxed">
              <div><strong>Type:</strong> {response.type}</div>
              <div><strong>Matched Prompt:</strong> {response.matched_prompt}</div>
              <div><strong>Diagnosis:</strong> {response.diagnosis}</div>
              <div><strong>Summary:</strong> {response.summary}</div>
              <div>
                <strong>Extracted Symptoms (JSON):</strong>
                <pre className="bg-gray-100 p-2 rounded text-xs text-black overflow-auto">
                  {JSON.stringify(response.structured_symptoms, null, 2)}
                </pre>
              </div>
            </div>

            <button
              onClick={() => setResponse(null)}
              className="mt-6 text-red-600 underline text-sm"
            >
              Clear Result
            </button>
          </div>
        )}
      </div>
    </main>
  )
}
