import React, { useCallback, useState } from 'react'

/**
 * UploadZone — drag-and-drop image upload with preview
 * Props:
 *   onImageSelected(file, previewUrl): void
 *   isLoading: bool
 */
export default function UploadZone({ onImageSelected, isLoading = false }) {
  const [isDragging, setIsDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    onImageSelected?.(file, url)
  }, [onImageSelected])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }, [handleFile])

  const onFileChange = (e) => handleFile(e.target.files?.[0])

  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = () => setIsDragging(false)

  return (
    <div style={{ width: '100%' }}>
      <label
        className={`upload-zone${isDragging ? ' upload-zone--active' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        htmlFor="image-upload"
        aria-label="Upload an image for face detection"
        style={{ cursor: isLoading ? 'not-allowed' : 'pointer', display: 'block' }}
      >
        {previewUrl ? (
          <div style={{ position: 'relative' }}>
            <img
              src={previewUrl}
              alt="Selected for analysis"
              style={{
                maxHeight: '280px', maxWidth: '100%',
                borderRadius: '8px', objectFit: 'contain',
                margin: '0 auto', display: 'block',
              }}
            />
            <div style={{
              marginTop: '12px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem', color: 'var(--text-muted)',
            }}>
              Click or drop to change image
            </div>
          </div>
        ) : (
          <>
            <span className="upload-zone__icon" aria-hidden="true">
              {isLoading ? (
                <span className="animate-spin" style={{ display: 'inline-block' }}>⟳</span>
              ) : '⬆'}
            </span>
            <div className="upload-zone__title">
              {isLoading ? 'Analyzing…' : 'Drop an image here'}
            </div>
            <div className="upload-zone__sub">
              or click to browse — JPG, PNG, BMP, WebP
            </div>
          </>
        )}
      </label>

      <input
        id="image-upload"
        type="file"
        accept="image/*"
        onChange={onFileChange}
        disabled={isLoading}
        style={{ display: 'none' }}
        aria-label="Image file input"
      />
    </div>
  )
}
