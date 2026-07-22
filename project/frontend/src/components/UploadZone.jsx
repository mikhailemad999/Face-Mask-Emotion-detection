import React, { useCallback, useState } from 'react'

/**
 * UploadZone — Drag-and-drop image upload component with live image preview.
 *
 * @param {Object} props - Component properties.
 * @param {Function} props.onImageSelected - Callback invoked when an image file is selected `(file: File, previewUrl: string) => void`.
 * @param {boolean} [props.isLoading=false] - Whether detection inference is active.
 * @returns {JSX.Element} Interactive file drag-and-drop upload zone component.
 */
export default function UploadZone({ onImageSelected, isLoading = false }) {
  const [isDragging, setIsDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)

  /**
   * Validate selected image file and create object URL preview.
   * @param {File} file - Selected browser File instance.
   */
  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    onImageSelected?.(file, url)
  }, [onImageSelected])

  /**
   * Handle drag drop event.
   * @param {DragEvent} e - HTML DragEvent.
   */
  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }, [handleFile])

  /**
   * Handle input file change event.
   * @param {ChangeEvent<HTMLInputElement>} e - React input change event.
   */
  const onFileChange = (e) => handleFile(e.target.files?.[0])

  /**
   * Handle dragover event to highlight upload zone.
   * @param {DragEvent} e - HTML DragEvent.
   */
  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true) }

  /** Handle dragleave event to clear active state. */
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
