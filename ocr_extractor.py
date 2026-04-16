"""OCR and text extraction helpers for PDF and image resumes."""

import io
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


def extract_text_from_pdf(pdf_input) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_input: File path (str) or file-like object (BytesIO).

    Returns:
        Extracted text as a single string. Returns an error
        message string if extraction fails.
    """
    try:
        reader = PdfReader(pdf_input)

        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            return ""

        return full_text

    except Exception as e:
        return f"[ERROR: Failed to extract text: {str(e)}]"


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from raw PDF bytes (for Streamlit file uploads)."""
    return extract_text_from_pdf(io.BytesIO(pdf_bytes))


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Extract text from image bytes using optional EasyOCR.

    Returns an error string when OCR backend is unavailable or extraction fails.
    """
    try:
        import easyocr
        import numpy as np
        from PIL import Image
    except Exception:
        return "[ERROR: easyocr/Pillow/numpy not installed for image OCR]"

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        reader = easyocr.Reader(["en"], gpu=False)
        chunks = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(chunks).strip()
        if not text:
            return ""
        return text
    except Exception as e:
        return f"[ERROR: Failed image OCR: {str(e)}]"


def extract_text_from_file_path(file_path: str) -> str:
    """Detect file type from extension and run matching extraction path."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        with open(file_path, "rb") as f:
            return extract_text_from_image_bytes(f.read())
    return f"[ERROR: Unsupported file type: {suffix}]"
