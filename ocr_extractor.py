# OCR and text extraction for PDF and image resumes
# Uses PyMuPDF (fitz) as primary extractor for superior text quality,
# with pypdf as fallback. Extracts hyperlinks from PDF annotations.

import io
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# URL regex for extracting links from plain text
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(
    r'(?:https?://|www\.)[^\s<>\"\')]+|'          # full URLs
    r'(?:linkedin\.com/in/[^\s<>\"\')]+)|'         # linkedin shorthand
    r'(?:github\.com/[^\s<>\"\')]+)|'              # github shorthand
    r'(?:twitter\.com/[^\s<>\"\')]+)|'             # twitter shorthand
    r'(?:x\.com/[^\s<>\"\')]+)|'                   # x.com shorthand
    r'(?:mailto:[^\s<>\"\')]+)',                    # mailto links
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Unicode / PDF artefact cleanup
# ---------------------------------------------------------------------------
_PDF_ARTEFACT_REPLACEMENTS = [
    # common icon font / symbol artefacts seen in resumes
    (r'♂\s*phone\s*', ''),
    (r'/envel⌢pe\s*', ''),
    (r'/linkedin\s*', ''),
    (r'/github\s*', ''),
    (r'/gl⌢be\s*', ''),
    (r'⌢', 'o'),
    (r'\uf0b7', '• '),       # bullet
    (r'\uf0a7', '• '),       # bullet variant
    (r'\uf0d8', ''),         # decorative
    (r'\uf095', ''),         # phone icon
    (r'\uf0e0', ''),         # envelope icon
    (r'\uf08c', ''),         # linkedin icon
    (r'\uf09b', ''),         # github icon
    (r'\uf0ac', ''),         # globe icon
    (r'\uf099', ''),         # twitter icon
    (r'\|', ' | '),          # pipe separator spacing
]


def _clean_pdf_text(text: str) -> str:
    """Clean common PDF artefacts and normalize whitespace."""
    if not text:
        return ""
    for pattern, replacement in _PDF_ARTEFACT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    # collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # collapse excessive spaces (but preserve newlines)
    text = re.sub(r'[^\S\n]{3,}', '  ', text)
    return text.strip()


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from plain text using regex."""
    if not text:
        return []
    urls = _URL_PATTERN.findall(text)
    # normalize: add https:// to bare domain patterns
    cleaned = []
    seen = set()
    for url in urls:
        url = url.rstrip('.,;:!?)')
        if not url.startswith(('http://', 'https://', 'mailto:')):
            url = 'https://' + url
        if url.lower() not in seen:
            seen.add(url.lower())
            cleaned.append(url)
    return cleaned


# ---------------------------------------------------------------------------
# PyMuPDF (fitz) — primary PDF extractor
# ---------------------------------------------------------------------------

def _extract_with_pymupdf(pdf_input) -> tuple[str, list[str]]:
    """Extract text and hyperlinks using PyMuPDF. Returns (text, links)."""
    import pymupdf

    if isinstance(pdf_input, bytes):
        doc = pymupdf.open(stream=pdf_input, filetype="pdf")
    elif isinstance(pdf_input, io.BytesIO):
        doc = pymupdf.open(stream=pdf_input.read(), filetype="pdf")
    else:
        doc = pymupdf.open(str(pdf_input))

    text_parts = []
    links = []

    for page in doc:
        # extract text with layout preservation
        page_text = page.get_text("text")
        if page_text:
            text_parts.append(page_text)

        # extract hyperlink annotations (the actual URL targets)
        for link in page.get_links():
            uri = link.get("uri", "")
            if uri and uri.startswith(("http://", "https://", "mailto:")):
                if uri not in links:
                    links.append(uri)

    doc.close()
    full_text = "\n".join(text_parts).strip()
    return full_text, links


# ---------------------------------------------------------------------------
# pypdf — fallback extractor
# ---------------------------------------------------------------------------

def _extract_with_pypdf(pdf_input) -> tuple[str, list[str]]:
    """Extract text and hyperlinks using pypdf. Returns (text, links)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_input)
    text_parts = []
    links = []

    for page in reader.pages:
        # try layout mode first for better formatting
        page_text = None
        try:
            page_text = page.extract_text(extraction_mode="layout")
        except Exception:
            pass
        if not page_text:
            page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

        # extract hyperlink annotations
        if "/Annots" in page:
            try:
                for annot in page["/Annots"]:
                    obj = annot.get_object() if hasattr(annot, "get_object") else annot
                    if obj.get("/Subtype") == "/Link":
                        action = obj.get("/A")
                        if action:
                            a_obj = action.get_object() if hasattr(action, "get_object") else action
                            uri = a_obj.get("/URI", "")
                            if uri and uri not in links:
                                links.append(str(uri))
            except Exception:
                pass

    full_text = "\n".join(text_parts).strip()
    return full_text, links


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_input) -> str:
    """Extract text from PDF file path or file-like object."""
    text, _ = extract_text_and_links_from_pdf(pdf_input)
    return text


def extract_text_and_links_from_pdf(pdf_input) -> tuple[str, list[str]]:
    """Extract cleaned text and deduplicated links from a PDF.
    
    Tries PyMuPDF first (superior quality), falls back to pypdf.
    Returns (cleaned_text, links_list).
    """
    text = ""
    links = []

    # try PyMuPDF first
    try:
        text, links = _extract_with_pymupdf(pdf_input)
    except Exception:
        pass

    # fallback to pypdf if PyMuPDF failed or returned empty
    if not text:
        try:
            # reset stream position if BytesIO
            if isinstance(pdf_input, io.BytesIO):
                pdf_input.seek(0)
            text, links_fallback = _extract_with_pypdf(pdf_input)
            if not links:
                links = links_fallback
        except Exception as e:
            return f"[ERROR: Failed to extract text: {str(e)}]", []

    if not text:
        return "", links

    # clean artefacts
    text = _clean_pdf_text(text)

    # also extract URLs from the text body itself
    text_urls = extract_urls_from_text(text)
    for url in text_urls:
        if url not in links:
            links.append(url)

    return text, links


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes (backward compatible)."""
    text, _ = extract_text_and_links_from_bytes(pdf_bytes)
    return text


def extract_text_and_links_from_bytes(pdf_bytes: bytes) -> tuple[str, list[str]]:
    """Extract text and links from PDF bytes."""
    return extract_text_and_links_from_pdf(io.BytesIO(pdf_bytes))


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Extract text from image using EasyOCR."""
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
    """Extract text from any supported file by path."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        with open(file_path, "rb") as f:
            return extract_text_from_image_bytes(f.read())
    return f"[ERROR: Unsupported file type: {suffix}]"
