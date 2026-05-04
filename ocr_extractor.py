"""OCR and text extraction helpers for PDF and image resumes."""

import io
import re
from pathlib import Path


_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"')]+|"
    r"(?:linkedin\.com/in/[^\s<>\"')]+)|"
    r"(?:github\.com/[^\s<>\"')]+)|"
    r"(?:twitter\.com/[^\s<>\"')]+)|"
    r"(?:x\.com/[^\s<>\"')]+)|"
    r"(?:mailto:[^\s<>\"')]+)",
    re.IGNORECASE,
)

_PDF_ARTEFACT_REPLACEMENTS = [
    (r"\u2642\s*phone\s*", ""),
    (r"/envel\u2322pe\s*", ""),
    (r"/linkedin\s*", ""),
    (r"/github\s*", ""),
    (r"/gl\u2322be\s*", ""),
    (r"\u2322", "o"),
    (r"\uf0b7", "- "),
    (r"\uf0a7", "- "),
    (r"\uf0d8", ""),
    (r"\uf095", ""),
    (r"\uf0e0", ""),
    (r"\uf08c", ""),
    (r"\uf09b", ""),
    (r"\uf0ac", ""),
    (r"\uf099", ""),
    (r"\|", " | "),
]


def _clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    for pattern, replacement in _PDF_ARTEFACT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]{3,}", "  ", text)
    return text.strip()


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = _URL_PATTERN.findall(text)
    cleaned = []
    seen = set()
    for url in urls:
        url = url.rstrip(".,;:!?)")
        if not url.startswith(("http://", "https://", "mailto:")):
            url = "https://" + url
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(url)
    return cleaned


def _extract_with_pymupdf(pdf_input) -> tuple[str, list[str]]:
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
        page_text = page.get_text("text")
        if page_text:
            text_parts.append(page_text)

        for link in page.get_links():
            uri = link.get("uri", "")
            if uri and uri.startswith(("http://", "https://", "mailto:")):
                if uri not in links:
                    links.append(uri)

    doc.close()
    full_text = "\n".join(text_parts).strip()
    return full_text, links


def _extract_with_pypdf(pdf_input) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    if isinstance(pdf_input, bytes):
        reader = PdfReader(io.BytesIO(pdf_input))
    else:
        reader = PdfReader(pdf_input)

    text_parts = []
    links = []

    for page in reader.pages:
        page_text = None
        try:
            page_text = page.extract_text(extraction_mode="layout")
        except Exception:
            pass
        if not page_text:
            page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

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


def extract_text_and_links_from_pdf(pdf_input) -> tuple[str, list[str]]:
    text = ""
    links: list[str] = []

    try:
        text, links = _extract_with_pymupdf(pdf_input)
    except Exception:
        pass

    if not text:
        try:
            if isinstance(pdf_input, io.BytesIO):
                pdf_input.seek(0)
            text, links_fallback = _extract_with_pypdf(pdf_input)
            if not links:
                links = links_fallback
        except Exception as e:
            return f"[ERROR: Failed to extract text: {str(e)}]", []

    if not text:
        return "", links

    text = _clean_pdf_text(text)

    text_urls = extract_urls_from_text(text)
    for url in text_urls:
        if url not in links:
            links.append(url)

    return text, links


def extract_text_from_pdf(pdf_input) -> str:
    text, _ = extract_text_and_links_from_pdf(pdf_input)
    return text


def extract_text_and_links_from_bytes(pdf_bytes: bytes) -> tuple[str, list[str]]:
    return extract_text_and_links_from_pdf(io.BytesIO(pdf_bytes))


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    text, _ = extract_text_and_links_from_bytes(pdf_bytes)
    return text


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
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
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        with open(file_path, "rb") as f:
            return extract_text_from_image_bytes(f.read())
    return f"[ERROR: Unsupported file type: {suffix}]"
