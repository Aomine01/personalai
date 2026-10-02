"""
services/file_reader.py — Extract readable text/tables from uploaded files.

Primary PDF backend:  PyMuPDF (fitz) — faster, better text ordering.
Fallback PDF backend: pdfplumber — better table detection on complex layouts.
Returned via ExtractedDocument dataclass so callers always know whether
content was truncated and what page counts / warnings are present.

Never let raw library exceptions propagate — wrap in UnreadableFileError.
"""
from __future__ import annotations

import csv
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from services.exceptions import UnreadableFileError
from utils.file_limits import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_CHARS_FOR_LLM,
    MAX_PDF_PAGES_TO_READ,
)

# Maximum pages to OCR (keeps Gemini API cost/time reasonable)
_MAX_OCR_PAGES = 10

logger = logging.getLogger(__name__)

_TRUNCATION_NOTICE = "\n\n[Content truncated to fit model context window — document may contain more information]"
_PAGE_CAP_NOTICE   = "[Note: document has {total} pages; only the first {cap} were read]\n\n"


@dataclass
class ExtractedDocument:
    text: str
    truncated: bool
    page_or_sheet_count: int
    warnings: list[str] = field(default_factory=list)


# ── PDF ───────────────────────────────────────────────────────────────────────

def _read_pdf_fitz(path: Path) -> ExtractedDocument:
    """Primary PDF reader using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError as exc:
        raise UnreadableFileError("PyMuPDF not installed — run: pip install pymupdf") from exc

    warnings: list[str] = []
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise UnreadableFileError(
            f"Could not open PDF — it may be corrupt or password-protected: {exc}"
        ) from exc

    if doc.needs_pass:
        raise UnreadableFileError(
            "This PDF is password-protected and cannot be read without the password."
        )

    total_pages = doc.page_count
    pages_to_read = min(total_pages, MAX_PDF_PAGES_TO_READ)
    chunks: list[str] = []
    image_page_count = 0

    if total_pages > MAX_PDF_PAGES_TO_READ:
        warnings.append(
            f"Document has {total_pages} pages; only the first {MAX_PDF_PAGES_TO_READ} were read."
        )

    for page_num in range(pages_to_read):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            chunks.append(f"[Page {page_num + 1}]\n{text}")
        else:
            image_page_count += 1

    doc.close()
    raw = "\n\n".join(chunks)

    # Return partial result with metadata so _read_pdf can decide on OCR
    doc_result = _finalize(raw, pages_to_read, warnings)
    # Tag as image-only if no text was found at all
    if not raw.strip():
        doc_result.warnings.append("__IMAGE_ONLY_PDF__")
    elif image_page_count > 0:
        doc_result.warnings.append(f"__PARTIAL_IMAGE_PDF__:{image_page_count}")

    return doc_result


def _read_pdf_pdfplumber(path: Path) -> ExtractedDocument:
    """Fallback PDF reader using pdfplumber (better table detection)."""
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError as exc:
        raise UnreadableFileError("pdfplumber not installed") from exc

    warnings: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(total_pages, MAX_PDF_PAGES_TO_READ)

            if total_pages > MAX_PDF_PAGES_TO_READ:
                warnings.append(
                    f"Document has {total_pages} pages; only the first {MAX_PDF_PAGES_TO_READ} were read."
                )

            chunks: list[str] = []
            for i, page in enumerate(pdf.pages[:pages_to_read], start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                table_text = ""
                for table in tables:
                    rows = [
                        "\t".join(str(c) if c else "" for c in row)
                        for row in table if row
                    ]
                    table_text += "\n".join(rows) + "\n"
                combined = (text + "\n" + table_text).strip()
                if combined:
                    chunks.append(f"[Page {i}]\n{combined}")

    except Exception as exc:
        raise UnreadableFileError(f"Could not read PDF: {exc}") from exc

    raw = "\n\n".join(chunks)
    if not raw.strip():
        raise UnreadableFileError("No readable text found in PDF.")

    return _finalize(raw, pages_to_read, warnings)


def _preprocess_for_ocr(img: "PIL.Image.Image") -> "PIL.Image.Image":  # noqa: F821
    """
    Enhance image quality for better OCR accuracy:
    - Convert to greyscale (reduces noise Gemini has to filter)
    - Sharpen edges (helps with blurry scans)
    - Boost contrast (makes faint text stand out)
    """
    try:
        from PIL import ImageEnhance, ImageFilter  # noqa: PLC0415
        # Convert RGBA/palette to RGB first, then greyscale
        img = img.convert("L")
        # Unsharp mask — sharpens without introducing artefacts
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        # Boost contrast — makes dark ink pop on white background
        img = ImageEnhance.Contrast(img).enhance(1.8)
        return img
    except Exception:
        return img  # return original if preprocessing fails


def _ocr_pdf_with_gemini(path: Path, max_pages: int = _MAX_OCR_PAGES) -> ExtractedDocument:
    """
    Render each page to a high-res PNG, preprocess it, then send to Gemini for OCR.
    Used as a fallback when the PDF contains only scanned images without a text layer.

    Accuracy techniques applied:
    - 300 DPI rendering (vs 150 — 4× more pixels, much sharper text)
    - Greyscale + contrast boost + unsharp-mask sharpening
    - Full gemini-3.5-flash model (vs lite — better at dense/handwritten text)
    - Rich OCR prompt asking Gemini to preserve document structure
    """
    try:
        import fitz               # noqa: PLC0415
        from google import genai   # noqa: PLC0415
        import PIL.Image           # noqa: PLC0415
        from config import settings  # noqa: PLC0415
    except ImportError as exc:
        raise UnreadableFileError(f"OCR dependencies not installed: {exc}") from exc

    if not settings.GEMINI_API_KEY:
        raise UnreadableFileError(
            "No readable text found — the PDF appears to contain only scanned images.\n"
            "To enable OCR for scanned PDFs, set GEMINI_API_KEY in your .env file."
        )

    logger.info("Falling back to Gemini OCR for image-based PDF: %s", path.name)
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise UnreadableFileError(f"Could not open PDF for OCR: {exc}") from exc

    total_pages = doc.page_count
    pages_to_ocr = min(total_pages, max_pages)
    warnings: list[str] = []

    if total_pages > max_pages:
        warnings.append(
            f"Document has {total_pages} pages; OCR applied to the first {max_pages} only."
        )

    _OCR_PROMPT = (
        "You are a precise OCR engine. Extract ALL text from this scanned document page "
        "EXACTLY as it appears — do not summarise, paraphrase, or omit anything.\n\n"
        "Rules:\n"
        "- Preserve the original language (do NOT translate)\n"
        "- Keep headings, subheadings, numbered lists, bullet points, and paragraph breaks\n"
        "- Reproduce tables in plain text (use | separators for columns)\n"
        "- Keep numbers, dates, codes, and signatures exactly as written\n"
        "- If a word is partially illegible, output your best guess followed by [?]\n"
        "- Output ONLY the extracted text — no explanations or commentary"
    )

    chunks: list[str] = []
    for page_num in range(pages_to_ocr):
        page = doc[page_num]
        # 300 DPI — 4× more pixels than 150 DPI, dramatically sharper text
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_img_path = tmp.name
            pix.save(tmp_img_path)

        try:
            img = PIL.Image.open(tmp_img_path)
            img.load()  # force full read before file is deleted
            img = _preprocess_for_ocr(img)

            # Direct synchronous call — this function runs inside a thread executor
            response = client.models.generate_content(
                model="models/gemini-3.5-flash",   # full model — better at dense/handwritten text
                contents=[_OCR_PROMPT, img],
            )
            page_text = response.text.strip()
            if page_text:
                chunks.append(f"[Page {page_num + 1} — OCR]\n{page_text}")
                logger.info("OCR page %d: %d chars extracted", page_num + 1, len(page_text))
        except Exception as exc:
            logger.warning("Gemini OCR failed for page %d: %s", page_num + 1, exc)
            warnings.append(f"OCR failed for page {page_num + 1}: {exc}")
        finally:
            Path(tmp_img_path).unlink(missing_ok=True)

    doc.close()
    raw = "\n\n".join(chunks)

    if not raw.strip():
        raise UnreadableFileError(
            "Could not extract any text from this PDF even with OCR. "
            "The document may be blank or the images are too low quality to read."
        )

    warnings.insert(0, "📷 This PDF contains scanned images — OCR was applied (300 DPI, high-accuracy mode).")
    return _finalize(raw, pages_to_ocr, warnings)


def _read_pdf(path: Path) -> ExtractedDocument:
    """Try fitz first; fall back to pdfplumber; then Gemini OCR for image-only PDFs."""
    try:
        result = _read_pdf_fitz(path)
    except UnreadableFileError:
        raise   # password / corrupt — nothing will help
    except Exception as exc:
        logger.warning("fitz failed on %s (%s), trying pdfplumber", path.name, exc)
        try:
            result = _read_pdf_pdfplumber(path)
        except UnreadableFileError:
            raise
        except Exception:
            raise

    # Check if fitz detected an image-only or partial-image PDF
    is_image_only = any("__IMAGE_ONLY_PDF__" in w for w in result.warnings)
    if is_image_only:
        # Strip the internal sentinel tag before passing to OCR
        result.warnings = [w for w in result.warnings if "__IMAGE_ONLY_PDF__" not in w and "__PARTIAL_IMAGE_PDF__" not in w]
        return _ocr_pdf_with_gemini(path)

    # Partial image PDF — clean up sentinels and return what we have
    result.warnings = [w for w in result.warnings if "__PARTIAL_IMAGE_PDF__" not in w]
    return result


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _read_docx(path: Path) -> ExtractedDocument:
    try:
        from docx import Document  # noqa: PLC0415
    except ImportError as exc:
        raise UnreadableFileError("python-docx not installed") from exc

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise UnreadableFileError(f"Could not open DOCX — may be corrupt: {exc}") from exc

    chunks: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            chunks.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                chunks.append("\t".join(cells))

    raw = "\n".join(chunks)
    if not raw.strip():
        raise UnreadableFileError("DOCX appears to contain no readable text.")

    section_count = len(doc.sections)
    return _finalize(raw, section_count, [])


# ── XLSX ──────────────────────────────────────────────────────────────────────

def _read_xlsx(path: Path) -> ExtractedDocument:
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError as exc:
        raise UnreadableFileError("openpyxl not installed") from exc

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise UnreadableFileError(f"Could not open XLSX — may be corrupt or encrypted: {exc}") from exc

    chunks: list[str] = []
    sheet_count = len(wb.sheetnames)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        sheet_rows: list[str] = [f"[Sheet: {sheet_name}]"]
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(c.strip() for c in cells):
                sheet_rows.append("\t".join(cells))
        if len(sheet_rows) > 1:
            chunks.append("\n".join(sheet_rows))
    wb.close()

    raw = "\n\n".join(chunks)
    if not raw.strip():
        raise UnreadableFileError("XLSX appears to contain no data.")

    return _finalize(raw, sheet_count, [])


# ── CSV ───────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> ExtractedDocument:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = ["\t".join(row) for row in reader]
        raw = "\n".join(rows)
    except Exception as exc:
        raise UnreadableFileError(f"Could not read CSV: {exc}") from exc

    if not raw.strip():
        raise UnreadableFileError("CSV appears to be empty.")

    return _finalize(raw, len(rows), [])


# ── Shared post-processing ────────────────────────────────────────────────────

def _finalize(raw: str, count: int, warnings: list[str]) -> ExtractedDocument:
    """Apply truncation if needed and return an ExtractedDocument."""
    truncated = False
    if len(raw) > MAX_CHARS_FOR_LLM:
        logger.warning("Extracted text (%d chars) truncated to %d", len(raw), MAX_CHARS_FOR_LLM)
        raw = raw[:MAX_CHARS_FOR_LLM] + _TRUNCATION_NOTICE
        truncated = True
        warnings.append(
            f"Extracted text exceeded {MAX_CHARS_FOR_LLM} characters and was truncated. "
            "The model's answer is based only on the portion that fits the context window."
        )
    return ExtractedDocument(
        text=raw,
        truncated=truncated,
        page_or_sheet_count=count,
        warnings=warnings,
    )


# ── Public API ────────────────────────────────────────────────────────────────

_READERS = {
    ".pdf":  _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".xls":  _read_xlsx,
    ".csv":  _read_csv,
}


def read_file(path: str | Path) -> ExtractedDocument:
    """
    Extract text content from *path*.

    Returns:
        ExtractedDocument with text, truncation flag, count, and warnings.

    Raises:
        UnreadableFileError: file is unsupported, corrupt, password-protected, or empty.
        FileNotFoundError:   path does not exist.
    """
    p = Path(path)
    ext = p.suffix.lower()

    # Extension check before existence check — gives clearer error for unsupported types
    reader = _READERS.get(ext)
    if reader is None:
        raise UnreadableFileError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Reading %s (%s)", p.name, ext)
    return reader(p)


def build_llm_prompt(doc: ExtractedDocument, filename: str, user_question: str) -> str:
    """
    Build the full prompt string to send to Ollama for file analysis.
    Kept here so the construction logic is co-located with extraction logic.
    """
    warnings_str = "; ".join(doc.warnings) if doc.warnings else "none"
    return (
        f'The user uploaded a file named "{filename}". '
        f"Below is its extracted content (may be partial — see warnings).\n\n"
        f"Warnings: {warnings_str}\n\n"
        f"Content:\n{doc.text}\n\n"
        f"Answer the user's question based only on this content. "
        f"If the answer is not present in the content, say so clearly instead of guessing.\n\n"
        f"User's question: {user_question}"
    )
