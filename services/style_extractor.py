"""
services/style_extractor.py — Extract a visual style profile from reference files.

Scope (explicitly stated — not a bug to "fix" without a real layout engine):
  This extracts look-and-feel: fonts, sizes, colors, approximate column count.
  It does NOT clone exact pixel-level layout, margin positions, or graphic placement.

For .docx: validates the reference opens and has styles; returns path unchanged
           for use as a base template in generators/docx_gen.py.
For .pdf:  uses fitz (PyMuPDF) span inspection to infer heading/body styles
           and a heuristic column-count estimate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Counter

from services.exceptions import StyleExtractionError

logger = logging.getLogger(__name__)


@dataclass
class StyleProfile:
    heading_font: str        = "Helvetica"
    heading_size_pt: float   = 16.0
    heading_color_hex: str   = "#000000"
    body_font: str           = "Helvetica"
    body_size_pt: float      = 10.0
    accent_color_hex: str    = "#1F4E79"
    columns: int             = 1        # 1 or 2, inferred from text block x-positions
    has_shaded_table_headers: bool = False


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _fitz_color_to_hex(color: int | tuple | None) -> str:
    """Convert fitz color (packed int or RGB tuple 0-1 floats) to #RRGGBB."""
    if color is None:
        return "#000000"
    if isinstance(color, (int, float)):
        # Packed integer: 0xRRGGBB
        c = int(color)
        r, g, b = (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF
        return _rgb_to_hex(r, g, b)
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        r, g, b = (int(v * 255) for v in color[:3])
        return _rgb_to_hex(r, g, b)
    return "#000000"


def extract_pdf_style(path: str | Path) -> StyleProfile:
    """
    Inspect the PDF at *path* using fitz and return a StyleProfile.

    Heuristics used:
      heading: largest font size on page 1 (capped at 40pt to skip logos/outliers)
      body:    most frequently occurring font+size pair across the document
      columns: cluster text-block x-start positions; two clear clusters → 2 columns
      accent:  first non-black/non-white colored text span found, else default blue

    Raises:
        StyleExtractionError: if the file cannot be opened or contains no text spans.
    """
    try:
        import fitz  # noqa: PLC0415
    except ImportError as exc:
        raise StyleExtractionError("PyMuPDF not installed — run: pip install pymupdf") from exc

    p = Path(path)
    if not p.exists():
        raise StyleExtractionError(f"Reference file not found: {path}")

    try:
        doc = fitz.open(str(p))
    except Exception as exc:
        raise StyleExtractionError(f"Could not open PDF for style extraction: {exc}") from exc

    profile = StyleProfile()
    all_spans: list[dict] = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span["_page"] = page_num
                    span["_x0"]   = block["bbox"][0]
                    all_spans.append(span)

    if not all_spans:
        raise StyleExtractionError(
            "No text spans found in PDF — it may contain only images/scans."
        )

    # ── Heading: largest font on page 0, capped at 40pt ──────────────────────
    page0_spans = [s for s in all_spans if s["_page"] == 0]
    heading_candidates = sorted(
        page0_spans, key=lambda s: s.get("size", 0), reverse=True
    )
    for cand in heading_candidates:
        size = cand.get("size", 0)
        if size <= 40:
            profile.heading_font     = cand.get("font", "Helvetica")
            profile.heading_size_pt  = round(size, 1)
            profile.heading_color_hex = _fitz_color_to_hex(cand.get("color"))
            break

    # ── Body: most frequent font+size pair ────────────────────────────────────
    freq: Counter[tuple[str, float]] = Counter()
    for span in all_spans:
        key = (span.get("font", ""), round(span.get("size", 10), 1))
        freq[key] += 1
    if freq:
        body_font, body_size = freq.most_common(1)[0][0]
        profile.body_font    = body_font
        profile.body_size_pt = body_size

    # ── Accent: first non-black, non-white colored span ───────────────────────
    for span in all_spans:
        color_hex = _fitz_color_to_hex(span.get("color"))
        if color_hex not in ("#000000", "#FFFFFF", "#000", "#FFF"):
            profile.accent_color_hex = color_hex
            break

    # ── Column count: cluster x-start positions ───────────────────────────────
    x_starts = sorted({round(s["_x0"]) for s in all_spans if s["_page"] == 0})
    profile.columns = _estimate_columns(x_starts)

    doc.close()
    logger.info("Style extracted from %s: %s", p.name, profile)
    return profile


def _estimate_columns(x_starts: list[int]) -> int:
    """
    Heuristic: two distinct x-position clusters with a consistent gap → 2 columns.
    Default to 1 if detection is ambiguous to avoid wrong layout.
    """
    if len(x_starts) < 4:
        return 1

    # Find the largest gap between consecutive sorted x positions
    gaps = [(x_starts[i+1] - x_starts[i], i) for i in range(len(x_starts) - 1)]
    max_gap, max_idx = max(gaps, key=lambda t: t[0])
    page_width_estimate = x_starts[-1] - x_starts[0]

    # Gap must be >25% of the text-spread width to be a real column separator
    if page_width_estimate > 0 and max_gap / page_width_estimate > 0.25:
        logger.info("Column heuristic: 2 columns detected (gap=%d, spread=%d)", max_gap, page_width_estimate)
        return 2

    logger.info("Column heuristic: defaulting to 1 column (gap=%d, spread=%d)", max_gap, page_width_estimate)
    return 1


def validate_docx_reference(path: str | Path) -> str:
    """
    Validate that *path* is a readable .docx with at least one style defined.
    Returns the path string unchanged (for passing to docx_gen as template base).

    Raises:
        StyleExtractionError: if file is unreadable or has no styles.
    """
    p = Path(path)
    if not p.exists():
        raise StyleExtractionError(f"Reference DOCX not found: {path}")

    try:
        from docx import Document  # noqa: PLC0415
        doc = Document(str(p))
        style_names = {s.name for s in doc.styles}
        if not style_names:
            raise StyleExtractionError("Reference DOCX has no styles defined.")
        logger.info("DOCX reference valid — %d styles found", len(style_names))
        return str(p)
    except StyleExtractionError:
        raise
    except Exception as exc:
        raise StyleExtractionError(f"Could not open DOCX reference: {exc}") from exc
