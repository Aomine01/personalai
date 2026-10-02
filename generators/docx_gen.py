"""
generators/docx_gen.py — Word document generation (plain + style-cloned).

Two entry points:
  generate_docx()                  — plain generation using built-in styles
  generate_docx_from_reference()   — style-cloned from a reference .docx

TOC LIMITATION (stated in code and surfaced to user):
  python-docx can insert a TOC field code but cannot populate it.
  The generated document will show an empty TOC field until the user
  opens it in Microsoft Word and uses: References → Update Table.
  This is a documented python-docx limitation, not a bug in this code.

Style-cloning approach:
  We do NOT recreate styles manually.
  We open the reference document, capture its existing style names,
  clear the body content, then rebuild using those style names.
  If the reference contains Jinja2 placeholder tags ({{ ... }}),
  docxtpl is used instead for template-driven rendering.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from config import settings
from services.exceptions import StyleExtractionError

logger = logging.getLogger(__name__)

_DOCX_DIR = Path(settings.TMP_DIR) / "docx"

# Content section type expected by both generators:
# {"heading": str, "body": str, "table_rows": list[list[str]] (optional)}
ContentSection = dict  # type: ignore[type-arg]


def _ensure_dir() -> None:
    _DOCX_DIR.mkdir(parents=True, exist_ok=True)


def _out_path() -> Path:
    return _DOCX_DIR / f"doc_{uuid.uuid4().hex[:8]}.docx"


# ── Plain generation ──────────────────────────────────────────────────────────

def generate_docx(
    content_sections: list[ContentSection],
    title: str = "AI Generated Report",
    include_toc: bool = True,
) -> str:
    """
    Generate a properly structured .docx using python-docx built-in styles.

    TOC note: Word must open the file and "Update Field" (right-click the TOC)
    to populate the table of contents — python-docx cannot render a live TOC.
    This is communicated to the user by the calling handler.

    Returns:
        Absolute path to the generated .docx file.
    """
    _ensure_dir()
    try:
        from docx import Document  # noqa: PLC0415
        from docx.oxml import OxmlElement  # noqa: PLC0415
        from docx.oxml.ns import qn  # noqa: PLC0415
        from docx.shared import Inches, Pt, RGBColor  # noqa: PLC0415
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("python-docx not installed") from exc

    doc = Document()

    # Margins
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    # Title
    title_para = doc.add_heading(title, level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # TOC field (Word must update on open)
    if include_toc and content_sections:
        doc.add_paragraph("Table of Contents")
        toc_para = doc.add_paragraph()
        run = toc_para.add_run()
        fld_char_begin = OxmlElement("w:fldChar")
        fld_char_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = 'TOC \\o "1-3" \\h \\z \\u'
        fld_char_end = OxmlElement("w:fldChar")
        fld_char_end.set(qn("w:fldCharType"), "end")
        run._r.append(fld_char_begin)
        run._r.append(instr)
        run._r.append(fld_char_end)
        doc.add_paragraph(
            "⚠ Open this document in Microsoft Word and press Ctrl+A → F9 "
            "(or right-click the TOC area → 'Update Field') to populate the table of contents.",
        ).italic = True

    doc.add_page_break()

    for section_data in content_sections:
        heading = section_data.get("heading", "")
        body    = section_data.get("body", "")
        rows    = section_data.get("table_rows")

        if heading:
            doc.add_heading(heading, level=1)
        if body:
            doc.add_paragraph(body)
        if rows and isinstance(rows, list) and rows:
            _add_table(doc, rows)
        doc.add_paragraph()  # spacer

    out = _out_path()
    doc.save(str(out))
    logger.info("DOCX generated: %s", out.name)
    return str(out)


def _add_table(doc, rows: list[list[str]]) -> None:  # type: ignore[type-arg]
    """Add a table to doc with the first row as a styled header."""
    from docx.shared import RGBColor, Pt  # noqa: PLC0415
    from docx.oxml import OxmlElement  # noqa: PLC0415
    from docx.oxml.ns import qn  # noqa: PLC0415

    def shade_cell(cell, hex_color: str) -> None:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"

    for row_i, row_data in enumerate(rows):
        for col_i, cell_val in enumerate(row_data):
            cell = table.cell(row_i, col_i)
            cell.text = str(cell_val)
            if row_i == 0:
                shade_cell(cell, "1F4E79")
                for run in cell.paragraphs[0].runs:
                    run.font.bold  = True
                    run.font.color.rgb = RGBColor(255, 255, 255)


# ── Style-cloned generation ───────────────────────────────────────────────────

def generate_docx_from_reference(
    reference_path: str,
    content_sections: list[ContentSection],
    title: str = "AI Generated Report",
) -> str:
    """
    Generate a .docx that uses the reference file's existing styles.

    Flow:
      A. If the reference contains Jinja2 tags ({{ ... }}), use docxtpl (template mode).
      B. Otherwise: open reference, capture style names, strip body, rebuild content
         applying the captured styles (style-clone mode).

    Args:
        reference_path:   Path to the reference .docx file.
        content_sections: List of {"heading": str, "body": str, "table_rows": list}

    Returns:
        Absolute path to the generated .docx.

    Raises:
        StyleExtractionError: if the reference file cannot be read.
        RuntimeError:          if required libraries are missing.
    """
    _ensure_dir()
    ref = Path(reference_path)
    if not ref.exists():
        raise StyleExtractionError(f"Reference file not found: {reference_path}")

    # Detect template mode: scan for {{ or {%
    try:
        raw_xml = ref.read_bytes().decode("utf-8", errors="ignore")
        is_template = bool(re.search(r"\{\{|\{%", raw_xml))
    except Exception:
        is_template = False

    if is_template:
        logger.info("Reference DOCX contains Jinja2 tags — using docxtpl template mode")
        return _generate_from_template(ref, content_sections, title)
    else:
        logger.info("Reference DOCX is a finished document — using style-clone mode")
        return _generate_style_clone(ref, content_sections, title)


def _generate_from_template(
    ref: Path, sections: list[ContentSection], title: str
) -> str:
    """docxtpl path: reference must contain {{ title }}, {{ sections }}, etc."""
    try:
        from docxtpl import DocxTemplate  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("docxtpl not installed — run: pip install docxtpl") from exc

    tpl = DocxTemplate(str(ref))
    context = {
        "title":    title,
        "sections": sections,
    }
    tpl.render(context)
    out = _out_path()
    tpl.save(str(out))
    logger.info("DOCX (template mode) saved: %s", out.name)
    return str(out)


def _generate_style_clone(
    ref: Path, sections: list[ContentSection], title: str
) -> str:
    """
    Style-clone mode:
      1. Open reference, record which style names exist.
      2. Copy the document object (preserving styles.xml).
      3. Delete body paragraphs/tables.
      4. Rebuild body using reference style names where they exist,
         falling back to 'Heading 1'/'Normal' if a particular custom name
         isn't present.
    """
    try:
        from docx import Document  # noqa: PLC0415
        from docx.shared import Inches  # noqa: PLC0415
        import copy  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("python-docx not installed") from exc

    try:
        ref_doc = Document(str(ref))
    except Exception as exc:
        raise StyleExtractionError(f"Cannot open reference DOCX: {exc}") from exc

    # Catalogue available style names (case-insensitive lookup)
    available_styles = {s.name.lower(): s.name for s in ref_doc.styles}
    logger.info(
        "Reference styles available: %s",
        sorted(available_styles.values()),
    )

    def resolve_style(candidates: list[str]) -> str:
        """Return the first candidate style name that exists in the reference."""
        for c in candidates:
            if c.lower() in available_styles:
                return available_styles[c.lower()]
        return candidates[-1]   # absolute fallback

    heading_style = resolve_style(["Heading 1", "heading 1", "h1", "Header"])
    body_style    = resolve_style(["Normal", "Body Text", "Default Paragraph Font"])

    # Clear body content (paragraphs and tables in the body XML element)
    from docx.oxml.ns import qn  # noqa: PLC0415
    body_elem = ref_doc.element.body
    for child in list(body_elem):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag in ("p", "tbl", "sdt"):
            body_elem.remove(child)

    # Rebuild body with new content, applying reference styles
    ref_doc.add_heading(title, level=0)

    for sec in sections:
        heading = sec.get("heading", "")
        body    = sec.get("body", "")
        rows    = sec.get("table_rows")

        if heading:
            p = ref_doc.add_paragraph(heading, style=heading_style)
        if body:
            ref_doc.add_paragraph(body, style=body_style)
        if rows:
            _add_table(ref_doc, rows)
        ref_doc.add_paragraph("", style=body_style)

    out = _out_path()
    ref_doc.save(str(out))
    logger.info(
        "DOCX (style-clone mode) saved: %s  styles used: heading=%r body=%r",
        out.name, heading_style, body_style,
    )
    return str(out)
