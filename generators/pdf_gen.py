"""
generators/pdf_gen.py — PDF generation via HTML/Jinja2.

Primary renderer: WeasyPrint (HTML+CSS → PDF via Cairo/Pango).
Fallback renderer: fpdf2 (already installed, no system deps required).

WeasyPrint requires system libraries (pango, cairo, gdk-pixbuf, gobject):
  brew install pango cairo gdk-pixbuf libffi gobject-introspection

If WeasyPrint is unavailable, the fpdf2 fallback is used and the user is
informed that the output won't reflect the reference style profile exactly.
This is explicit degraded-mode behaviour, not a silent fallback.

Style injection: CSS custom properties (--heading-font, --heading-color, etc.)
are set from the StyleProfile dataclass so generated PDFs visually match
the reference file's look-and-feel.

Scope limit (stated, not a bug):
  Style matching covers fonts, colors, column count, and table header shading.
  Exact pixel-level layout reproduction requires a dedicated layout engine
  and is out of scope for this phase.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from services.style_extractor import StyleProfile

logger = logging.getLogger(__name__)

_PDF_DIR    = Path(settings.TMP_DIR) / "pdf"
_TMPL_PATH  = Path(__file__).parent.parent / "templates" / "pdf_base.html"


def _ensure_dir() -> None:
    _PDF_DIR.mkdir(parents=True, exist_ok=True)


def _out_path() -> Path:
    return _PDF_DIR / f"report_{uuid.uuid4().hex[:8]}.pdf"


def _check_weasyprint() -> bool:
    """Return True if WeasyPrint and its system libs are importable."""
    try:
        from weasyprint import HTML  # noqa: PLC0415, F401
        return True
    except (ImportError, OSError):
        return False


def _render_html(
    content_sections: list[dict],  # type: ignore[type-arg]
    style: "StyleProfile | None",
    title: str,
    image_path: str | None = None,
) -> str:
    """Render the Jinja2 HTML template to a string."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("jinja2 not installed — run: pip install jinja2") from exc

    import markdown  # noqa: PLC0415

    # Process section bodies from markdown to clean HTML
    processed_sections: list[dict] = []
    for sec in content_sections:
        sec_copy = dict(sec)
        body = sec.get("body", "")
        if body:
            sec_copy["body_html"] = markdown.markdown(body, extensions=["extra", "nl2br"])
        processed_sections.append(sec_copy)

    # Default StyleProfile-like object for use when style=None
    class _DefaultStyle:
        heading_font = "'Playfair Display', Georgia, serif"
        heading_color_hex = "#0d1b2a"
        body_font = "'Inter', Helvetica, Arial, sans-serif"
        body_size_pt = 10
        accent_color_hex = "#1F4E79"
        columns = 1
        has_shaded_table_headers = True

    s = style if style is not None else _DefaultStyle()

    # Calculate natural aspect ratio and optimal display sizing for image
    image_meta: dict | None = None
    if image_path and Path(image_path).exists():
        try:
            from PIL import Image as PILImage  # noqa: PLC0415
            with PILImage.open(image_path) as img:
                w, h = img.size
                aspect = (w / h) if h > 0 else 1.0
                is_portrait = aspect < 0.95
                if is_portrait:
                    disp_h = 260
                    disp_w = int(260 * aspect)
                else:
                    disp_h = min(220, int(600 / aspect))
                    disp_w = int(disp_h * aspect)

                image_meta = {
                    "path": image_path,
                    "width": w,
                    "height": h,
                    "aspect": round(aspect, 2),
                    "is_portrait": is_portrait,
                    "disp_w": disp_w,
                    "disp_h": disp_h,
                }
        except Exception as e:
            logger.warning("Could not calculate image dimensions: %s", e)
            image_meta = {"path": image_path, "is_portrait": False, "disp_h": 220}

    env = Environment(
        loader=FileSystemLoader(str(_TMPL_PATH.parent)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(_TMPL_PATH.name)
    return template.render(
        title=title,
        sections=processed_sections,
        style=s,
        image_path=image_path,
        image_meta=image_meta,
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
    )


def _generate_weasyprint(html: str, out: Path) -> None:
    from weasyprint import HTML  # noqa: PLC0415
    HTML(string=html, base_url=str(Path(__file__).parent.parent)).write_pdf(str(out))


def _to_latin1(text: str) -> str:
    """
    Replace common Unicode characters with Latin-1-safe ASCII equivalents,
    then strip anything still outside Latin-1.  This prevents fpdf2 from
    raising 'Character X is outside the range supported by font Helvetica'.
    """
    _REPLACEMENTS: list[tuple[str, str]] = [
        # Dashes
        ("\u2013", "-"),   # en-dash –
        ("\u2014", "--"),  # em-dash —
        ("\u2015", "--"),  # horizontal bar ―
        # Quotation marks
        ("\u2018", "'"),   # left single quote '
        ("\u2019", "'"),   # right single quote / apostrophe '
        ("\u201a", ","),   # single low-9 quote ‚
        ("\u201b", "'"),   # single high-reversed-9 quote ‛
        ("\u201c", '"'),   # left double quote "
        ("\u201d", '"'),   # right double quote "
        ("\u201e", '"'),   # double low-9 quote „
        ("\u201f", '"'),   # double high-reversed-9 quote ‟
        ("\u2039", "<"),   # single left-pointing angle quote ‹
        ("\u203a", ">"),   # single right-pointing angle quote ›
        ("\u00ab", "<<"),  # left-pointing double angle «
        ("\u00bb", ">>"),  # right-pointing double angle »
        # Ellipsis & misc punctuation
        ("\u2026", "..."), # ellipsis …
        ("\u2022", "-"),   # bullet •
        ("\u2023", "-"),   # triangular bullet ‣
        ("\u25cf", "-"),   # black circle ●
        ("\u25e6", "-"),   # white bullet ◦
        ("\u2043", "-"),   # hyphen bullet ⁃
        ("\u2219", "-"),   # bullet operator ∙
        # Spaces
        ("\u00a0", " "),   # non-breaking space
        ("\u2009", " "),   # thin space
        ("\u200b", ""),    # zero-width space
        ("\u202f", " "),   # narrow no-break space
        # Mathematical / other common chars
        ("\u00d7", "x"),   # multiplication sign ×
        ("\u00f7", "/"),   # division sign ÷
        ("\u2260", "!="),  # not equal ≠
        ("\u2264", "<="),  # less-than or equal ≤
        ("\u2265", ">="),  # greater-than or equal ≥
        ("\u00b0", " deg"),# degree °
        ("\u00b2", "2"),   # superscript 2 ²
        ("\u00b3", "3"),   # superscript 3 ³
        ("\u2192", "->"),  # rightwards arrow →
        ("\u2190", "<-"),  # leftwards arrow ←
        ("\u2764", "<3"),  # heart ❤
    ]
    for src, dst in _REPLACEMENTS:
        text = text.replace(src, dst)
    # Final pass: encode to latin-1, replacing anything still unsupported
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _generate_fpdf2(
    content_sections: list[dict],  # type: ignore[type-arg]
    style: "StyleProfile | None",
    title: str,
    out: Path,
    image_path: str | None = None,
) -> None:
    """
    fpdf2 fallback — basic formatting only.
    Respects heading color and font name from the style profile where possible,
    but fpdf2 cannot load arbitrary system fonts without explicit registration.
    Supports an optional *image_path* to embed a full-width image at the top.
    """
    from fpdf import FPDF, XPos, YPos  # noqa: PLC0415

    BLUE = (31, 78, 121)
    GREY = (245, 245, 245)
    DARK = (50, 50, 50)
    WHITE = (255, 255, 255)

    class StyledPDF(FPDF):
        def header(self) -> None:  # type: ignore[override]
            self.set_fill_color(*BLUE)
            self.rect(0, 0, self.w, 20, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*WHITE)
            self.set_y(4)
            self.cell(0, 12, _to_latin1(title), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*DARK)
            self.ln(5)

        def footer(self) -> None:  # type: ignore[override]
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.cell(0, 10, _to_latin1(f"Generated by AI Agent - {ts} - Page {self.page_no()}"), align="C")

    pdf = StyledPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 26, 18)
    pdf.add_page()

    # ── Embed image at the top if provided ───────────────────────────────────
    if image_path:
        try:
            from PIL import Image as PILImage  # noqa: PLC0415
            with PILImage.open(image_path) as img:
                img_w, img_h = img.size
            # Scale image to fit within usable page width (max 160mm), keep aspect ratio
            max_w_mm = pdf.epw  # usable width in mm
            aspect = img_h / img_w if img_w else 1.0
            disp_w = min(max_w_mm, 160.0)
            disp_h = disp_w * aspect
            # Center horizontally
            x_pos = (pdf.w - disp_w) / 2
            pdf.image(image_path, x=x_pos, y=pdf.get_y(), w=disp_w, h=disp_h)
            pdf.ln(disp_h + 8)  # move cursor below image + small gap
        except Exception as img_exc:
            logger.warning("Could not embed image in PDF: %s", img_exc)

    # ── Sections (text content) ───────────────────────────────────────────────
    for sec in content_sections:
        heading = _to_latin1(sec.get("heading", ""))
        body    = _to_latin1(sec.get("body", ""))
        rows    = sec.get("table_rows")

        if heading:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*BLUE)
            pdf.cell(0, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*DARK)

        if body:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 6, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        if rows and isinstance(rows, list):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(*BLUE)
            pdf.set_text_color(*WHITE)
            col_w = pdf.epw / max(len(rows[0]), 1)
            for i, cell in enumerate(rows[0]):
                pdf.cell(col_w, 7, _to_latin1(str(cell)), fill=True, border=1)
            pdf.ln()
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(*DARK)
            for row in rows[1:]:
                for cell in row:
                    pdf.cell(col_w, 6, str(cell), border=1)
                pdf.ln()

        pdf.ln(3)

    pdf.output(str(out))


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    content_sections: list[dict],  # type: ignore[type-arg]
    style: "StyleProfile | None" = None,
    title: str = "AI Generated Report",
    image_path: str | None = None,
) -> tuple[str, bool]:
    """
    Generate a PDF document.

    Args:
        content_sections: List of {\"heading\": str, \"body\": str, \"table_rows\": list, \"level\": int}
        style:            StyleProfile from style_extractor, or None for defaults.
        title:            Document title shown in header.
        image_path:       Optional path to an image file to embed at the top of the PDF.

    Returns:
        (file_path, weasyprint_used)
        weasyprint_used=False means fpdf2 fallback was used (style matching limited).

    Raises:
        RuntimeError: if neither renderer is available.
    """
    _ensure_dir()
    out = _out_path()

    if _check_weasyprint():
        logger.info("PDF renderer: WeasyPrint")
        html = _render_html(content_sections, style, title, image_path=image_path)
        _generate_weasyprint(html, out)
        return str(out), True
    else:
        logger.warning(
            "WeasyPrint unavailable (system libs missing) — using fpdf2 fallback. "
            "Install: brew install pango cairo gdk-pixbuf gobject-introspection && pip install weasyprint"
        )
        _generate_fpdf2(content_sections, style, title, out, image_path=image_path)
        return str(out), False

