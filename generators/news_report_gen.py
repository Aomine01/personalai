"""
generators/news_report_gen.py — Premium news/search summary PDF generator.

Framework-agnostic: no Telegram imports.  Can be imported and called directly
by the Telegram bot (main.py → _handle_web_search) and by the openclaw CLI
wrapper (openclaw_skills/aiagent/cli_news_report.py) without duplication.

Renderer strategy (same guard pattern as generators/pdf_gen.py):
  - WeasyPrint (HTML+CSS → PDF) when available — best visual fidelity.
  - fpdf2 fallback when WeasyPrint is unavailable (system libs missing).

AI-generated image caption rule (enforce here, not just in the template):
  Any ArticleCard whose image.is_ai_generated is True MUST result in a visible
  "AI-generated illustration" caption.  The Jinja2 template handles this for
  the WeasyPrint path; the fpdf2 fallback path handles it explicitly below.
  Do not remove this logic — see services/news_image_fetcher.py for context.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import settings
from services.news_image_fetcher import FetchedImage

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(settings.TMP_DIR) / "reports"
_TMPL_PATH = Path(__file__).parent.parent / "templates" / "news_report.html"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ArticleCard:
    """
    Input data for a single news article card in the report.

    Attributes:
        title:          Article headline.
        url:            Canonical article URL.
        snippet:        Short excerpt (a sentence or two) — never full-article text.
        image:          FetchedImage from news_image_fetcher.  Use image.is_ai_generated
                        to apply mandatory "AI-generated illustration" caption.
        published_date: Optional ISO date string from Tavily, may be None.
    """
    title: str
    url: str
    snippet: str
    image: FetchedImage
    published_date: str | None = None


# ── Headline generator ──────────────────────────────────────────────────────────

def _generate_headline(query: str, answer: str, articles: list[ArticleCard]) -> str:
    """
    Derive an editorial headline from the search results rather than
    echoing the raw user query verbatim.

    Strategy:
      1. Use the first real article title if it reads like a proper headline
         (>= 5 words, no URL-like content).
      2. Otherwise, derive a title from the AI answer (first sentence).
      3. Fall back to a cleaned-up version of the query.
    """
    # 1. First article title heuristic
    for article in articles[:3]:
        title = (article.title or "").strip()
        words = title.split()
        if len(words) >= 5 and "://" not in title and title[0].isupper():
            # Looks like a real headline — use it
            return title[:120]

    # 2. First sentence of the AI answer
    if answer:
        first_sentence = answer.split(".")[0].strip()
        if len(first_sentence.split()) >= 5:
            return first_sentence[:120]

    # 3. Capitalise the query words as a last resort (better than raw lowercase)
    return " ".join(w.capitalize() for w in query.split())[:120]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _out_path() -> Path:
    return _REPORTS_DIR / f"news_report_{uuid.uuid4().hex[:8]}.pdf"


def _check_weasyprint() -> bool:
    """Return True if WeasyPrint and its system libs are importable."""
    try:
        from weasyprint import HTML  # noqa: PLC0415, F401
        return True
    except (ImportError, OSError):
        return False


# ── WeasyPrint renderer ───────────────────────────────────────────────────────

def _render_html(
    query: str,
    headline: str,
    answer: str,
    articles: list[ArticleCard],
    generated_at: str,
) -> str:
    """Render templates/news_report.html to an HTML string via Jinja2."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("jinja2 not installed — run: pip install jinja2") from exc

    env = Environment(
        loader=FileSystemLoader(str(_TMPL_PATH.parent)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(_TMPL_PATH.name)
    return template.render(
        query=query,
        headline=headline,
        answer=answer,
        articles=articles,
        generated_at=generated_at,
    )


def _generate_weasyprint(html: str, out: Path) -> None:
    from weasyprint import HTML  # noqa: PLC0415
    HTML(string=html, base_url=str(Path(__file__).parent.parent)).write_pdf(str(out))


# ── fpdf2 fallback renderer ───────────────────────────────────────────────────

def _to_safe(text: str) -> str:
    """Replace common Unicode chars with Latin-1 safe equivalents for fpdf2."""
    replacements = [
        ("\u2013", "-"), ("\u2014", "--"), ("\u2018", "'"), ("\u2019", "'"),
        ("\u201c", '"'), ("\u201d", '"'), ("\u2026", "..."), ("\u2022", "-"),
        ("\u00a0", " "), ("\u2009", " "), ("\u200b", ""), ("\u00d7", "x"),
        ("\u2192", "->"), ("\u2190", "<-"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _generate_fpdf2(
    query: str,
    answer: str,
    articles: list[ArticleCard],
    generated_at: str,
    out: Path,
) -> None:
    """
    fpdf2 fallback renderer — basic but clean formatting.
    Embeds images when available.  Shows mandatory "AI-generated illustration"
    caption for any card where image.is_ai_generated is True.
    """
    from fpdf import FPDF, XPos, YPos  # noqa: PLC0415

    NAVY   = (13, 27, 42)
    AMBER  = (245, 158, 11)
    DARK   = (26, 32, 44)
    MID    = (74, 85, 104)
    LIGHT  = (240, 244, 248)
    WHITE  = (255, 255, 255)

    class NewsPDF(FPDF):
        def header(self) -> None:  # type: ignore[override]
            self.set_fill_color(*NAVY)
            self.rect(0, 0, self.w, 18, "F")
            self.set_fill_color(*AMBER)
            self.rect(0, 0, 4, 18, "F")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*WHITE)
            self.set_y(3)
            self.set_x(10)
            self.cell(0, 12, _to_safe(f"AI Agent News Report — {query}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*DARK)
            self.ln(4)

        def footer(self) -> None:  # type: ignore[override]
            self.set_y(-14)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(140, 140, 140)
            self.cell(0, 10, _to_safe(f"Generated by AI Agent · {generated_at} · Page {self.page_no()}"), align="C")

    pdf = NewsPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 24, 15)
    pdf.add_page()

    # ── AI Summary block ──────────────────────────────────────────────────────
    if answer:
        pdf.set_fill_color(*LIGHT)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*AMBER[0:2] + (0,))  # dark amber
        pdf.set_text_color(180, 100, 0)
        pdf.cell(0, 6, "AI SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(*DARK)
        pdf.set_fill_color(*LIGHT)
        pdf.multi_cell(0, 6, _to_safe(answer), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

    # ── Article cards ─────────────────────────────────────────────────────────
    for i, article in enumerate(articles, 1):
        # Card header bar
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 9)
        label = _to_safe(f"{i}. {article.title[:90]}")
        pdf.cell(0, 8, label, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Embed article image if available
        if article.image and article.image.local_path:
            try:
                from PIL import Image as PILImage  # noqa: PLC0415
                with PILImage.open(article.image.local_path) as img:
                    img_w, img_h = img.size
                max_w = pdf.epw
                aspect = img_h / img_w if img_w else 1.0
                disp_w = min(max_w, 140.0)
                disp_h = min(disp_w * aspect, 60.0)  # cap height at 60mm
                pdf.image(article.image.local_path, x=pdf.get_x(), y=pdf.get_y(), w=disp_w, h=disp_h)
                pdf.ln(disp_h + 2)

                # MANDATORY AI caption — do not remove
                if article.image.is_ai_generated:
                    pdf.set_font("Helvetica", "I", 7)
                    pdf.set_text_color(*MID)
                    pdf.cell(0, 5, "AI-generated illustration", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            except Exception as img_exc:
                logger.warning("fpdf2: could not embed card image: %s", img_exc)

        # Snippet
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(*MID)
        pdf.multi_cell(0, 5, _to_safe(article.snippet or ""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # URL footer
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(180, 100, 0)
        pdf.cell(0, 5, _to_safe(article.url[:90]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_draw_color(220, 220, 220)
        pdf.ln(2)
        pdf.cell(0, 0, "", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    pdf.output(str(out))


# ── Public API ────────────────────────────────────────────────────────────────

def generate_news_report(
    query: str,
    answer: str,
    articles: list[ArticleCard],
    output_dir: str | None = None,
) -> tuple[str, str]:
    """
    Generate a styled news/search summary PDF.

    Framework-agnostic — no Telegram imports.  Safe to call from both the
    Telegram bot and the openclaw CLI wrapper.

    Args:
        query:       The original search query (used as the report title).
        answer:      AI-synthesized summary from Tavily.
        articles:    List of ArticleCard with titles, snippets, URLs, and images.
        output_dir:  Override output directory (default: /tmp/aiagent/reports).

    Returns:
        (file_path, renderer_used) — renderer_used is "weasyprint" or "fpdf2".

    Raises:
        ReportGenerationError: if both renderers fail.
    """
    from services.exceptions import ReportGenerationError  # noqa: PLC0415

    _ensure_dir()
    out = Path(output_dir) / f"news_report_{uuid.uuid4().hex[:8]}.pdf" if output_dir else _out_path()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")
    headline = _generate_headline(query, answer, articles)

    try:
        if _check_weasyprint():
            logger.info("News report renderer: WeasyPrint")
            html = _render_html(query, headline, answer, articles, generated_at)
            _generate_weasyprint(html, out)
            return str(out), "weasyprint"
        else:
            logger.info(
                "News report renderer: fpdf2 (WeasyPrint unavailable — "
                "install: brew install pango cairo gdk-pixbuf gobject-introspection && pip install weasyprint)"
            )
            _generate_fpdf2(query, answer, articles, generated_at, out)
            return str(out), "fpdf2"

    except Exception as exc:
        logger.exception("News report generation failed: %s", exc)
        raise ReportGenerationError(f"Could not generate news report: {exc}") from exc
