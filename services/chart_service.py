import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Set headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import settings

logger = logging.getLogger(__name__)

_CHARTS_DIR = Path(settings.TMP_DIR) / "charts"
_TMPL_PATH = Path(__file__).parent.parent / "templates" / "table_snapshot.html"

def _ensure_dir() -> None:
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def render_bar_chart(labels: list[str], values: list[float], title: str, output_path: str | None = None) -> str:
    """
    Renders a premium bar chart using matplotlib and returns the output path.
    Uses the project's Navy/Amber/Ice color palette.
    """
    _ensure_dir()
    if not output_path:
        output_path = str(_CHARTS_DIR / f"chart_{uuid.uuid4().hex[:8]}.png")

    try:
        # Standard size for Telegram viewing
        fig, ax = plt.subplots(figsize=(8, 5))
        
        NAVY = "#0d1b2a"
        AMBER = "#f59e0b"
        LIGHT_BG = "#f8fafc"
        
        fig.patch.set_facecolor(LIGHT_BG)
        ax.set_facecolor("#ffffff")
        
        # Clean labels (wrap or truncate long values)
        clean_labels = []
        for label in labels:
            lbl = str(label)
            if len(lbl) > 20:
                lbl = lbl[:18] + "..."
            clean_labels.append(lbl)
            
        # Draw bars
        bars = ax.bar(clean_labels, values, color=NAVY, edgecolor=AMBER, linewidth=1.2, width=0.6)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.2f}" if isinstance(height, float) and not height.is_integer() else f"{int(height):,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color=NAVY
            )
            
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=NAVY, family="sans-serif")
        
        # Hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        
        ax.tick_params(colors=NAVY, labelsize=9)
        plt.xticks(rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        
        logger.info("Rendered bar chart to %s", output_path)
        return output_path
    except Exception as exc:
        logger.exception("Failed rendering bar chart: %s", exc)
        raise

def render_table_snapshot(headers: list[str], rows: list[list[str]], title: str, output_path: str | None = None) -> str:
    """
    Renders non-numeric tabular data into a PNG snapshot using the existing WeasyPrint
    and PyMuPDF (fitz) stack.
    """
    _ensure_dir()
    if not output_path:
        output_path = str(_CHARTS_DIR / f"table_{uuid.uuid4().hex[:8]}.png")

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from weasyprint import HTML
        import fitz  # PyMuPDF
        
        # 1. Render Jinja2 template
        env = Environment(
            loader=FileSystemLoader(str(_TMPL_PATH.parent)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template(_TMPL_PATH.name)
        generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")
        html_content = template.render(
            title=title,
            headers=headers,
            rows=rows,
            generated_at=generated_at
        )
        
        # 2. Write temp PDF via WeasyPrint
        temp_pdf = _CHARTS_DIR / f"temp_{uuid.uuid4().hex[:8]}.pdf"
        HTML(string=html_content).write_pdf(str(temp_pdf))
        
        # 3. Convert page 0 of PDF to PNG via fitz (PyMuPDF)
        doc = fitz.open(str(temp_pdf))
        page = doc[0]
        # Make the image higher resolution (2.0 scale factor)
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        pix.save(output_path)
        doc.close()
        
        # Clean up temp PDF
        if temp_pdf.exists():
            temp_pdf.unlink()
            
        logger.info("Rendered table snapshot to %s", output_path)
        return output_path
    except Exception as exc:
        logger.exception("Failed rendering table snapshot: %s", exc)
        raise
