"""
generators/xlsx_gen.py — AI-driven Excel spreadsheet generation.

The JSON-robustness problem is the highest-risk part of this module.
Full hardening approach:
  1. Strict system prompt: JSON-only, no fences, no commentary.
  2. Defensive strip of common wrapper artifacts before json.loads().
  3. json.loads() in try/except — on failure, retry ONCE with an explicit
     "your previous response was invalid JSON" message.
  4. Schema validation after successful parse (non-empty headers, consistent
     row lengths). Treat validation failure same as parse failure.
  5. If both attempts fail: raise TableGenerationError. The handler must
     catch this and tell the user rather than emitting a broken spreadsheet.

openpyxl formatting:
  - Header row: bold white text on deep-blue fill, frozen at A2
  - Auto-filter on header row
  - Alternating light-grey data rows
  - Auto column widths (capped at 50)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from config import settings
from services.exceptions import TableGenerationError

logger = logging.getLogger(__name__)

_XLSX_DIR = Path(settings.TMP_DIR) / "xlsx"

_SYSTEM_PROMPT = (
    "Respond with ONLY a valid JSON object. "
    "No markdown code fences, no explanation, no leading or trailing text. "
    "Your entire response must be parseable by json.loads().\n"
    "Required schema:\n"
    '{"title": "string", "headers": ["string", ...], "rows": [["value", ...], ...]}\n'
    "Every row array MUST have exactly the same length as the headers array."
)

_RETRY_ADDON = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY the JSON object, nothing else — no markdown, no explanation."
)


def _strip_artifacts(raw: str) -> str:
    """
    Defensively strip common LLM output artifacts before json.loads():
      - ```json ... ``` or ``` ... ``` fences
      - Leading/trailing whitespace and stray backticks
      - Occasional 'json' prefix without fences
    """
    raw = raw.strip()
    # Remove triple-backtick fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Remove stray backtick remnants
    raw = raw.strip("`").strip()
    # Remove leading 'json' word if the model still prefixed it
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return raw


def _validate_schema(data: object) -> tuple[list[str], list[list[str]]]:
    """
    Validate the parsed JSON object matches the required schema.

    Returns:
        (headers, rows) — both as plain string lists.

    Raises:
        ValueError: descriptive message for the retry/fail path.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (dict), got {type(data).__name__}")

    headers = data.get("headers")
    rows = data.get("rows")

    if not headers or not isinstance(headers, list):
        raise ValueError("'headers' key is missing or empty")
    if rows is None or not isinstance(rows, list):
        raise ValueError("'rows' key is missing")

    # Coerce headers to non-empty strings
    headers = [str(h).strip() for h in headers]
    if not all(headers):
        raise ValueError("One or more header values are empty strings")

    n_cols = len(headers)
    validated_rows: list[list[str]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"Row {i} is not a list (got {type(row).__name__})")
        if len(row) != n_cols:
            raise ValueError(
                f"Row {i} has {len(row)} cells but headers define {n_cols} columns"
            )
        validated_rows.append([str(c) for c in row])

    return headers, validated_rows


async def _ask_ollama(prompt: str, model: str | None = None) -> str:
    """Call ollama_service.structured_chat and return raw response string."""
    from services.ollama_service import structured_chat  # noqa: PLC0415
    # structured_chat returns a dict — convert back to string for our retry logic
    # Actually: structured_chat already handles one retry internally.
    # We call it directly and let it raise ValueError on total failure.
    result = await structured_chat(prompt, model=model)
    return json.dumps(result)   # re-serialize so _strip_artifacts + json.loads flow works


async def generate_excel(user_request: str, model: str | None = None) -> tuple[str, str, list[str], list[list[str]]]:
    """
    Generate a formatted .xlsx file based on a natural-language request.

    Args:
        user_request: The user's original text (e.g. "top 10 programming languages").
        model: The target model name to use.

    Returns:
        A tuple of (file_path, title, headers, rows).

    Raises:
        TableGenerationError: if LLM cannot produce valid schema-conformant JSON
                              after all retries — handler must surface this to user.
    """
    _XLSX_DIR.mkdir(parents=True, exist_ok=True)

    base_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Generate a data table for: {user_request}"
    )

    last_error: Exception | None = None

    for attempt in range(2):
        prompt = base_prompt if attempt == 0 else base_prompt + _RETRY_ADDON
        try:
            raw = await _ask_ollama(prompt, model=model)
            cleaned = _strip_artifacts(raw)
            parsed = json.loads(cleaned)
            headers, rows = _validate_schema(parsed)
            title = str(parsed.get("title", user_request[:50]))

            # Both parse and validate succeeded
            path = _build_xlsx(title, headers, rows)
            logger.info(
                "Excel generated: %s  (%d rows × %d cols) on attempt %d",
                path, len(rows), len(headers), attempt + 1,
            )
            return path, title, headers, rows

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Excel attempt %d failed validation: %s", attempt + 1, exc
            )

    raise TableGenerationError(
        f"Could not generate valid table data after 2 attempts. "
        f"Last error: {last_error}"
    )


def _build_xlsx(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Build and save the formatted xlsx. Returns the file path."""
    _XLSX_DIR.mkdir(parents=True, exist_ok=True)   # ensure dir exists for direct calls too
    try:
        import openpyxl  # noqa: PLC0415
        from openpyxl.styles import (  # noqa: PLC0415
            Alignment, Border, Font, PatternFill, Side,
        )
        from openpyxl.utils import get_column_letter  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("openpyxl not installed") from exc

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"

    # ── Styles ────────────────────────────────────────────────────────────────
    HEADER_BLUE = "1F4E79"
    ALT_GREY    = "F2F2F2"
    BORDER_GREY = "CCCCCC"

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill("solid", fgColor=HEADER_BLUE)
    alt_fill    = PatternFill("solid", fgColor=ALT_GREY)
    data_font   = Font(size=10)
    thin        = Side(style="thin", color=BORDER_GREY)
    border      = Border(bottom=thin, right=thin)
    center      = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left        = Alignment(horizontal="left",   vertical="center", wrap_text=True)

    # ── Title row ─────────────────────────────────────────────────────────────
    ws.merge_cells(
        start_row=1, start_column=1,
        end_row=1,   end_column=len(headers),
    )
    title_cell = ws.cell(row=1, column=1, value=title)
    title_cell.font      = Font(bold=True, size=13, color=HEADER_BLUE)
    title_cell.alignment = center
    ws.row_dimensions[1].height = 28

    # ── Header row (row 2) ────────────────────────────────────────────────────
    for col_i, hdr in enumerate(headers, start=1):
        c = ws.cell(row=2, column=col_i, value=hdr)
        c.font      = header_font
        c.fill      = header_fill
        c.alignment = center
        c.border    = border
    ws.row_dimensions[2].height = 22

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_i, row in enumerate(rows, start=3):
        use_fill = (row_i % 2 == 1)   # alternate starting on first data row
        for col_i, val in enumerate(row, start=1):
            c = ws.cell(row=row_i, column=col_i, value=val)
            c.font      = data_font
            c.fill      = alt_fill if use_fill else PatternFill()
            c.alignment = left
            c.border    = border
        ws.row_dimensions[row_i].height = 18

    # ── Auto-filter (covers header + data, row 2 onward) ─────────────────────
    ws.auto_filter.ref = (
        f"A2:{get_column_letter(len(headers))}{max(2, 2 + len(rows))}"
    )

    # ── Freeze panes below header row ─────────────────────────────────────────
    ws.freeze_panes = "A3"

    # ── Auto column widths (capped at 50) ─────────────────────────────────────
    for col_i in range(1, len(headers) + 1):
        col_letter = get_column_letter(col_i)
        max_len = 0
        for row_cells in ws.iter_rows(min_col=col_i, max_col=col_i):
            for cell in row_cells:
                try:
                    max_len = max(max_len, len(str(cell.value or "")))
                except Exception:
                    pass
        ws.column_dimensions[col_letter].width = max(10, min(max_len + 4, 50))

    out_path = _XLSX_DIR / f"table_{uuid.uuid4().hex[:8]}.xlsx"
    wb.save(str(out_path))
    return str(out_path)
