---
name: aiagent
description: Run tasks inside the Telegram AI Agent codebase, such as generating Excel spreadsheets and sweeping temporary files, or generating styled news-search PDF reports.
---

# AI Agent Utilities Skill

Use this skill when the user wants to generate structured Excel files, data tables, perform maintenance on the local `aiagent` project files, or generate a styled news/search summary PDF report.

The project root directory is: `/Users/sharofiddin/Desktop/aiagent`
The Python interpreter is: `/Users/sharofiddin/Desktop/aiagent/.venv/bin/python`

### 1. Generating an Excel Spreadsheet
When the user asks to generate a spreadsheet, table, or `.xlsx` file from natural language description (e.g., "create a table of top 10 movies" or "Excel sheet for top tech companies"), run:

```bash
/Users/sharofiddin/Desktop/aiagent/.venv/bin/python /Users/sharofiddin/Desktop/aiagent/cli.py excel "<Prompt describing the data>"
```

**Instructions:**
- Pass the user's description in quotes inside the prompt argument.
- The command outputs `SUCCESS: <file_path>` upon completion.
- Retrieve the file at `<file_path>` and attach/send it back to the user in the messaging channel.

### 2. Cleaning Up Temp Files
When the user asks to clean up temporary directories, run:

```bash
/Users/sharofiddin/Desktop/aiagent/.venv/bin/python /Users/sharofiddin/Desktop/aiagent/cli.py cleanup
```

### 3. Generating a Styled News/Search PDF Report
When the user asks for a premium news summary document (e.g., "search AI news and make a report", "give me a PDF of today's top stories"), use the CLI wrapper:

```bash
/Users/sharofiddin/Desktop/aiagent/.venv/bin/python \
  /Users/sharofiddin/Desktop/aiagent/openclaw_skills/aiagent/cli_news_report.py \
  "<search query>"
```

**Example:**
```bash
/Users/sharofiddin/Desktop/aiagent/.venv/bin/python \
  /Users/sharofiddin/Desktop/aiagent/openclaw_skills/aiagent/cli_news_report.py \
  "latest breakthroughs in artificial intelligence"
```

The command outputs:
```
✅ Report generated: /tmp/aiagent/reports/news_report_<hash>.pdf (renderer: weasyprint)
```

Retrieve the PDF at the reported path and send it to the user.

#### How image enrichment works
- Tavily search results include `image_url` fields per article.
- The fetcher downloads each image with a **5MB cap**, **8s timeout**, and **Content-Type validation** (only `image/*` accepted).
- Images are fetched **concurrently** — a single broken URL never stalls the batch.
- Articles with no available image receive a styled gradient placeholder card in the PDF.

#### AI-generated image fallback — OPT-IN ONLY
- **Default behavior**: placeholder gradient cards. No AI images are generated silently.
- AI-generated images (`allow_ai_fallback=True`) are disabled by default and must be explicitly enabled. If enabled:
  - The image prompt includes `"editorial illustration, not photorealistic"` — never a realistic photo style.
  - A mandatory `"AI-generated illustration"` caption appears on that card in both Telegram and the PDF.
  - **Do not enable this by default.** It exists for cases where the user explicitly requests illustrated content alongside news.

#### When to recommend a report vs. plain search
- **Plain search**: user wants quick facts, links, or a brief answer → use Telegram text reply.
- **News report**: user wants to save/share/read a nicely formatted summary with article cards and images → run `cli_news_report.py` and send the PDF.

#### Shared code — no duplication
The CLI wrapper (`cli_news_report.py`) imports from:
- `services/search_service.py` — structured Tavily search
- `services/news_image_fetcher.py` — safe image download with all guardrails
- `generators/news_report_gen.py` — WeasyPrint/fpdf2 PDF renderer
- `templates/news_report.html` — Jinja2 visual template

Any visual changes must go into those shared files, not into the CLI wrapper itself.
