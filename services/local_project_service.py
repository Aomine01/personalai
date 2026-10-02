"""
services/local_project_service.py — Local project scanner & codebase analyzer.

Allows the bot to explore and inspect local folders/projects on the user's Mac
(e.g., ~/Desktop, ~/Projects, ~/Documents), read dependencies, inspect structure,
and produce detailed architecture reports via Ollama.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from config import settings
from services import ollama_service

logger = logging.getLogger(__name__)

# Common ignored directories and files
IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", ".next", ".nuxt", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "coverage",
    ".turbo", ".cache", "target", "vendor", "bin", "obj"
}

IGNORED_EXTENSIONS = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".woff",
    ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".pdf", ".zip", ".tar", ".gz",
    ".lock", "-lock.json", ".db", ".sqlite", ".sqlite3"
}

CONFIG_FILES = [
    "package.json", "tsconfig.json", "requirements.txt", "pyproject.toml",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "README.md", "readme.md"
]


def extract_project_path_candidates(query: str) -> list[str]:
    """Extract candidate directory paths or project names from natural language."""
    candidates: list[str] = []
    
    # 1. Look for explicit multi-segment paths (e.g. jamgarma/yangiavlod, ~/Desktop/...)
    path_matches = re.findall(r"(?:~|/Users/[\w\-]+|[\w\-]+(?:/[\w\-]+)+)", query)
    for p in path_matches:
        if len(p) > 2:
            candidates.append(p.strip())

    # 2. Look for project-specific and folder phrases
    proj_matches = re.findall(
        r"(?:project|codebase|repo|app)\s+(?:called\s+|named\s+|in\s+)?([a-zA-Z0-9_\-\.\/]+)",
        query,
        re.IGNORECASE
    )
    folder_matches = re.findall(
        r"(?:folder|directory)\s+(?:called\s+|named\s+|in\s+)?([a-zA-Z0-9_\-\.\/]+)",
        query,
        re.IGNORECASE
    )

    # If both a parent folder and a child project are mentioned (e.g. folder jamgarma and project yangiavlod),
    # combine them as "jamgarma/yangiavlod"
    if folder_matches and proj_matches:
        f = folder_matches[0].strip().strip("'\"`")
        p = proj_matches[0].strip().strip("'\"`")
        if f.lower() not in {"the", "my", "a", "this", "and"} and p.lower() not in {"the", "my", "a", "this", "and"}:
            candidates.insert(0, f"{f}/{p}")

    for kw in proj_matches:
        clean_kw = kw.strip().strip("'\"`")
        if clean_kw and clean_kw.lower() not in {"called", "named", "and", "the", "it", "my", "this"}:
            candidates.append(clean_kw)

    for kw in folder_matches:
        clean_kw = kw.strip().strip("'\"`")
        if clean_kw and clean_kw.lower() not in {"called", "named", "and", "the", "it", "my", "this"}:
            candidates.append(clean_kw)

    return candidates


def resolve_local_project_path(query_or_name: str) -> Path | None:
    """
    Locates an existing project directory on the Mac based on a path or name.
    Searches exact path, Desktop, Documents, Home, and search subdirectories.
    """
    home = Path.home()
    search_roots = [
        home / "Desktop",
        home / "Documents",
        home / "Projects",
        home / "Developer",
        home,
    ]

    candidates = extract_project_path_candidates(query_or_name)
    if not candidates:
        # Fallback: try raw query as single word candidate
        single_words = [w.strip(".,'\"`") for w in query_or_name.split() if len(w) > 2]
        candidates = single_words

    for cand in candidates:
        cand_str = cand.replace("~", str(home))
        
        # 1. Direct path check
        p = Path(cand_str).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

        # 2. Check directly under search roots
        for root in search_roots:
            direct_child = root / cand
            if direct_child.exists() and direct_child.is_dir():
                return direct_child

        # 3. Check 2-level deep under Desktop / search roots (e.g. Desktop/jamgarma/yangiavlod)
        clean_name = Path(cand).name.lower()
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for entry in root.iterdir():
                    if entry.is_dir() and entry.name not in IGNORED_DIRS:
                        if entry.name.lower() == clean_name:
                            return entry
                        # Search 1 level deeper (e.g. ~/Desktop/jamgarma/*)
                        try:
                            for sub in entry.iterdir():
                                if sub.is_dir() and sub.name not in IGNORED_DIRS and sub.name.lower() == clean_name:
                                    return sub
                        except (PermissionError, OSError):
                            pass
            except (PermissionError, OSError):
                pass

    return None


def scan_project(dir_path: Path) -> dict[str, Any]:
    """
    Scans a local project directory:
    - Generates a directory tree
    - Extracts configuration files (package.json, README, etc.)
    - Samples key source code files
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files_list: list[str] = []
    tree_lines: list[str] = [f"📂 {dir_path.name}/"]
    config_data: dict[str, str] = {}
    source_samples: dict[str, str] = {}
    total_files = 0

    # Build Tree & collect file list (max 3 levels)
    def _walk(current: Path, prefix: str = "", depth: int = 0):
        nonlocal total_files
        if depth > 3:
            return
        try:
            entries = sorted(list(current.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (PermissionError, OSError):
            return

        for idx, entry in enumerate(entries):
            if entry.name in IGNORED_DIRS or entry.name.startswith("."):
                continue
            if any(entry.name.endswith(ext) for ext in IGNORED_EXTENSIONS):
                continue

            is_last = (idx == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

            rel_path = entry.relative_to(dir_path)
            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}📁 {entry.name}/")
                _walk(entry, new_prefix, depth + 1)
            else:
                total_files += 1
                files_list.append(str(rel_path))
                if len(tree_lines) < 60:
                    tree_lines.append(f"{prefix}{connector}📄 {entry.name}")

    _walk(dir_path)

    # Read important configuration and documentation files
    for conf_name in CONFIG_FILES:
        target = dir_path / conf_name
        if target.exists() and target.is_file():
            try:
                config_data[conf_name] = target.read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception as e:
                logger.warning("Could not read config file %s: %s", target, e)

    # Read key source files (up to 6 files, 150 lines each)
    source_candidates = [
        f for f in files_list
        if any(f.endswith(ext) for ext in [".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go", ".vue", ".svelte", ".html"])
        and not f.endswith(".d.ts")
        and not f.endswith(".test.ts")
        and not f.endswith(".spec.ts")
    ]

    # Prioritize entrypoints & main files
    priority_keywords = ["main", "index", "app", "server", "page", "route", "landing", "config", "core", "api"]
    sorted_sources = sorted(
        source_candidates,
        key=lambda p: (
            -sum(2 for kw in priority_keywords if kw in p.lower()),
            len(p)
        )
    )

    for src_rel in sorted_sources[:6]:
        src_path = dir_path / src_rel
        try:
            lines = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
            snippet = "\n".join(lines[:120])
            source_samples[src_rel] = snippet
        except Exception as e:
            logger.warning("Could not read source sample %s: %s", src_path, e)

    return {
        "project_name": dir_path.name,
        "absolute_path": str(dir_path.resolve()),
        "total_files": total_files,
        "tree_str": "\n".join(tree_lines[:60]),
        "configs": config_data,
        "source_samples": source_samples,
    }


async def analyze_local_project(
    project_path: Path,
    user_query: str = "",
    model: str | None = None
) -> tuple[str, dict[str, Any]]:
    """
    Scans the local project and generates an in-depth AI architectural analysis.
    """
    info = scan_project(project_path)
    
    # Extract package metadata if available
    tech_stack_clues: list[str] = []
    if "package.json" in info["configs"]:
        try:
            pkg = json.loads(info["configs"]["package.json"])
            deps = list(pkg.get("dependencies", {}).keys()) + list(pkg.get("devDependencies", {}).keys())
            tech_stack_clues.extend(deps[:15])
        except Exception:
            pass

    # Build context prompt for Ollama
    configs_summary = "\n\n".join([
        f"--- {fname} ---\n{content}" for fname, content in info["configs"].items()
    ])
    
    sources_summary = "\n\n".join([
        f"--- File: {fpath} ---\n{code}" for fpath, code in info["source_samples"].items()
    ])

    system_instruction = (
        "You are an expert Principal Software Architect analyzing a local project repository on the user's computer.\n"
        "Analyze the provided files and structure accurately, concisely, and specifically.\n\n"
        "STRICT RULES:\n"
        "1. Focus 100% on THIS specific project's custom codebase, pages, components, and purpose.\n"
        "2. Do NOT output external documentation links, HTML links, or generic tutorials/descriptions about libraries (do NOT explain what Next.js, React, or Tailwind is).\n"
        "3. Write in clean Markdown using **bold**, *italic*, and `code` tags. Do NOT write raw HTML tags (no <b>, <code>, <a>).\n"
        "4. Structure your response into these clean sections:\n"
        "## 📌 Project Overview\n(What this application is built for and what business problem it solves)\n\n"
        "## 🛠 Tech Stack\n(List the exact technologies used in bullet points with brief context on how they are used in this app)\n\n"
        "## 📂 Key Architecture & Components\n(Explain the core folders, key components, routes, and services found in the code)\n\n"
        "## 💡 Implemented Features\n(Specific features, multilingual translations, forms, Supabase tables, etc. found in the files)\n\n"
        "## 🔍 Code Insights & Recommendations\n(Code quality, architectural strengths, and potential improvements)"
    )

    user_content = (
        f"Project Name: {info['project_name']}\n"
        f"Path on Mac: {info['absolute_path']}\n"
        f"User question/request: \"{user_query}\"\n\n"
        f"Directory Tree:\n```\n{info['tree_str']}\n```\n\n"
        f"Key Configuration & Documentation:\n{configs_summary}\n\n"
        f"Source Code Samples:\n{sources_summary}"
    )

    target_model = model or (settings.DEEP_MODEL if "qwen2.5-coder" in settings.DEEP_MODEL else settings.DEFAULT_MODEL)
    
    logger.info("Analyzing local project %s with model %s", info['project_name'], target_model)
    analysis = await ollama_service.chat(
        user_message=f"{system_instruction}\n\n{user_content}",
        model=target_model,
        num_predict=3500,
    )

    return analysis, info
