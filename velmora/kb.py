"""Load the help-center articles (Markdown files with a small front-matter header)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    updated: date
    body: str  # article text without the header and without the "# Title" line

    @property
    def sentences(self) -> list[str]:
        return split_sentences(self.body)


def split_sentences(text: str) -> list[str]:
    # Split after . ! ? followed by whitespace. Decimals like "7.90" and times like "14:00" are safe.
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def parse_article(text: str) -> Article:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not match:
        raise ValueError("article has no front-matter header")
    header: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    body_lines = [l for l in match.group(2).splitlines() if not l.startswith("#")]
    return Article(
        id=header["id"],
        title=header["title"],
        category=header.get("category", ""),
        updated=date.fromisoformat(header["updated"]),
        body=" ".join(" ".join(body_lines).split()),
    )


def load_kb(kb_dir: str | Path) -> list[Article]:
    articles = [parse_article(p.read_text(encoding="utf-8")) for p in sorted(Path(kb_dir).glob("*.md"))]
    if not articles:
        raise FileNotFoundError(f"no articles found in {kb_dir}")
    return articles
