"""Ingest the Catan rulebook PDF into small, section-tagged chunks.

The corpus is tiny and well structured (one topic per section), so we rebuild
those sections from the PDF text: a short line that doesn't end in a period is
treated as a heading, and the lines beneath it form that section's chunk. Each
chunk keeps its heading (for nice citations) and page number.

No heavy dependencies — just pdfplumber.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class Chunk:
    id: int
    title: str          # section heading, e.g. "Rolling a 7 and the robber"
    text: str
    page: int

    @property
    def citation(self) -> str:
        return f"[{self.title} · p.{self.page}]"


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60:
        return False
    if s.endswith((".", ":", ";", ",")):
        return False
    # Headings are a few words, mostly letters, no sentence punctuation.
    words = s.split()
    return 1 <= len(words) <= 7 and sum(c.isalpha() or c.isspace() for c in s) >= len(s) - 2


def build_chunks(pdf_path: str | Path) -> List[Chunk]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Corpus not found at {pdf_path}. Place the rulebook PDF in data/.")

    chunks: List[Chunk] = []
    title = "Catan Rules"
    buf: List[str] = []
    page_of_title = 1

    def flush():
        nonlocal buf
        text = " ".join(" ".join(buf).split())
        if len(text) >= 40:
            chunks.append(Chunk(id=len(chunks), title=title, text=text, page=page_of_title))
        buf = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for raw in (page.extract_text() or "").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if _is_heading(line):
                    flush()
                    title = line
                    page_of_title = pno
                else:
                    buf.append(line)
    flush()
    return chunks


if __name__ == "__main__":
    cs = build_chunks(Path(__file__).resolve().parents[1] / "data" / "catan_rules.pdf")
    print(f"{len(cs)} chunks")
    for c in cs[:4]:
        print(f"\n{c.citation}\n{c.text[:160]}")
