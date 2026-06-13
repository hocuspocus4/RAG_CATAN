"""Lightweight RAG: BM25 retrieval + Groq-hosted generation.

Design for a small, deployable, mobile-friendly app:
  * Retrieval is lexical BM25 (rank-bm25) — no embeddings, no torch. The corpus
    is a small, keyword-rich rulebook, so exact-term matching ("robber", "longest
    road", "city cost") is both accurate and feather-light to host.
  * Generation runs on Groq's hosted LLMs (OpenAI-compatible API) so no model
    weights load locally — perfect for free hosting tiers and phones. We call the
    HTTP API with the standard library only.

Set GROQ_API_KEY (env or Streamlit secret). Without it, the app still answers in
retrieval-only mode by returning the most relevant rule passages.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from .ingest import Chunk, build_chunks

# Load a local .env (repo root) so GROQ_API_KEY works for every entry-point,
# including the Streamlit app. Harmless if python-dotenv or the file is absent.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

DATA_PDF = Path(__file__).resolve().parents[1] / "data" / "catan_rules.pdf"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a friendly Catan rules expert. Answer the player's question using ONLY "
    "the rule passages provided. Be clear and concise, and quote the exact numbers "
    "(costs, victory points, dice) when relevant. If the passages don't contain the "
    "answer, say you can't find that rule. End with the section name(s) you used."
)


def _tok(text: str) -> List[str]:
    """Tokenize with a tiny plural/stem normaliser so 'cost' matches 'costs',
    'card'/'cards', 'point'/'points' — important for lexical search on a small,
    keyword-driven corpus like a rulebook."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"      # probabilities -> probability(ish)
        elif len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s"):
            t = t[:-1]            # costs -> cost, cards -> card
        out.append(t)
    return out


@dataclass
class Source:
    chunk: Chunk
    score: float


class CatanRAG:
    def __init__(self, pdf_path: Path = DATA_PDF):
        self.chunks: List[Chunk] = build_chunks(pdf_path)
        self._bm25 = BM25Okapi([_tok(c.text + " " + c.title) for c in self.chunks])

    # ---- retrieval ----
    def retrieve(self, question: str, k: int = 3) -> List[Source]:
        scores = self._bm25.get_scores(_tok(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Source(self.chunks[i], float(scores[i])) for i in order]

    # ---- generation ----
    @staticmethod
    def groq_key() -> str:
        return os.getenv("GROQ_API_KEY", "")

    def answer(self, question: str, k: int = 4,
               model: str = "llama-3.1-8b-instant",
               temperature: float = 0.2) -> dict:
        sources = self.retrieve(question, k=k)
        context = "\n\n".join(f"{s.chunk.citation}\n{s.chunk.text}" for s in sources)
        key = self.groq_key()
        if not key:
            answer = ("(No GROQ_API_KEY set — showing the most relevant rules.)\n\n" +
                      "\n\n".join(f"{s.chunk.citation}\n{s.chunk.text}" for s in sources))
            return {"answer": answer, "sources": sources, "generated": False}

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Rule passages:\n{context}\n\nQuestion: {question}"},
            ],
        }
        req = urllib.request.Request(
            GROQ_URL, data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # A normal User-Agent avoids Cloudflare blocking the default
                # "Python-urllib/x.y" signature (seen as HTTP 403, code 1010).
                "User-Agent": "catan-rag/1.0 (+https://github.com)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"].strip()
            return {"answer": answer, "sources": sources, "generated": True}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            return {"answer": f"[Groq API error {e.code}] {detail}", "sources": sources, "generated": False}
        except Exception as e:  # network/SSL/etc.
            return {"answer": f"[Could not reach Groq: {e}]", "sources": sources, "generated": False}


if __name__ == "__main__":
    rag = CatanRAG()
    print(f"Loaded {len(rag.chunks)} rule chunks.")
    out = rag.answer("How much does a city cost and what does it do?")
    print("\n" + out["answer"])
    print("\nSources:", [s.chunk.citation for s in out["sources"]])
