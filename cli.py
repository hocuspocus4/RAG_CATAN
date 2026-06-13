"""Quick terminal tester (no UI):  python cli.py "How much does a city cost?" """
from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.rag import CatanRAG


def main() -> None:
    rag = CatanRAG()
    q = " ".join(sys.argv[1:]) or "How do I win the game?"
    out = rag.answer(q)
    print("\nQ:", q)
    print("\n" + out["answer"])
    print("\nSources:", ", ".join(s.chunk.citation for s in out["sources"]))


if __name__ == "__main__":
    main()
