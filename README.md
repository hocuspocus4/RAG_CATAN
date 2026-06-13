# 🎲 Catan Rules Assistant — a tiny, deployable RAG app

Ask any question about the rules of **CATAN** and get a grounded answer that cites the
exact rule sections. Built as a lightweight RAG app that runs on your phone: **lexical
retrieval over the rulebook + a Groq-hosted LLM** for the answers. No GPU, no model
downloads — it installs in seconds and deploys free.

> A fun, minimal companion to the larger FATF-RAG project. Same idea (retrieve → ground →
> answer with citations), but stripped down for size and mobile use.

## Why it's built this way

- **Lexical retrieval (BM25), not embeddings.** The corpus is a small, keyword-rich
  rulebook, so exact-term matching ("robber", "longest road", "city cost") is both accurate
  and feather-light. Skipping neural embeddings means **no PyTorch** — the whole app is a
  few MB and fits free hosting tiers and phones. A tiny plural-normaliser in the tokenizer
  makes "cost"/"costs" and "card"/"cards" match.
- **Hosted generation via Groq.** The LLM runs on Groq's servers (OpenAI-compatible API),
  so nothing heavy loads locally — ideal for a phone or a 1 GB cloud instance. Called with
  the Python standard library, so there's no SDK to install.
- **Grounded + cited.** Every answer is built only from retrieved rule passages, and the
  app shows the exact sections (and page) it used.
- **Degrades gracefully.** No Groq key? The app still runs and returns the most relevant
  rule passages (retrieval-only mode).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a free key at https://console.groq.com, then:
cp .env.example .env        # put your GROQ_API_KEY inside

streamlit run app/streamlit_app.py      # opens http://localhost:8501
# or test in the terminal:
python cli.py "What happens when someone rolls a 7?"
```

## Put it on your phone (deploy free to Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**.
2. Go to **share.streamlit.io** → *New app* → pick your repo and `app/streamlit_app.py`.
3. In the app's **Settings → Secrets**, add your key:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

4. Deploy. You'll get a public `https://…streamlit.app` URL — open it on your phone (and
   "Add to Home Screen" for an app-like icon). The layout is mobile-first (single column,
   tap-able example chips, chat input).

Hugging Face Spaces (Streamlit SDK) works the same way if you prefer it — add `GROQ_API_KEY`
as a Space secret.

## Swap in a different / fuller rulebook

The app reads `data/catan_rules.pdf`. Replace it with any rules PDF (or regenerate the
included one from `data/catan_rules_source.md`) and restart — ingestion and retrieval adapt
automatically.

## Project layout

```
catan-rag/
├── data/
│   ├── catan_rules.pdf         # the corpus (generated from the source below)
│   └── catan_rules_source.md   # the original rules text (edit + re-render)
├── src/
│   ├── ingest.py               # PDF -> section-tagged chunks
│   └── rag.py                  # BM25 retrieval + Groq generation
├── app/streamlit_app.py        # mobile-friendly chat UI
├── cli.py                      # terminal tester
├── .streamlit/
│   ├── config.toml             # theme
│   └── secrets.toml.example    # how to set the key on the cloud
├── requirements.txt
└── README.md
```

## Notes & honesty

The bundled rules text is an **original, paraphrased reference** to the base game's
mechanics (for the standard 3–4 player game), written for this demo — it is not the
publisher's copyrighted rulebook. For edge cases, always check the official rules. CATAN is
a trademark of its respective owner; this project is an unaffiliated educational demo.

## AI usage

Built with AI assistance (Claude) for the app scaffolding, the retrieval/Groq glue code, and
this README. The design choices — lexical retrieval for a small corpus, hosted generation for
deployability, the grounding/citation approach — are the author's.
