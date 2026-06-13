"""Catan Rules Assistant — a mobile-friendly RAG chat app.

Lexical retrieval over the rulebook + Groq-hosted generation. Runs great on a
phone browser and deploys free to Streamlit Community Cloud.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.rag import CatanRAG

# "centered" layout is the mobile-friendly default (single readable column).
st.set_page_config(page_title="Catan Rules Assistant", page_icon="🎲", layout="centered")

# Pull the Groq key from Streamlit secrets (cloud) or env (local) into the env
# so src.rag picks it up uniformly. Accessing st.secrets with no secrets file
# can raise, so guard it.
try:
    if not os.getenv("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

GROQ_MODELS = {
    "Llama 3.1 8B — fast (recommended)": "llama-3.1-8b-instant",
    "Llama 3.3 70B — best quality": "llama-3.3-70b-versatile",
    "Gemma2 9B": "gemma2-9b-it",
}


@st.cache_resource(show_spinner="Loading the rulebook…")
def get_rag() -> CatanRAG:
    return CatanRAG()


rag = get_rag()

st.title("🎲 Catan Rules Assistant")
st.caption("Ask anything about the rules of CATAN. Answers are grounded in the rulebook, with the rule sections cited.")

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Groq model", list(GROQ_MODELS.keys()), index=0)
    k = st.slider("Rule passages used", 2, 6, 4)
    st.divider()
    if rag.groq_key():
        st.success("Groq key detected ✓")
    else:
        st.warning("No Groq key — answering in retrieval-only mode. "
                   "Add GROQ_API_KEY to enable generated answers.")
    st.caption("Get a free key at console.groq.com")

EXAMPLES = [
    "How do I win the game?",
    "What does a city cost and what does it do?",
    "What happens when someone rolls a 7?",
    "How does the Longest Road work?",
    "How does trading with the bank work?",
    "What do development cards do?",
]

# --- chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick example chips (handy on mobile).
st.write("**Try an example:**")
cols = st.columns(2)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, use_container_width=True, key=f"ex{i}"):
        clicked = ex

# Render history.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s.chunk.citation}**  \n{s.chunk.text}")

prompt = st.chat_input("Ask a Catan rules question…") or clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking the rulebook…"):
            out = rag.answer(prompt, k=k, model=GROQ_MODELS[model_label])
        st.markdown(out["answer"])
        with st.expander("Sources"):
            for s in out["sources"]:
                st.markdown(f"**{s.chunk.citation}**  \n{s.chunk.text}")
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
    )
