"""
Indian Law Bot — Streamlit UI
--------------------------------
A simple web interface for the RAG pipeline you already built. This file
doesn't contain any new logic — it just reuses retrieve_sections,
build_context_block, and ask_llm from ask_law_bot.py and wraps them in a
web page.

Install (one-time):
    pip install streamlit

Run it:
    streamlit run app.py

This opens a local web page in your browser (usually http://localhost:8501).
Anyone else on your network could technically visit it too if you share
the address — for your project demo, that's actually a nice bonus.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

import config

# When deployed on Streamlit Community Cloud, real secrets live in the
# app's "Secrets" settings (st.secrets), NOT in config.py (which is
# gitignored and never uploaded). Locally, config.py's values are used
# as-is. This block lets the same code work in both places.
if hasattr(st, "secrets"):
    config.LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", config.LLM_PROVIDER)
    config.LLM_API_BASE_URL = st.secrets.get("LLM_API_BASE_URL", config.LLM_API_BASE_URL)
    config.LLM_API_KEY = st.secrets.get("LLM_API_KEY", config.LLM_API_KEY)
    config.LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", config.LLM_MODEL_NAME)

from ask_law_bot import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    retrieve_sections,
    build_context_block,
    ask_llm,
)

st.set_page_config(page_title="Indian Law Bot", page_icon="⚖️")


# @st.cache_resource means: run this function once, keep the result in
# memory, and reuse it on every subsequent question instead of reloading
# the model/database each time. This is what keeps the app responsive.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


model = load_embedding_model()
collection = load_collection()

st.title("⚖️ Indian Law Bot")
st.caption(
    "Ask a question in plain language about your rights under Indian law. "
    "This tool provides general legal information, not legal advice."
)
st.sidebar.metric("Sections in database", collection.count())

question = st.text_input(
    "Your question:",
    placeholder="e.g. What can I do if a shopkeeper refuses to refund a defective product?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching relevant law sections..."):
        sections = retrieve_sections(question, model=model, collection=collection)

    with st.expander(f"📚 {len(sections)} sections used to answer this — click to view"):
        for s in sections:
            st.markdown(f"**{s['act']} — {s['chapter']} — Section {s['section']}**")
            st.write(s["text"])
            st.markdown("---")

    context = build_context_block(sections)

    with st.spinner("Generating answer..."):
        try:
            answer = ask_llm(question, context)
            st.markdown("### Answer")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Something went wrong calling the LLM: {e}")

elif ask_clicked:
    st.warning("Please type a question first.")
