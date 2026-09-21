"""
Indian Law Bot — Streamlit UI (chat-based, multi-turn)
----------------------------------------------------------
A conversational web interface for the RAG pipeline. Supports follow-up
questions — the whole conversation is sent to the LLM each time, not just
the latest message.

Install (one-time):
    pip install streamlit

Run it:
    streamlit run app.py
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

import config

# When deployed on Streamlit Community Cloud, real secrets live in the
# app's "Secrets" settings (st.secrets), NOT in config.py. Locally,
# config.py's own values (loaded from local_secrets.py) are used as-is.
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

st.set_page_config(page_title="Act Wise — Indian Law Bot", page_icon="⚖️", layout="centered")

# ---------------------------------------------------------------------------
# Visual design: a "gazette masthead" identity — dark navy header with serif
# type (echoing how Indian bare acts are published in The Gazette of India),
# flat document-style message blocks instead of rounded chat bubbles, and a
# single deliberate accent color (saffron gold) used only for emphasis.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #232B3A;
}

.block-container {
    padding-top: 0 !important;
    max-width: 760px;
}

/* Masthead */
.masthead {
    background-color: #14213D;
    margin: 0 -1rem 1.5rem -1rem;
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    border-bottom: 3px solid #C98A2B;
}
.masthead h1 {
    font-family: 'IBM Plex Serif', serif;
    color: #FBF9F4;
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: 0.01em;
}
.masthead p {
    color: #C9CEDA;
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
}

/* Chat messages: flat blocks with a left border by role, not bubbles */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0.75rem 0 0.75rem 1rem !important;
    margin-bottom: 0.5rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #33513A;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #14213D;
}

/* Buttons and links: saffron accent used deliberately, once */
.stButton > button, [data-testid="stChatInputSubmitButton"] {
    background-color: #C98A2B;
    color: #FBF9F4;
    border: none;
}
.stButton > button:hover, [data-testid="stChatInputSubmitButton"]:hover {
    background-color: #B37A24;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FBF9F4;
    border-right: 1px solid #E4DFD3;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


model = load_embedding_model()
collection = load_collection()

st.markdown("""
<div class="masthead">
    <h1>Act Wise</h1>
    <p>Ask in plain language. Answers are grounded in Indian statutes, with citations — this is legal information, not legal advice.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.metric("Sections in database", collection.count())
if st.sidebar.button("Start a new conversation"):
    st.session_state.messages = []
    st.rerun()

# session_state.messages holds the DISPLAY conversation (plain question/
# answer text). A separate, longer version (with retrieved law text
# injected) is built just before calling the LLM — see below.
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sections"):
            with st.expander(f"{len(msg['sections'])} sections used to answer this"):
                for s in msg["sections"]:
                    st.markdown(f"**{s['act']} — {s['chapter']} — Section {s['section']}**")
                    st.write(s["text"])
                    st.markdown("---")

question = st.chat_input("e.g. What can I do if a shopkeeper refuses to refund a defective product?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant law sections..."):
            sections = retrieve_sections(question, model=model, collection=collection)
            context = build_context_block(sections)

        # Build the LLM's view of the conversation: prior turns exactly as
        # displayed, plus the new turn with retrieved law text injected so
        # the model has fresh grounding for this specific question.
        llm_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        llm_messages.append({
            "role": "user",
            "content": (
                f"Relevant law sections:\n\n{context}\n\n"
                f"---\n\nUser's question: {question}"
            ),
        })

        with st.spinner("Thinking..."):
            try:
                answer = ask_llm(llm_messages)
            except Exception as e:
                answer = f"Something went wrong calling the LLM: {e}"

        st.markdown(answer)
        with st.expander(f"{len(sections)} sections used to answer this"):
            for s in sections:
                st.markdown(f"**{s['act']} — {s['chapter']} — Section {s['section']}**")
                st.write(s["text"])
                st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sections": sections,
    })