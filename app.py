"""
Indian Law Bot — Streamlit UI (chat-based, multi-turn)
----------------------------------------------------------
Install (one-time):
    pip install streamlit

Run it:
    streamlit run app.py

ADDING YOUR OWN LOGO / BOT AVATAR:
Drop image files into an "assets" folder next to this script:
    assets/logo.png         -> shown in the top masthead
    assets/bot_avatar.png   -> shown next to every bot reply
If either file is missing, a sensible emoji fallback is used instead, so
the app still runs fine without them.
"""

import os
import re
import base64

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

import config

# On Streamlit Cloud, a secrets.toml (or dashboard-configured secrets)
# always exists, so this succeeds and overrides config.py's placeholders.
# Locally, there's usually no secrets.toml at all (you're using
# local_secrets.py instead) — accessing st.secrets in that case raises,
# which we catch here and simply keep config.py's own values.
try:
    config.LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", config.LLM_PROVIDER)
    config.LLM_API_BASE_URL = st.secrets.get("LLM_API_BASE_URL", config.LLM_API_BASE_URL)
    config.LLM_API_KEY = st.secrets.get("LLM_API_KEY", config.LLM_API_KEY)
    config.LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", config.LLM_MODEL_NAME)
except Exception:
    pass

from ask_law_bot import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    retrieve_sections,
    build_context_block,
    ask_llm,
)

st.set_page_config(page_title="Act Wise — Indian Law RAG Bot", page_icon="⚖️", layout="centered")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BOT_AVATAR_PATH = os.path.join(ASSETS_DIR, "bot_avatar.png")

BOT_AVATAR = BOT_AVATAR_PATH if os.path.exists(BOT_AVATAR_PATH) else "⚖️"


def get_base64_image(path):
    """Read a local image and return it as a data URI, or None if the
    file doesn't exist — used to embed the logo inside custom HTML."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = path.rsplit(".", 1)[-1]
    return f"data:image/{ext};base64,{encoded}"


def section_sort_key(section_str):
    """Sort section numbers naturally: 2, 2A, 9B, 10, 65, 100 — instead
    of alphabetically, which would wrongly put '10' before '2'."""
    m = re.match(r'(\d+)([A-Za-z]*)', str(section_str))
    if m:
        return (int(m.group(1)), m.group(2))
    return (999999, str(section_str))


def get_available_acts(collection) -> list:
    """Distinct Act names currently in the database, pulled live — so
    this list grows automatically as you embed more Acts, no code
    changes needed."""
    all_data = collection.get(include=["metadatas"])
    if not all_data["metadatas"]:
        return []
    return sorted(set(m["act"] for m in all_data["metadatas"]))


def get_act_sections(collection, act_name: str) -> list:
    """All sections belonging to one Act, sorted in reading order."""
    result = collection.get(
        where={"act": act_name},
        include=["metadatas", "documents"],
    )
    sections = [
        {"section": m["section"], "chapter": m["chapter"], "text": doc}
        for m, doc in zip(result["metadatas"], result["documents"])
    ]
    return sorted(sections, key=lambda s: section_sort_key(s["section"]))


# ---------------------------------------------------------------------------
# Visual design
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
    padding: 1.5rem 1.5rem 1.5rem 1.5rem;
    border-bottom: 3px solid #C98A2B;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.masthead img {
    height: 48px;
    width: 48px;
    object-fit: contain;
    border-radius: 8px;
}
.masthead .masthead-emoji-fallback {
    font-size: 2.2rem;
    line-height: 1;
}
.masthead h1 {
    font-family: 'IBM Plex Serif', serif;
    color: #FBF9F4;
    font-size: 1.85rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: 0.01em;
}
.masthead p {
    color: #C9CEDA;
    margin: 0.3rem 0 0 0;
    font-size: 0.92rem;
}

/* Chat messages: flat blocks, user on the right, bot on the left */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0.75rem 0 !important;
    margin-bottom: 0.25rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #14213D;
    padding-left: 1rem !important;
    justify-content: flex-start;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-right: 3px solid #33513A;
    padding-right: 1rem !important;
    flex-direction: row-reverse;
    text-align: right;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    text-align: right;
}

/* Buttons */
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
.sidebar-acts-heading {
    font-family: 'IBM Plex Serif', serif;
    font-size: 0.95rem;
    color: #14213D;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    border-top: 1px solid #E4DFD3;
    padding-top: 1rem;
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

# Masthead — logo if present, emoji fallback otherwise
logo_data_uri = get_base64_image(LOGO_PATH)
logo_html = (
    f'<img src="{logo_data_uri}" alt="logo">'
    if logo_data_uri
    else '<span class="masthead-emoji-fallback">⚖️</span>'
)
st.markdown(f"""
<div class="masthead">
    {logo_html}
    <div>
        <h1>Act Wise</h1>
        <p>Ask in plain language. Answers are grounded in Indian statutes, with citations — this is legal information, not legal advice.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if st.sidebar.button("🗂️ New Case", use_container_width=True):
    st.session_state.messages = []
    st.session_state.view = "chat"
    st.rerun()

st.sidebar.metric("Sections in database", collection.count())

available_acts = get_available_acts(collection)
st.sidebar.markdown('<p class="sidebar-acts-heading">Acts loaded</p>', unsafe_allow_html=True)

if not available_acts:
    st.sidebar.caption("No Acts embedded yet.")
else:
    for act_name in available_acts:
        if st.sidebar.button(act_name, key=f"act_btn_{act_name}", use_container_width=True):
            st.session_state.view = f"read_act::{act_name}"
            st.rerun()

st.sidebar.caption("More Acts will appear here automatically as they're added to the database.")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "view" not in st.session_state:
    st.session_state.view = "chat"

# ---------------------------------------------------------------------------
# Main area: either the chat, or an Act reader
# ---------------------------------------------------------------------------
if st.session_state.view.startswith("read_act::"):
    act_name = st.session_state.view.split("::", 1)[1]

    if st.button("← Back to chat"):
        st.session_state.view = "chat"
        st.rerun()

    st.markdown(f"## {act_name}")
    sections = get_act_sections(collection, act_name)
    st.caption(f"{len(sections)} sections")

    current_chapter = None
    for s in sections:
        if s["chapter"] and s["chapter"] != current_chapter:
            current_chapter = s["chapter"]
            st.markdown(f"#### {current_chapter}")
        with st.expander(f"Section {s['section']}"):
            st.write(s["text"])

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=BOT_AVATAR if msg["role"] == "assistant" else None):
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

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Searching relevant law sections..."):
                sections = retrieve_sections(question, model=model, collection=collection)
                context = build_context_block(sections)

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