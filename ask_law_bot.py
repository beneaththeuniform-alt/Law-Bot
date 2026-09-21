"""
Ask the law bot a question — full RAG pipeline (provider-flexible)
--------------------------------------------------------------------
Retrieves the most relevant law sections for your question from the local
Chroma database, then sends them to an LLM to generate a grounded answer.

Which LLM you use is controlled entirely by config.py — this script
itself doesn't need to change when you switch providers.

Install (one-time):
    pip install chromadb sentence-transformers requests
    (only install "anthropic" too if you plan to use that provider)

Usage:
    python ask_law_bot.py "What can I do if a company sells me a defective product?"
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer
import requests

import config

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "law_sections"
TOP_K = 4  # how many sections to retrieve and hand to the LLM

SYSTEM_PROMPT = """You are a legal information assistant for Indian citizens. \
You are NOT a lawyer and must never claim to be one or predict case outcomes.

Rules you must follow:
1. Answer ONLY using the law sections provided below. Do not use any outside \
knowledge of Indian law.
2. If the provided sections don't actually contain the answer, say so clearly \
instead of guessing.
3. Always cite the specific Act and Section number(s) you are basing the \
answer on.
4. End every answer with a short reminder to consult a lawyer or their \
nearest legal aid clinic (via NALSA) for advice specific to their situation.
5. Keep the tone plain and accessible — the person asking is not a lawyer.
"""


def retrieve_sections(question: str, model=None, collection=None) -> list:
    """Find the TOP_K most relevant sections for a question.

    model/collection can be passed in already-loaded (the Streamlit UI
    does this, so it only loads them once instead of on every question).
    If omitted, they're loaded fresh — used when running this script
    directly from the command line.
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)

    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
    )

    sections = []
    for i in range(len(results["ids"][0])):
        sections.append({
            "act": results["metadatas"][0][i]["act"],
            "chapter": results["metadatas"][0][i]["chapter"],
            "section": results["metadatas"][0][i]["section"],
            "text": results["documents"][0][i],
        })
    return sections


def build_context_block(sections: list) -> str:
    blocks = []
    for s in sections:
        blocks.append(
            f"[{s['act']} — {s['chapter']} — Section {s['section']}]\n{s['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def ask_openai_compatible(user_message: str) -> str:
    """Works with Groq, OpenRouter, local Ollama, Together AI, and any
    other provider that copies OpenAI's chat completions API format."""
    url = f"{config.LLM_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 2048,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if not response.ok:
        print(f"\n--- LLM API ERROR ---")
        print(f"URL: {url}")
        print(f"Status: {response.status_code}")
        print(f"Response body: {response.text}")
        print(f"---------------------\n")
    response.raise_for_status()
    data = response.json()

    finish_reason = data["choices"][0].get("finish_reason")
    if finish_reason == "length":
        print("NOTE: the response was cut off by the token limit — "
              "consider raising max_tokens further if this keeps happening.")

    return data["choices"][0]["message"]["content"]


def ask_anthropic(user_message: str) -> str:
    import anthropic  # imported here so it's only required if you use this provider
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL_NAME,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def ask_llm(question: str, context: str) -> str:
    user_message = (
        f"Relevant law sections:\n\n{context}\n\n"
        f"---\n\nUser's question: {question}"
    )

    if config.LLM_PROVIDER == "openai_compatible":
        return ask_openai_compatible(user_message)
    elif config.LLM_PROVIDER == "anthropic":
        return ask_anthropic(user_message)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' in config.py — "
            f"use 'openai_compatible' or 'anthropic'."
        )


def main():
    if len(sys.argv) != 2:
        print('Usage: python ask_law_bot.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    print(f"Using provider: {config.LLM_PROVIDER}")
    print("Retrieving relevant sections...")
    sections = retrieve_sections(question)
    print(f"Found {len(sections)} relevant sections:")
    for s in sections:
        print(f"  - {s['act']}, Section {s['section']}")

    context = build_context_block(sections)

    print("\nAsking the LLM...\n")
    answer = ask_llm(question, context)

    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
