"""
Query the law vector database
-------------------------------
Lets you type a plain-language question and see which stored sections
come back as the closest match by MEANING (not just keyword match).

This is purely a retrieval test — no LLM involved yet. The goal here is
to confirm your embeddings + database actually find relevant sections
before we add answer-generation on top.

Usage:
    python query_test.py "What can I do if a company sells me a defective product?"
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "law_sections"
TOP_K = 3  # how many results to show


def main():
    if len(sys.argv) != 2:
        print('Usage: python query_test.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Total sections available to search: {collection.count()}")

    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
    )

    print(f"\nTop {TOP_K} matches for: \"{question}\"\n")
    print("=" * 70)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta = metadatas[i]
        print(f"\n#{i+1}  Act: {meta['act']}")
        print(f"     Chapter: {meta['chapter']}")
        print(f"     Section: {meta['section']}")
        print(f"     Match distance: {distances[i]:.4f}  (lower = closer match)")
        print(f"     Text preview: {documents[i][:200]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
