import sys
from pathlib import Path

from assignment_omni.config.settings import Settings
from assignment_omni.vectorstore.qdrant_store import build_embeddings, get_qdrant, upsert_documents, similarity_search
from assignment_omni.rag.retriever import prepare_corpus
from qdrant_client import QdrantClient


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_qdrant_store.py <path_to_pdf>")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    cfg = Settings.load()

    print(f"Connecting to Qdrant at {cfg.qdrant.url} (collection: {cfg.qdrant.collection})")
    client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)
    embeddings = build_embeddings()
    store = get_qdrant(client, cfg.qdrant.collection, embeddings)

    print("Preparing corpus and upserting...")
    docs = prepare_corpus(str(pdf_path))
    upsert_documents(store, docs)
    print(f"Upserted {len(docs)} chunks")

    print("Running similarity search for a sample query: 'introduction'")
    results = similarity_search(store, "introduction", k=3)
    print(f"Retrieved {len(results)} docs")
    for i, d in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(d.page_content[:300].replace("\n", " "))


if __name__ == "__main__":
    main()

