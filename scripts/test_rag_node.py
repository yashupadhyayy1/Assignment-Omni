import sys
from pathlib import Path

from assignment_omni.graph.nodes import setup_rag_corpus, rag_node
from assignment_omni.config.settings import Settings


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: uv run python scripts/test_rag_node.py <path_to_pdf> <question>")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    question = " ".join(sys.argv[2:])

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    print("Loading settings and preparing vector store...")
    cfg = Settings.load()
    print(f"Qdrant URL: {cfg.qdrant.url} | Collection: {cfg.qdrant.collection}")

    print(f"Setting up RAG corpus from: {pdf_path}")
    setup_rag_corpus(str(pdf_path))

    print(f"Querying rag_node with question: {question!r}")
    result = rag_node({"query": question})
    print("Route:", result.get("route"))
    print("Answer:")
    print(result.get("result"))


if __name__ == "__main__":
    main()
