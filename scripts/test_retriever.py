import sys
from pathlib import Path

from assignment_omni.rag.retriever import prepare_corpus


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_retriever.py <path_to_pdf>")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    print(f"Preparing corpus from: {pdf_path}")
    chunks = prepare_corpus(str(pdf_path))
    print(f"Prepared {len(chunks)} chunks")

    for i, d in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(d.page_content[:500].replace("\n", " "))


if __name__ == "__main__":
    main()

