import sys
from pathlib import Path

from assignment_omni.rag.pdf_utils import load_pdf, split_documents


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_pdf_utils.py <path_to_pdf>")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf(str(pdf_path))
    print(f"Loaded {len(docs)} pages")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    for i, d in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(d.page_content[:500].replace("\n", " "))


if __name__ == "__main__":
    main()

