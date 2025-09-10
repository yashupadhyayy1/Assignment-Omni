import sys
from pathlib import Path

from assignment_omni.rag.retriever import prepare_corpus

# Default PDF path
DEFAULT_PDF = Path(r"E:\Projects\assignment\Assignment-Omni\EMROPUB_2019_en_23536.pdf")


def main() -> None:
    # Use argument if provided, otherwise fallback to default
    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = DEFAULT_PDF

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
