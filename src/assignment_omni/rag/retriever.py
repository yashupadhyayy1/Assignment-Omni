from __future__ import annotations

from typing import List

from langchain.schema import Document

from .pdf_utils import load_pdf, split_documents


def prepare_corpus(pdf_path: str) -> List[Document]:
    docs = load_pdf(pdf_path)
    return split_documents(docs)


