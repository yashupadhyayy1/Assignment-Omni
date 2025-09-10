from __future__ import annotations

from typing import List, Optional

from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.storage import InMemoryStore
from qdrant_client.models import Distance, VectorParams


def build_embeddings() -> CacheBackedEmbeddings:
    base = FastEmbedEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(base, InMemoryStore())


def get_qdrant(client: QdrantClient, collection_name: str, embeddings) -> Qdrant:
    """Return a Qdrant vector store, creating the collection if needed.

    We detect the embedding dimensionality from the provided embeddings implementation
    and create the collection with cosine distance if it does not already exist.
    """
    # Ensure collection exists
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        # Compute embedding size by running a single query embedding
        sample_vector = embeddings.embed_query("dimension probe")
        vector_size = len(sample_vector)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # For langchain-qdrant>=0.2.0 the kwarg is 'embedding'
    try:
        return Qdrant(client=client, collection_name=collection_name, embedding=embeddings)
    except TypeError:
        # Fallback for older versions
        return Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)


def upsert_documents(store: Qdrant, docs: List[Document]) -> None:
    store.add_documents(docs)


def similarity_search(store: Qdrant, query: str, k: int = 5) -> List[Document]:
    return store.similarity_search(query, k=k)


