"""Embed document chunks (see src/geokg/documents.py) into a local Qdrant collection.

Qdrant runs in on-disk local mode (so we dont need a server, or Docker, in this case). 
As I understand it, each point stores the chunk text and its where it comes from, so retrieval can filter by document and return citable metadata.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from . import config, documents


def get_client() -> QdrantClient:
    """Open the on-disk Qdrant store."""
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def build_index() -> int:
    """(Re)build the chunk collection. Returns the number of chunks indexed."""
    chunks = documents.iter_chunks()
    client = get_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)  # idempotent rebuild
    client.create_collection(
        config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(config.EMBED_MODEL),
            distance=models.Distance.COSINE,
        ),
    )
    client.upsert(config.QDRANT_COLLECTION, points=[
        models.PointStruct(
            id=i,
            vector=models.Document(text=c.text, model=config.EMBED_MODEL),
            payload={"doc_id": c.doc_id, "title": c.title, "url": c.url,
                     "region": c.region, "chunk_index": c.index, "text": c.text},
        )
        for i, c in enumerate(chunks)
    ])
    return len(chunks)
