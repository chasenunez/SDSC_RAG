"""Load the document corpus and split it into chunks for embedding.

Documents live in ``data/documents/manifest.json`` as source-grounded summaries
with verified metadata. Keeping them small means the whole corpus is in the repo
and the RAG layer needs no scraping step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config

_MANIFEST = config.DOCS_DIR / "manifest.json"


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a slice of a document and wheare it cme from."""

    doc_id: str
    title: str
    url: str
    region: str
    index: int  # position of the chunk within its document
    text: str


def load_documents() -> list[dict]:
    """Return the document records from the manifest."""
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return data["documents"]


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries.

    Packing whole sentences keeps chunks readable; the small overlap stops a
    fact that straddles a boundary from being lost. Documents here are short, so
    im getting one or two chunks each.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            # Carry a little context, but start the overlap at a word boundary so
            # the next chunk does not begin mid-word.
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def iter_chunks() -> list[Chunk]:
    """Flatten the corpus into chunks ready for indexing."""
    out: list[Chunk] = []
    for doc in load_documents():
        for i, piece in enumerate(chunk_text(doc["content"])):
            out.append(Chunk(
                doc_id=doc["id"],
                title=doc["title"],
                url=doc["url"],
                region=doc["region"],
                index=i,
                text=piece,
            ))
    return out
