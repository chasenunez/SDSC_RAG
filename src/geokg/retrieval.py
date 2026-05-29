"""OK KG time. The graph chooses the documents, vectors rank chunks.

here the code is only answering for a region. We first walk the knowledge graph to find
the documents attached to that region and its parents, then run vector search
*restricted to those documents*. The graph therefore decides what is in scope;
the embeddings only rank within that scope. The ``use_graph`` switch runs the
same query without the filter so the difference can be shown side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import models

from . import config, graph, index


@dataclass
class Retrieval:
    """Everything the answer step and the dashboard panel need."""

    region_id: str
    candidate_documents: list[dict]  # from the graph traversal
    datasets: list[dict]             # datasets covering the region
    chunks: list[dict]               # ranked chunks with provenance
    used_graph: bool = True
    candidate_doc_ids: list[str] = field(default_factory=list)


def retrieve(question: str, region_id: str, top_k: int = 5, use_graph: bool = True,
             store=None, client=None) -> Retrieval:
    """Run graph-filtered (or, for comparison, unfiltered) chunk retrieval.

    ``store`` and ``client`` can be passed in so a long-running app reuses one
    graph and one Qdrant handle instead of rebuilding/reopening per query.
    """
    store = store or graph.build_graph()
    candidates = graph.documents_for_region(store, region_id)
    datasets = graph.datasets_for_region(store, region_id)
    candidate_ids = [c["doc_id"] for c in candidates]

    query_filter = None
    if use_graph:
        # Restrict vector search to the documents the graph selected.
        query_filter = models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=candidate_ids))
        ])

    client = client or index.get_client()
    hits = client.query_points(
        config.QDRANT_COLLECTION,
        query=models.Document(text=question, model=config.EMBED_MODEL),
        query_filter=query_filter,
        limit=top_k,
    ).points
    chunks = [{
        "doc_id": h.payload["doc_id"],
        "title": h.payload["title"],
        "url": h.payload["url"],
        "region": h.payload["region"],
        "text": h.payload["text"],
        "score": round(h.score, 4),
    } for h in hits]

    return Retrieval(
        region_id=region_id,
        candidate_documents=candidates,
        datasets=datasets,
        chunks=chunks,
        used_graph=use_graph,
        candidate_doc_ids=candidate_ids,
    )
