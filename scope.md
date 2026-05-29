# Scope

This document scopes a working vertical slice of a geospatial RAG + knowledge graph platform, built
to answer one question end-to-end: "how has urban heat changed in (1 of 5 major swiss cities) over the last
decade?" In this design the graph is the gatekeeper — a question resolves to a region, the graph
selects the documents attached to that region or its ancestors, and vector search ranks chunks only
within that set. The sections below describe what was built fully, what was deliberately
simplified, and what was left out, with the reasoning for each.

## Built fully

- **Geospatial data.** Real surface temperature from Landsat Collection 2 Level-2 on Microsoft
    Planetary Computer, with USGS scale/offset to Celsius and pixel-level QA masking (fill, dilated
    cloud, cirrus, cloud, shadow). Projections are explicit: each scene is read in its native UTM,
    the city bbox is reprojected from WGS84 into the scene's CRS for a windowed read, and the
    overlay is reprojected back to WGS84 for display. Swiss boundaries come from swissBOUNDARIES3D
    (swisstopo, EPSG:2056), selected by federal BFS number with a name cross-check.
- **Knowledge graph.** Oxigraph RDF store with `Region`, `Dataset`, and `Document` nodes and
    `within`, `describesRegion`, `coversRegion`, and `delineatedBy` edges. A `geo:within*` SPARQL
    traversal walks city → canton → country in one query.
- **RAG layer.** Document chunks embedded into a local Qdrant collection. Retrieval applies a
    `MatchAny` filter over the graph-selected `doc_id`s before vector search. The dashboard exposes
    a toggle to disable the filter so the graph's effect is demonstrable side-by-side.
- **Answer generation.** pydantic-ai agent over OpenRouter with a structured `Answer` schema. A
    grounding guard drops any citation id the model returns that was not in the supplied SOURCES,
    so a citation can never point at a document the model did not see.
- **Dashboard.** Streamlit + folium with one shared city selection driving a map (boundary + LST
    overlay), a graph-context panel (which documents, at which hierarchy level), and the cited
    answer.

## Simplified

- **Five Swiss cities only** (Zürich, Genève, Basel, Bern, Lausanne). A small region set lets each
    city carry distinct, real documents rather than pretending a tiny corpus covers a country. The
    hierarchy already supports canton and national rollups; only the data-loading step would
    change.
- **One clear-sky summer Landsat scene per year**, not a monthly composite. Each value carries that
    day's weather, so the decadal slope is reported with its year-to-year spread and labeled
    indicative. This is the single biggest honest weakness and the first thing to replace with a
    monthly median.
- **Seven curated documents** (real titles, publishers, years, URLs) stored as source-grounded
    summaries in `data/documents/manifest.json`, not full PDF ingestion. Keeps the repo small and
    license-clean.
- **Deterministic templated fallback** when no `OPENROUTER_API_KEY` is set or the LLM call fails, so
    the narrative panel is never empty for a reviewer without an account.

## Deliberately left out

Canton and national scale, sensor time series, multi-narrative dashboards, a real GIS stack, and
user auth. The graph hierarchy already supports rolling up to canton and country; that extension
would live in the data-loading step, not the retrieval. The rest adds breadth, not new design.