"""so this is the knowledge graph thAT is linking regions, datasets, and documents.

Built as RDF triples in an in-memory Oxigraph store. The graph is the entry
point for retrieval: a question resolves to a region, and a SPARQL traversal
walks the administrative hierarchy (city within canton within country) to find
the documents and datasets attached to that region or any of its parents. This
hierarchy walk is what lets a city-level question pull in canton and national
material without listing those links by hand.

Node types:  Region (country/canton/city), Dataset, Document.
Edge types:  within (Region->Region), describesRegion (Document->Region),
             coversRegion (Dataset->Region), delineatedBy (Region->Dataset).

The architecture of this code (for the knoweldge graph with Oxigraph) was built using Claude Opus 4.7 (model ID claude-opus-4-7), running via Claude Code. Where possible claude was constrained to code review, and not code creation. 
"""

from __future__ import annotations

from urllib.parse import quote

import pyoxigraph as ox

from . import boundaries, config, documents

# Ontology and instance namespaces.
ONT = "https://geokg.local/ontology#"
R_NS = "https://geokg.local/region/"
DS_NS = "https://geokg.local/dataset/"
DOC_NS = "https://geokg.local/document/"
RDF_TYPE = ox.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_LABEL = ox.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")


def _n(ns: str, local: str) -> ox.NamedNode:
    """Build a node IRI, percent-encoding the local part (city names have ü)."""
    return ox.NamedNode(ns + quote(local, safe=""))


def region_uri(region_id: str) -> ox.NamedNode:
    """Region IRI for a country code, canton abbreviation, or city name."""
    return _n(R_NS, region_id)


def _ont(term: str) -> ox.NamedNode:
    return ox.NamedNode(ONT + term)


def build_graph() -> ox.Store:
    """Assemble the RDF graph from config, boundaries, and the manifest."""
    store = ox.Store()

    def triple(s, p, o):
        store.add(ox.Quad(s, p, o))

    # Regions: country, the five cantons, the five cities, with the hierarchy.
    country = region_uri("CH")
    triple(country, RDF_TYPE, _ont("Region"))
    triple(country, _ont("level"), ox.Literal("country"))
    triple(country, RDFS_LABEL, ox.Literal("Switzerland"))

    for city in config.CITIES:
        canton = region_uri(city.canton)
        triple(canton, RDF_TYPE, _ont("Region"))
        triple(canton, _ont("level"), ox.Literal("canton"))
        triple(canton, RDFS_LABEL, ox.Literal(city.canton))
        triple(canton, _ont("within"), country)

        city_node = region_uri(city.name)
        bbox = boundaries.city_bbox(city.name)
        triple(city_node, RDF_TYPE, _ont("Region"))
        triple(city_node, _ont("level"), ox.Literal("city"))
        triple(city_node, RDFS_LABEL, ox.Literal(city.name))
        triple(city_node, _ont("within"), canton)
        triple(city_node, _ont("bbox"), ox.Literal(",".join(str(v) for v in bbox)))

    # Datasets and what they cover.
    boundaries_ds = _n(DS_NS, "swissboundaries3d")
    triple(boundaries_ds, RDF_TYPE, _ont("Dataset"))
    triple(boundaries_ds, RDFS_LABEL, ox.Literal("swissBOUNDARIES3D (swisstopo)"))
    triple(boundaries_ds, _ont("coversRegion"), country)

    lst_ds = _n(DS_NS, "landsat-lst")
    triple(lst_ds, RDF_TYPE, _ont("Dataset"))
    triple(lst_ds, RDFS_LABEL, ox.Literal("Landsat C2 L2 land surface temperature"))
    triple(lst_ds, _ont("temporalStart"), ox.Literal(str(min(config.LST_YEARS))))
    triple(lst_ds, _ont("temporalEnd"), ox.Literal(str(max(config.LST_YEARS))))

    for city in config.CITIES:
        city_node = region_uri(city.name)
        # Region geometries are derived from the boundary dataset.
        triple(city_node, _ont("delineatedBy"), boundaries_ds)
        # The heat dataset has a per-city spatial extent.
        triple(lst_ds, _ont("coversRegion"), city_node)

    # Documents and the region each describes.
    for doc in documents.load_documents():
        node = _n(DOC_NS, doc["id"])
        triple(node, RDF_TYPE, _ont("Document"))
        triple(node, RDFS_LABEL, ox.Literal(doc["title"]))
        triple(node, _ont("url"), ox.Literal(doc["url"]))
        triple(node, _ont("publisher"), ox.Literal(doc["publisher"]))
        triple(node, _ont("year"), ox.Literal(str(doc["year"])))
        triple(node, _ont("describesRegion"), region_uri(doc["region"]))

    return store


# City is the most specific match, country the least; used to order results.
_LEVEL_RANK = {"city": 0, "canton": 1, "country": 2}


def documents_for_region(store: ox.Store, region_id: str) -> list[dict]:
    """Documents describing this region or any ancestor (city -> canton -> country).

    The ``geo:within*`` property path is the traversal: zero-or-more hops up the
    hierarchy from the queried region. ``matched_level`` records whether each
    document was attached to the city itself, its canton, or the country, so the
    context panel can show why it was pulled in.
    """
    query = f"""
    PREFIX geo: <{ONT}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?doc ?title ?url ?publisher ?level ?regionLabel WHERE {{
      <{region_uri(region_id).value}> geo:within* ?ancestor .
      ?ancestor geo:level ?level ; rdfs:label ?regionLabel .
      ?doc a geo:Document ;
           geo:describesRegion ?ancestor ;
           rdfs:label ?title ; geo:url ?url ; geo:publisher ?publisher .
    }}
    """
    out = []
    for s in store.query(query):
        out.append({
            "doc_id": s["doc"].value.rsplit("/", 1)[-1],
            "title": s["title"].value,
            "url": s["url"].value,
            "publisher": s["publisher"].value,
            "matched_region": s["regionLabel"].value,
            "matched_level": s["level"].value,
        })
    out.sort(key=lambda d: _LEVEL_RANK.get(d["matched_level"], 9))
    return out


def datasets_for_region(store: ox.Store, region_id: str) -> list[dict]:
    """Datasets whose extent covers this region (directly or via a parent)."""
    query = f"""
    PREFIX geo: <{ONT}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?ds ?label ?start ?end WHERE {{
      <{region_uri(region_id).value}> geo:within* ?ancestor .
      ?ds a geo:Dataset ; geo:coversRegion ?ancestor ; rdfs:label ?label .
      OPTIONAL {{ ?ds geo:temporalStart ?start ; geo:temporalEnd ?end }}
    }}
    """
    out = []
    for s in store.query(query):
        out.append({
            "label": s["label"].value,
            "temporal": f"{s['start'].value}-{s['end'].value}" if s["start"] else None,
        })
    return out
