"""Shared configuration: cities, paths, data endpoints, model defaults.

Everything that another module might want to tweak lives here so the rest of
the code stays free of magic strings and hard-coded paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project layout. All generated artifacts go under data/cache so a clean
# checkout only needs the committed manifest plus a setup run.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
CACHE_DIR = DATA_DIR / "cache"

# swisstopo coordinate system (LV95). Web maps and STAC bboxes use WGS84.
SWISS_CRS = "EPSG:2056"
WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class City:
    """One study area. `bfs` is the federal municipality number, the stable
    key used to select the polygon from swissBOUNDARIES3D; `name` is only for
    display and is cross-checked against the data to catch a wrong number."""

    name: str
    bfs: int
    canton: str  # canton abbreviation, e.g. "ZH"


# Five of the largest Swiss cities.
# BFS numbers are verified against the boundary data at load time.
CITIES: list[City] = [
    City("Zürich", 261, "ZH"),
    City("Genève", 6621, "GE"),
    City("Basel", 2701, "BS"),
    City("Bern", 351, "BE"),
    City("Lausanne", 5586, "VD"),
]


def city_by_name(name: str) -> City:
    for c in CITIES:
        if c.name == name:
            return c
    raise KeyError(f"unknown city: {name!r}")


# Landsat surface temperature: summer scenes across this span give a decadal
# heat trend. July/August only, to compare like with like across years.
LST_YEARS = range(2014, 2025)
SUMMER_MONTHS = (6, 7, 8)
MAX_CLOUD_COVER = 30  # percent; skip cloudier scenes over the city

# STAC endpoints.
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
SWISSTOPO_STAC_URL = "https://data.geo.admin.ch/api/stac/v1"
SWISSBOUNDARIES_COLLECTION = "ch.swisstopo.swissboundaries3d"

# Local vector store and embedding model. fastembed runs on CPU (ONNX) and
# downloads this small model once on first use; index and query must agree on it.
QDRANT_PATH = CACHE_DIR / "qdrant"
QDRANT_COLLECTION = "doc_chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# OpenRouter (OpenAI-compatible). Key and model are read from the environment at
# call time, after load_env() pulls in a local .env file.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


def load_env() -> None:
    """Load KEY=VALUE lines from a local .env into the environment (if present).

    A tiny reader avoids a dependency just to find one API key; existing
    environment variables win over the file.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def model_name() -> str:
    """The OpenRouter model id to use (override with OPENROUTER_MODEL)."""
    return os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
