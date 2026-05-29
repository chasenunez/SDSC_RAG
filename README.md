# Swiss Urban Heat: a Geo-RAG + Knowledge Graph slice

You pick a Swiss city and ask "how has urban heat changed here over the last decade?" The tool pulls real satellite measurements of how hot the ground gets, finds the official reports that describe that city and its region, and writes a short answer that cites those reports. It shows all of this on one screen: a map with a heat overlay, a panel listing the documents it used, and the written answer.

### The structure of this repo

The goal (as stipulated by the instructions) is to build "a working vertical slice" that answers one question end to end. To that end, this is a proof of concept for how three components can be combined in a scalable way:

1) *Geospatial data*: ideally a publicly accessible satellite remote sensing product with a sufficiently long time series. The example here is urban heat — temperature is one of the more robust and well-documented data products, so it's a sensible starting point. Precipitation and snow cover would be the next candidates for Switzerland.

2) *A knowledge graph (KG)* that links different data entities with the semantic relationships a large language model (LLM) can lean on. Geospatial layers usually come with rich metadata, especially those derived from the same sensors/satellites, so there's plenty for the graph to bind. The graph is the gatekeeper: a question resolves to a region, the graph selects the documents attached to that region or its parents, and vector search ranks chunks only inside those documents.

3) *A RAG (retrieval-augmented generation) module* that grounds the LLM in external data, letting it return specific, citable, up-to-date material instead of relying on static training data.

The shape, in one line: the knowledge graph decides *which* documents are relevant by where they apply, and only then does the search look *inside* those documents. The graph leads, the search follows.

A note on background: my experience with KGs, RAG, and LLMs comes mostly from a boot.dev course, while I've spent more time retrieving, combining, and analyzing geospatial remote sensing data for global inference. I leaned into the geospatial layer where I had the most leverage and kept the graph and RAG layers a tight, scoped slice rather than overreaching. With that out of the way, let's talk turkey.

### Scoping and specifics

The full scope document lives in [`SCOPE.md`](SCOPE.md). The short version is below.

#### What I built fully

- **Real surface temperature**, from Landsat Collection 2 Level-2 on Microsoft Planetary Computer, with the USGS scale and offset applied to get Celsius and pixel-level cloud and shadow masking from the QA band. Projections were going to be a pain point so I kept them explicit: each scene is read in its native UTM, the city bbox is reprojected from WGS84 into the scene's CRS for a windowed read, and the overlay is reprojected back to WGS84 for display on the map.
- **Real Swiss boundaries** from swissBOUNDARIES3D (swisstopo, native EPSG:2056), reprojected to WGS84 with explicit bbox handling. Cities are selected by federal BFS number and the resulting polygon name is cross-checked against the expected city to catch any silent mismatch.
- **A knowledge graph in Oxigraph** (new to me but small enough to model cleanly), with Region, Dataset, and Document nodes and within, describesRegion, coversRegion, and delineatedBy edges. There's a toggle in the dashboard that turns the graph filter off, so you can watch other cities' plans leak into the result.
- **A graph-driven RAG**: document chunks are embedded into Qdrant, a question resolves to a region, the graph picks candidate documents via a `within*` traversal, and vector search runs only inside that set. The answer is generated through OpenRouter with structured output via pydantic-ai, and a grounding guard drops any citation that points at a source the model did not actually see.

#### What I simplified for time and computing budget

- I stayed inside Switzerland and worked at city scale: Zürich, Genève, Basel, Bern, Lausanne. A small region set lets each city carry real, distinct documents rather than pretending a tiny corpus covers a whole country. The graph hierarchy is modelled as city within canton within country, so the same retrieval code works at coarser scales; only the data loading would change.
- One clear-sky summer Landsat scene per year, not monthly composites. Each value carries that day's weather, so the headline decadal trend is reported alongside its year-to-year spread and labelled indicative rather than definitive. This is the single biggest honest weakness, and the first thing I'd replace with a monthly median.
- The corpus is seven (real) climate-policy documents stored as source-grounded summaries with verified titles, publishers, years, and URLs, rather than full PDF ingestion. Keeps the repo small and license-clean while still giving retrieval real, citable material per region.
- A deterministic, clearly-labelled templated fallback runs when no LLM key is set (or when the call fails), so the narrative panel is never empty for a reviewer without an OpenRouter account.

#### What I deliberately left out

Canton and national scale, sensor time series, multi-narrative dashboards, a real GIS stack, and user authentication. The graph hierarchy already supports rolling up to canton and country; that extension would live in the data-loading step, not the retrieval. The rest adds breadth, not new design, so I described it here instead of building it shallow.

### Putting the bits together

```
========================  STEP 1: SETUP THE PIECES  ========================
Goal: gather the data and prepare it so questions can be answered quickly.

  swisstopo                       boundaries.py           data/cache/
  swissBOUNDARIES3D  ───────────► download, convert  ───► cities.gpkg
  (official city shapes)          map coordinates          (city outlines +
                                                            bounding boxes)

  Landsat satellite               heat.py                 data/cache/
  (via Planetary       ─────────► pick clear summer  ───► lst_annual.csv +
   Computer)                      scenes, drop clouds,     overlay images
  (ground temperature)            average per city/year

  manifest.json                   documents.py            index.py
  (7 real reports,    ──────────► split into chunks  ───► Qdrant store
   as text)                                                (text turned into
                                                            searchable numbers)

============================  STEP 2: ASK A QUESTION  ======================
Goal: answer "how has urban heat changed in <a big Swiss city>?" using the prepared data.

   You: pick a city, type a question
                 │
                 ▼
   ┌───────────────────────────────────────────────┐
   │ graph.py   =  the KNOWLEDGE GRAPH (Oxigraph)    │
   │ Walks a map of relationships:                   │
   │     city  →  canton  →  country                 │
   │ Returns: which documents describe this place.   │
   └───────────────────────────────────────────────┘
                 │  a short list of "in-scope" documents
                 ▼
   ┌───────────────────────────────────────────────┐
   │ retrieval.py  =  the SEARCH STEP                │
   │ Searches the Qdrant numbers, but only inside    │
   │ the documents the graph chose. Returns the most │
   │ relevant passages plus the heat statistics.     │
   └───────────────────────────────────────────────┘
                 │  passages + statistics
                 ▼
   ┌───────────────────────────────────────────────┐
   │ rag.py     =  the ANSWER STEP                   │
   │ A language model (via OpenRouter) writes a short│
   │ answer that cites those sources. With no API key│
   │ a plain templated summary is shown instead.     │
   └───────────────────────────────────────────────┘
                 │  cited answer
                 ▼
   ┌───────────────────────────────────────────────┐
   │ dashboard.py  =  the SCREEN (Streamlit)         │
   │  ┌──────────┬───────────────┬────────────────┐  │
   │  │  MAP +   │  GRAPH        │  CITED          │  │
   │  │  heat    │  context      │  ANSWER         │  │
   │  │  overlay │  (which docs) │                 │  │
   │  └──────────┴───────────────┴────────────────┘  │
   └───────────────────────────────────────────────┘
```

## Layout

```
src/geokg/
  config.py      cities, paths, endpoints, model defaults
  boundaries.py  swissBOUNDARIES3D download, reproject, city polygons
  heat.py        Landsat LST series, cloud masking, decadal summary
  documents.py   corpus loader + chunker
  graph.py       Oxigraph RDF model + SPARQL region traversal
  index.py       embed chunks into local Qdrant
  retrieval.py   graph-first retrieval (graph filters, vectors rank)
  rag.py         pydantic-ai agent over OpenRouter, grounded + cited
  viz.py         LST raster to map-overlay image
scripts/setup_data.py   one-shot data preparation
app/dashboard.py        Streamlit map + graph panel + answer
```

## First-time setup guide

### 1. Install `uv` if you don't already have it

- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Other options: <https://docs.astral.sh/uv/getting-started/installation/>

### 2. Clone and install dependencies

```
git clone https://github.com/chasenunez/SDSC_RAG.git
cd SDSC_RAG
uv sync
```

`uv sync` reads `.python-version` and `uv.lock`, downloads Python 3.12 if needed, and builds the environment. First run takes about a minute.

### 3. Set the LLM API key (the easy way)

The app reads exactly two environment variables: `OPENROUTER_API_KEY` (required for the LLM answer) and `OPENROUTER_MODEL` (an optional override; the default is set in `src/geokg/config.py`). Pick one of the three options below.

#### Option A: a `.env` file (recommended)

Get a key first: sign up at <https://openrouter.ai>, go to Keys, create one. It looks like `sk-or-v1-...`.

Then:

```
cp .env.example .env
```

Open `.env` in any editor and replace the blank line:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

The `.env` file is gitignored, so the key is never committed.

#### Option B: export in your shell (one-off, no file)

```
export OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

Works for the current terminal only. The code uses `os.environ.setdefault`, which means a value already in the environment wins over the `.env` file. Useful for temporary overrides.

#### Option C: no key at all

The app runs fully without a key. The answer panel shows a clearly-labelled templated summary built from the retrieved sources and the heat statistics, with the same citations and caveat. You only lose the LLM-written prose. The map, graph panel, retrieved chunks, and citations all still render.

### 4. Prepare the data (one time, about two minutes)

```
make setup
```

Or, equivalently:

```
uv run python scripts/setup_data.py
```

This downloads the Swiss city outlines (~37 MB), fetches one cloud-masked summer Landsat scene per city per year from 2014 to 2024, and builds the local Qdrant search index. Everything caches under `data/cache/` so it only happens once.

### 5. Run the dashboard

```
make run
```

Or:

```
uv run streamlit run app/dashboard.py
```

A browser tab opens at <http://localhost:8501>. If that port is busy, add `--server.port 8520` (or any free port you have).

## How to tell which mode is active

The dashboard tells you explicitly:

- **With a key set**: pressing the sidebar "Run" button shows a "Generating grounded answer…" spinner, then a prose answer.
- **Without a key**: a caption appears above the answer reading "Generated without an LLM (no `OPENROUTER_API_KEY` set): a templated summary of the retrieved evidence. Set a key for a written narrative."

If you set the key and still see the fallback caption, check, in this order:

1. The file is named exactly `.env` (not `.env.txt` or `.env.example`) and sits in the repo root next to `pyproject.toml`. The loader looks for `<repo-root>/.env`.
2. The value has no surrounding quotes.
3. You need to trigger a Streamlit rerun for the new key to be picked up. The simplest way: change the city in the sidebar, or stop the app with Ctrl+C and run `make run` again.
4. Your shell does not already have a stale empty `OPENROUTER_API_KEY` exported. Check with `echo $OPENROUTER_API_KEY`. Shell exports win over `.env`. If you see something unexpected, run `unset OPENROUTER_API_KEY` and restart the app.

## What to do once it's open

- Pick a city in the sidebar. The map, graph context panel, and answer all update together.
- Toggle **"Use graph to filter retrieval"** off and on. With it off, the "Retrieved chunks" expander will show passages from other cities' plans creeping in; with it on, retrieval stays inside the in-scope set. That is the graph earning its place in the pipeline.
- Open the **"Retrieved chunks"** expander to see the exact passages the answer is grounded in, with relevance scores.
- The data caveat under the answer is honest about the noise: the trend per decade is similar in magnitude to the year-to-year spread, so treat it as indicative.

## Next steps

The clearest improvement for scaling is to compute the heat trend on a server-side service like Google Earth Engine instead of pulling single scenes here. GEE runs the analysis over its Landsat catalog, so cloud masking, monthly compositing, and a per-pixel trend (`ee.Reducer.linearFit` over `system:time_start`) would replace the weather-noisy single-scene loop in `heat.py`, and `reduceRegions` returns per-region statistics for every municipality in one call, which is what makes canton or national scale practical rather than a per-area download.

The tradeoff is access. GEE needs a registered Google Cloud project (free for noncommercial and academic use, but with an eligibility questionnaire and quota tiers), so depending on it at runtime would break the current clone-and-run setup. The way to get both is to use GEE as an offline precompute step: run it once to produce the per-region, per-year statistics and a composite raster, commit those as cached artifacts, and keep the dashboard dependency-free. That is the same precompute-and-cache shape `scripts/setup_data.py` already uses, just with a more capable backend. At national scale the cached statistics would move from CSV to DuckDB or Parquet so the lookups stay fast.
