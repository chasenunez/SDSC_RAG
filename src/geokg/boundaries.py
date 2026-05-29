"""City boundaries from swissBOUNDARIES3D (swisstopo).

Thos script is mostly tasked with downloading the latest municipal-boundary GeoPackage, pulling out the five study
cities by their federal (BFS) number, and reprojecting them to WGS84 for web mapping. 
TI want to take this moment to complain about projections and geospatial data. thank you. 
The selected subset is a tiny GeoPackage so the ~37 MB national file
is read only once.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import geopandas as gpd
import shapely

from . import config

# Cached outputs.
_NATIONAL_GPKG = config.CACHE_DIR / "swissboundaries3d.gpkg"
_CITIES_GPKG = config.CACHE_DIR / "cities.gpkg"


def _latest_gpkg_url() -> str:
    """I Find the newest swissBOUNDARIES3D GeoPackage via the swisstopo STAC API.

    I have kept the URL discovery (rather than hard-coding an edition) so that this is usable into the future. 
    """
    import json

    base = config.SWISSTOPO_STAC_URL
    url = f"{base}/collections/{config.SWISSBOUNDARIES_COLLECTION}/items?limit=100"
    items = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        items += page.get("features", [])
        nxt = [l["href"] for l in page.get("links", []) if l.get("rel") == "next"]
        url = nxt[0] if nxt else None

    if not items:
        raise RuntimeError("no swissBOUNDARIES3D items returned by swisstopo STAC")
    latest = max(items, key=lambda f: f.get("id", ""))
    for key, asset in latest["assets"].items():
        # The LV95 (EPSG:2056) GeoPackage is the one we want.
        if key.endswith(".gpkg.zip") and "_2056_" in key:
            return asset["href"]
    raise RuntimeError(f"no LV95 GeoPackage asset on item {latest.get('id')}")


def _download_national_gpkg() -> None:
    """I find and unzip the national GeoPackage into the cache. I am idempotent but not omnipotent."""
    if _NATIONAL_GPKG.exists():
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = _latest_gpkg_url()
    with urllib.request.urlopen(url, timeout=300) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".gpkg")]
        if not names:
            raise RuntimeError("no .gpkg inside the swissBOUNDARIES3D archive")
        with zf.open(names[0]) as src, open(_NATIONAL_GPKG, "wb") as dst:
            dst.write(src.read())


def _municipality_layer(gpkg_path) -> str:
    """I Return the layer holding municipalities. The challenge is that swissBOUNDARIES3D packages several admin levels in one big file.
    So I also pick the layer that carries a BFS number column, which is unique to municipalities and what we need to pull out the swiss cities.
    """
    import pyogrio

    for name in pyogrio.list_layers(gpkg_path)[:, 0]:
        cols = pyogrio.read_info(gpkg_path, layer=name)["fields"]
        if any("bfs" in c.lower() for c in cols):
            return name
    raise RuntimeError("no municipality layer with a BFS number found")


def build_city_boundaries() -> gpd.GeoDataFrame:
    """I build (and cache) the WGS84 city polygons for the big 5 sewiss cities.

    Returns a GeoDataFrame indexed by city name with columns: bfs, canton,
    geometry (2D, EPSG:4326). Reading from the cache on later runs is instant.
    """
    if _CITIES_GPKG.exists():
        return gpd.read_file(_CITIES_GPKG).set_index("name")

    _download_national_gpkg()
    layer = _municipality_layer(_NATIONAL_GPKG)
    gdf = gpd.read_file(_NATIONAL_GPKG, layer=layer)

    # Normalise the BFS and name column names across editions.
    bfs_col = next(c for c in gdf.columns if "bfs" in c.lower())
    name_col = next(c for c in gdf.columns if c.lower() in ("name", "gemname"))

    rows = []
    for city in config.CITIES:
        match = gdf[gdf[bfs_col] == city.bfs]
        if match.empty:
            raise RuntimeError(f"BFS {city.bfs} ({city.name}) not in boundary data")
        # Guard against a stale/wrong BFS number: the data name must match.
        found = str(match.iloc[0][name_col])
        if city.name not in found:
            raise RuntimeError(
                f"BFS {city.bfs} maps to {found!r}, expected {city.name!r}"
            )
        rows.append({"name": city.name, "bfs": city.bfs, "canton": city.canton,
                     "geometry": match.union_all()})

    cities = gpd.GeoDataFrame(rows, crs=gdf.crs).to_crs(config.WGS84)
    # Drop the Z dimension; folium and bbox math only need lon/lat.
    cities["geometry"] = shapely.force_2d(cities["geometry"].values)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cities.to_file(_CITIES_GPKG, driver="GPKG")
    return cities.set_index("name")


def city_bbox(name: str) -> tuple[float, float, float, float]:
    """WGS84 bounding box (minx, miny, maxx, maxy) for one city."""
    geom = build_city_boundaries().loc[name, "geometry"]
    return tuple(round(v, 6) for v in geom.bounds)
