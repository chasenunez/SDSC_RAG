"""timeseries (Decadal?) land surface temperature (LST) for each of our big 5 swiss citiws from Landsat.

Source: Landsat Collection 2 Level-2 on Microsoft Planetary Computer. The
Level-2 product ships a Surface Temperature band (asset ``lwir11``) derived from
the thermal sensor. We are using one low-cloud summer scene per year, clipping it to the
city, converting to degrees Celsius, and then recording the mean. That yields an honest
decadal trend without a full cloud-masking/compositing pipeline.

Some of this code I wrote previously as part of a climate chnage project.

Two artifacts are cached:
- ``lst_annual.csv``  one row per city/year with the mean LST and the scene used.
- ``lst/<city>_<year>.tif``  the clipped LST raster (WGS84) for the latest year,
  used as the map overlay.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.warp import transform_bounds

from . import boundaries, config

_LST_DIR = config.CACHE_DIR / "lst"
_ANNUAL_CSV = config.CACHE_DIR / "lst_annual.csv"

# USGS Collection 2 Level-2 Surface Temperature scaling to Kelvin. I think this is pretty standarf, but I still prefer per-asset metadata when present.
_ST_SCALE = 0.00341802
_ST_OFFSET = 149.0
_KELVIN_TO_C = -273.15

# Landsat Collection 2 QA_PIXEL bit flags. Scene-level cloud cover is not enough:
# a scene under 30% cloud can still be cloudy directly over a city. We drop any
# pixel flagged fill, (dilated) cloud, cirrus, or cloud shadow before averaging.
_QA_FILL = 1 << 0
_QA_DILATED_CLOUD = 1 << 1
_QA_CIRRUS = 1 << 2
_QA_CLOUD = 1 << 3
_QA_CLOUD_SHADOW = 1 << 4
_QA_BAD = _QA_FILL | _QA_DILATED_CLOUD | _QA_CIRRUS | _QA_CLOUD | _QA_CLOUD_SHADOW

# A scene is only usable if enough of the city window survives masking.
_MIN_VALID_FRACTION = 0.4


def _open_catalog() -> pystac_client.Client:
    """Open the Planetary Computer STAC, signing asset URLs on read.

    Signing is anonymous (no API key); ``sign_inplace`` adds the short-lived SAS
    token each asset href needs.
    """
    return pystac_client.Client.open(
        config.PC_STAC_URL, modifier=planetary_computer.sign_inplace
    )


def _summer_scene_per_year(catalog, bbox) -> dict[int, object]:
    """Pick the least-cloudy summer scene for each year over the bbox.

    Restricted to Landsat 8/9 to avoid the Landsat 7 scan-line gaps that would
    bias a small urban average.
    """
    start, end = min(config.LST_YEARS), max(config.LST_YEARS)
    search = catalog.search(
        collections=[config.LANDSAT_COLLECTION],
        bbox=bbox,
        datetime=f"{start}-01-01/{end}-12-31",
        query={"eo:cloud_cover": {"lt": config.MAX_CLOUD_COVER},
               "platform": {"in": ["landsat-8", "landsat-9"]}},
    )
    by_year: dict[int, list] = defaultdict(list)
    for item in search.items():
        dt = item.datetime
        if dt.month in config.SUMMER_MONTHS:
            by_year[dt.year].append(item)

    best = {}
    for year, items in by_year.items():
        best[year] = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return best


def _surface_temp_asset(item):
    """Return the Surface Temperature asset, by key then by description."""
    if "lwir11" in item.assets:
        return item.assets["lwir11"]
    for asset in item.assets.values():
        title = (asset.title or "").lower()
        if "surface temperature" in title or "lwir" in title:
            return asset
    raise RuntimeError(f"no surface temperature asset on scene {item.id}")


def _open_clip(href, bbox, masked) -> xr.DataArray:
    """Open a COG asset and clip to the WGS84 bbox via a windowed read.

    The bbox is WGS84 but the scene is UTM, so we reproject the box (not the
    raster) and let GDAL read only the city window.
    """
    da = rioxarray.open_rasterio(href, masked=masked).squeeze()
    minx, miny, maxx, maxy = transform_bounds(config.WGS84, da.rio.crs, *bbox)
    return da.rio.clip_box(minx, miny, maxx, maxy)


def _clip_to_celsius(item, bbox) -> xr.DataArray:
    """Cloud-masked land surface temperature (Celsius) over the city window.

    Cloud/shadow pixels are set to NaN using the QA band so the city mean is not
    pulled down by cloud tops. Both bands come from the same scene grid, so they
    align after an identical clip.
    """
    asset = _surface_temp_asset(item)
    lst = _open_clip(asset.href, bbox, masked=True)

    # Prefer scale/offset from the asset metadata; fall back to USGS constants.
    scale, offset = _ST_SCALE, _ST_OFFSET
    bands = asset.extra_fields.get("raster:bands")
    if bands:
        scale = bands[0].get("scale", scale)
        offset = bands[0].get("offset", offset)
    celsius = lst * scale + offset + _KELVIN_TO_C

    qa = _open_clip(item.assets["qa_pixel"].href, bbox, masked=False)
    bad = (qa.astype("int32") & _QA_BAD) != 0
    return celsius.where(~bad.values)


def build_heat_series() -> pd.DataFrame:
    """Build (and cache) the decadal LST table for all cities.

    Returns a DataFrame with columns: city, year, mean_lst_c, scene_id,
    scene_date, cloud_cover. Also writes the latest-year overlay raster per city.
    """
    if _ANNUAL_CSV.exists():
        return pd.read_csv(_ANNUAL_CSV)

    _LST_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _open_catalog()
    rows = []
    for city in config.CITIES:
        bbox = boundaries.city_bbox(city.name)
        scenes = _summer_scene_per_year(catalog, bbox)
        latest_valid = None  # (year, masked LST DataArray) for the map overlay
        for year, item in sorted(scenes.items()):
            lst = _clip_to_celsius(item, bbox)
            valid_fraction = float(lst.notnull().mean())
            if valid_fraction < _MIN_VALID_FRACTION:
                continue  # too cloudy over the city to trust the average
            rows.append({
                "city": city.name,
                "year": year,
                "mean_lst_c": round(float(lst.mean()), 2),
                "scene_id": item.id,
                "scene_date": item.datetime.date().isoformat(),
                "cloud_cover": round(item.properties.get("eo:cloud_cover", float("nan")), 1),
                "valid_fraction": round(valid_fraction, 2),
            })
            latest_valid = (year, lst)

        # Save the most recent usable scene as the map overlay (WGS84).
        if latest_valid:
            year, lst = latest_valid
            lst.rio.reproject(config.WGS84).rio.to_raster(_LST_DIR / f"{city.name}_{year}.tif")

    df = pd.DataFrame(rows)
    df.to_csv(_ANNUAL_CSV, index=False)
    return df


def overlay_path(city: str) -> str | None:
    """Path to the latest cached LST overlay GeoTIFF for a city, if built."""
    hits = sorted(_LST_DIR.glob(f"{city}_*.tif"))
    return str(hits[-1]) if hits else None


def heat_summary(df: pd.DataFrame, city: str) -> dict:
    """Compact decadal summary for one city, fed to the LLM as concrete numbers.

    The trend is a least-squares slope over all years, not an endpoint
    difference: each value is one clear-sky daytime scene, so it carries the
    weather of that particular day. We also report the year-to-year spread so
    the slope is read with its uncertainty rather than as a clean signal.
    """
    import numpy as np

    sub = df[df["city"] == city].sort_values("year")
    if sub.empty:
        return {}
    years = sub["year"].to_numpy(dtype=float)
    temps = sub["mean_lst_c"].to_numpy(dtype=float)
    slope = float(np.polyfit(years, temps, 1)[0]) if len(sub) > 1 else 0.0
    return {
        "city": city,
        "first_year": int(years[0]),
        "last_year": int(years[-1]),
        "n_years": int(len(sub)),
        "trend_c_per_decade": round(slope * 10, 2),
        "mean_c": round(float(temps.mean()), 2),
        "year_to_year_std_c": round(float(temps.std(ddof=0)), 2),
        "warmest": {"year": int(years[temps.argmax()]), "mean_c": round(float(temps.max()), 2)},
        "coolest": {"year": int(years[temps.argmin()]), "mean_c": round(float(temps.min()), 2)},
    }
