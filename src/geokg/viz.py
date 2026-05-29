"""oh fun! visualization time. Too bad I probably dont have time to give this as much attention as it deserves
I'll probably come back and spend any additional time tuning up this code. 
Here we will turn a land surface temperature GeoTIFF into a map overlay image. 
Probably a small blue-to-red ramp built with numpy, with intention of avoiding matplotlib just to colour one raster. 
Returns an RGBA array plus the lat/lon bounds folium needs for an ImageOverlay.
"""

from __future__ import annotations

import numpy as np
import rasterio

# Three-stop ramp (cool blue -> pale -> hot red), values are 0-255 per channel. maybe this coudl be viridis, but then people may assume matlibplot is involved. 
_RAMP = {
    "r": (49, 255, 165),
    "g": (54, 255, 0),
    "b": (149, 191, 38),
}


def _colorize(norm: np.ndarray) -> np.ndarray:
    """Map values normalised to 0-1 onto the ramp, returning an RGBA uint8 array."""
    xp = [0.0, 0.5, 1.0]
    safe = np.nan_to_num(norm, nan=0.0)
    r = np.interp(safe, xp, _RAMP["r"])
    g = np.interp(safe, xp, _RAMP["g"])
    b = np.interp(safe, xp, _RAMP["b"])
    a = np.full(norm.shape, 200.0)
    return np.dstack([r, g, b, a]).astype("uint8")


def lst_to_rgba(path: str) -> tuple[np.ndarray, list, float, float]:
    """Read a Celsius LST raster (WGS84) and return (rgba, bounds, vmin, vmax).

    The colour range is clipped to the 2nd-98th percentile so a few extreme
    pixels do not wash out the contrast. ``bounds`` is [[south, west],
    [north, east]] for folium; no-data pixels are made transparent.
    """
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32")
        b = src.bounds

    data = np.ma.filled(arr, np.nan)
    finite = np.isfinite(data)
    vmin = float(np.nanpercentile(data, 2))
    vmax = float(np.nanpercentile(data, 98))
    norm = np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)

    rgba = _colorize(norm)
    rgba[~finite] = (0, 0, 0, 0)  # transparent where there is no data
    bounds = [[b.bottom, b.left], [b.top, b.right]]
    return rgba, bounds, round(vmin, 1), round(vmax, 1)
