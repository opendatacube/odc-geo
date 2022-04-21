import math
from functools import lru_cache
from typing import Tuple

from . import geom
from .data import ocean_geom
from .types import XY, Shape2d, wh_, xy_

# pylint: disable=too-many-locals
_UNIT_REMAPS = {
    "metre": "m",
    "meter": "m",
    "degrees": "°",
    "degrees_north": "°",
    "degrees_south": "°",
    "degrees_east": "°",
    "degrees_west": "°",
}


def norm_units(unit: str) -> str:
    return _UNIT_REMAPS.get(unit, unit)


def pick_grid_step(N: int, at_least: int = 4, no_more: int = 11) -> int:
    if N <= 0:
        return 1

    factors = [1, 5, 10, 2, 4, 3, 6, 7, 8, 9, 1.5, 2.5]
    n = 10 ** (math.floor(math.log10(N)) - 1)
    if n < 1:
        return 1

    for f in factors:
        step = int(n * f)
        count = N // step
        if at_least <= count < no_more:
            return step

    return N // at_least


@lru_cache
def _ocean_svg_path(ndecimal=3, clip_bbox=None):
    g = ocean_geom()
    if clip_bbox is not None:
        g = g & geom.box(*clip_bbox, crs=g.crs)
    return g.svg_path(ndecimal)


def _compute_display_box(
    span: XY[float], sz: int, min_sz: int, tol: float = 1e-6
) -> Tuple[Shape2d, float]:
    """
    return: shape in pixels, ``s`` maps sizes from world to pix
    """
    span_x, span_y = span.xy
    if max(span_x, span_y) < tol:
        # both too small, make it square
        span_x = span_y = tol
        w, h = sz, sz
        s = w / span_x
    elif span_x > span_y:
        # wide case: w = sz, span_x == sz
        w, h = sz, max(int(sz * span_y / span_x), min_sz)
        s = w / span_x
    else:
        # tall case: h = sz, span_y == sz
        h, w = sz, max(int(sz * span_x / span_y), min_sz)
        s = h / span_y
    return wh_(w, h), s


def svg_base_map(
    *extras,
    fill="#CCCCCC",
    opacity=1,
    stroke_width=1,
    stroke="#555555",
    stroke_opacity=0.9,
    bbox=None,
    sz=360,
    map_svg_path=None,
    target=None,
):
    clip_bbox = None
    if bbox is not None:
        clip_bbox = geom.bbox_intersection(
            [geom.BoundingBox(*bbox).buffered(1), geom.BoundingBox(-180, -90, 180, 90)]
        )
    else:
        bbox = (-180, -90, 180, 90)

    if map_svg_path is None:
        map_svg_path = _ocean_svg_path(clip_bbox=clip_bbox)

    x0, y0, x1, y1 = bbox
    shape, s = _compute_display_box(xy_(x1 - x0, y1 - y0), sz, min(40, sz))
    w, h = shape.wh

    scale_factor = 1 / s
    stroke_width = stroke_width * scale_factor

    more_svg = ""
    for x in extras:
        if isinstance(x, str):
            more_svg += x
        else:
            more_svg += x.svg(scale_factor)

    if target is not None:
        target_x, target_y = target
        more_svg += (
            f'<path stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="{stroke_opacity*0.8}"'
            f' d="M{target_x},{target_y} V90zV-90zH180zH-180z"/>'
        )

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{w:d}" height="{h:d}"
     viewBox="0 0 {w:d} {h:d}"
     preserveAspectRatio="xMinYMin meet">
<g transform="matrix({s},0,0,{-s},{-x0*s},{y1*s})">
<path fill-rule="evenodd"
  fill="{fill}"
  opacity="{opacity}"
  stroke="{stroke}"
  stroke-width="{stroke_width}"
  stroke-opacity="{stroke_opacity}"
  d="{map_svg_path}"/>
{more_svg}</g></svg>"""


def make_svg(
    *extras,
    stroke_width=1,
    bbox=None,
    sz=360,
):
    if bbox is None:
        bbox = (0, 0, sz, sz)

    x0, y0, x1, y1 = bbox
    shape, s = _compute_display_box(xy_(x1 - x0, y1 - y0), sz, min(40, sz))
    w, h = shape.wh

    scale_factor = 1 / s
    stroke_width = stroke_width * scale_factor

    more_svg = ""
    for x in extras:
        if isinstance(x, str):
            more_svg += x
        else:
            more_svg += x.svg(scale_factor)

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{w:d}" height="{h:d}"
     viewBox="0 0 {w:d} {h:d}"
     preserveAspectRatio="xMinYMin meet">
<g transform="matrix({s},0,0,{-s},{-x0*s},{y1*s})">
{more_svg}
</g></svg>"""
