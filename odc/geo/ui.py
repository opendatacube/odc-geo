import itertools
import math
from functools import lru_cache
from textwrap import dedent
from typing import Any, Optional, Tuple

from . import geom
from .crs import CRS
from .types import XY, OutlineMode, Shape2d, wh_, xy_

# pylint: disable=too-many-locals,import-outside-toplevel
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
    from .data import ocean_geom

    g = ocean_geom(bbox=clip_bbox)
    return g.svg_path(ndecimal)


def _compute_display_box(
    span: XY[float], sz: int, min_sz: int, tol: float = 1e-6
) -> Tuple[Shape2d, float]:
    """
    return: shape in pixels, ``s`` maps sizes from world to pix
    """
    span_x, span_y = ((v if math.isfinite(v) else 0) for v in span.xy)
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
        ).bbox
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


class PixelGridDisplay:
    """
    Common code for visualization of geoboxes.

    """

    def __init__(
        self,
        src: Any,
        pix2world: Any,
        gsd: float,
    ) -> None:
        self._src = src
        self._shape: Shape2d = src.shape
        self._crs: Optional[CRS] = src.crs
        self._pix2world = pix2world
        self._gsd = gsd

    def svg(
        self,
        scale_factor: float = 1.0,
        mode: OutlineMode = "auto",
        notch: float = 0.0,
        grid_stroke: str = "pink",
    ) -> str:
        """
        Produce SVG paths.

        :param mode: One of pixel, native, geo (default is geo)
        :return: SVG path
        """
        if mode == "auto":
            mode = "native" if self._crs is None else "geo"

        grids = self.grid_lines(mode=mode)
        outline = self.outline(mode, notch=notch)

        grid_svg = (
            '<path fill="none" opacity="0.8"'
            f' stroke-width="{0.8*scale_factor}"'
            f' stroke="{grid_stroke}"'
            f' d="{grids.svg_path()}" />'
        )

        return outline.svg(scale_factor) + grid_svg

    def grid_lines(self, step: int = 0, mode: OutlineMode = "native") -> geom.Geometry:
        """
        Construct pixel edge aligned grid lines.
        """

        nx, ny = self._shape.xy
        if nx > 0 and ny > 0:
            if step == 0:
                step = pick_grid_step(max(nx, ny))
            xx = [*range(0, nx, step), nx]
            yy = [*range(0, ny, step), ny]
            vertical = [list(itertools.product([x], yy)) for x in xx[1:-1]]
            horizontal = [list(itertools.product(xx, [y])) for y in yy[1:-1]]
            lines = geom.multiline(vertical + horizontal, self._crs)
        else:
            lines = geom.multiline([], self._crs)

        if mode == "pixel":
            return lines

        lines = lines.transform(self._pix2world)
        if mode == "native":
            return lines

        dx, dy = self._pix2world(step, 0)
        res = math.sqrt(dx * dx + dy * dy) / 5
        return lines.to_crs("epsg:4326", resolution=res).dropna()

    def outline(
        self, mode: OutlineMode = "native", notch: float = 0.1
    ) -> geom.Geometry:
        """
        Produce Line Geometry around perimeter.

        .. code-block:: txt

             +---+-------------+
             |   |             |
             +---+             |
             |                 |
             |                 |
             +-----------------+
        """

        assert notch < 1
        w, h = self._shape.wh
        if notch > 0:
            nn = min(notch * max(w, h), w, h)
            pix = geom.line(
                [
                    (0, nn),
                    (0, 0),
                    (nn, 0),
                    (w, 0),
                    (w, h),
                    (0, h),
                    (0, nn),
                    (nn, nn),
                    (nn, 0),
                ],
                self._crs,
            )
        else:
            pix = geom.multigeom(
                [
                    geom.line([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], self._crs),
                    geom.point(0, 0, self._crs),
                ]
            )
        if mode == "pixel":
            return geom.Geometry(pix.geom, crs=None)

        native = pix.transform(self._pix2world)
        if mode == "native":
            return native

        # about 100 pts per side
        bbox = native.boundingbox
        res = max(bbox.span_x, bbox.span_y) / 100

        return native.to_crs("EPSG:4326", resolution=res).dropna()

    def _display_bbox(self, pad_fraction: float = 0.1):
        bbox = self._src.geographic_extent.boundingbox
        pad_deg = max(bbox.span_x, bbox.span_y) * pad_fraction
        return bbox.buffered(pad_deg)

    def _render_svg(self, sz=360):
        if self._crs is None:
            bbox = self._src.extent.boundingbox
            margin = 0.1 * max(bbox.span_x, bbox.span_y)
            bbox = bbox.buffered(margin)
            return make_svg(
                self,
                bbox=bbox,
                sz=sz,
            )

        return svg_base_map(self, bbox=self._display_bbox(), sz=sz)

    def _repr_svg_(self):
        return self._render_svg()

    def _repr_html_(self):
        from .data import gbox_css

        W, H = self._shape.wh
        grid_step = pick_grid_step(max(W, H))
        svg_zoomed_txt = self._render_svg(sz=320)

        crs = self._crs
        if crs is None:
            authority = ("", "")
            wkt = "not set"
            units = ""
            svg_global_txt = ""
        else:
            authority = crs.authority
            wkt = crs.to_wkt(pretty=True).replace("\n", "<br/>").replace(" ", "&nbsp;")
            units = crs.units[0]
            svg_global_txt = svg_base_map(
                sz=200, target=self._src.geographic_extent.centroid.coords[0]
            )

        if authority == ("", ""):
            authority = ("CRS", "WKT")

        units = norm_units(units)
        pix_sz = self._gsd

        info = [
            ("Dimensions", f"{W:,d}x{H:,d}"),
            authority,
            ("Resolution", f"{pix_sz:g}{units}"),
            ("Cell", f"{grid_step:,d}px"),
        ]

        info_html = "\n".join(
            [
                (
                    f'<div class="row"><div class="column">{hdr}</div>'
                    f'<div class="column value">{val}</div></div>'
                )
                for hdr, val in info
            ]
        )
        src_class_name = type(self._src).__name__

        return dedent(
            f"""\
        <style>{gbox_css()}</style>
        <div class="gbox-info">
        <h4>{src_class_name}</h4>
        <div class="row">
            <div class="column">
                <div class="info-box">
                    {info_html}
                    <div>{svg_global_txt}</div>
                </div>
            </div>
            <div class="column svg-zoomed">{svg_zoomed_txt}</div>
        </div>
        <details>
            <summary>WKT</summary>
            <div class="wkt">{wkt}</div>
        </details>
        </div>"""
        )
