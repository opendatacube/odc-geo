# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Write Cloud Optimized GeoTIFFs from xarrays.
"""
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import rasterio
import xarray as xr
from rasterio.shutil import copy as rio_copy  # pylint: disable=no-name-in-module

from ..geobox import GeoBox
from ..types import MaybeNodata, SomeShape, shape_, wh_
from ..warp import resampling_s2rio
from ._shared import adjust_blocksize

# pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-statements,too-many-instance-attributes


def _without(cfg: Dict[str, Any], *skip: str) -> Dict[str, Any]:
    skip = set(skip)
    return {k: v for k, v in cfg.items() if k not in skip}


def check_write_path(fname: Union[Path, str], overwrite: bool) -> Path:
    """
    Check path before overwriting.

    :param fname: string or Path object
    :param overwrite: Whether to remove file when it exists

    exists   overwrite   Action
    ----------------------------------------------
    T            T       delete file, return Path
    T            F       raise IOError
    F            T       return Path
    F            F       return Path
    """
    if not isinstance(fname, Path):
        fname = Path(fname)

    if fname.exists():
        if overwrite:
            fname.unlink()
        else:
            raise IOError("File exists")
    return fname


def _default_cog_opts(
    *, blocksize: int = 512, shape: SomeShape = (0, 0), is_float: bool = False, **other
) -> Dict[str, Any]:
    nx, ny = shape_(shape).xy
    return {
        "tiled": True,
        "blockxsize": adjust_blocksize(blocksize, nx),
        "blockysize": adjust_blocksize(blocksize, ny),
        "zlevel": 6,
        "predictor": 3 if is_float else 2,
        "compress": "DEFLATE",
        **other,
    }


def _norm_compression_opts(
    compression: Union[bool, str, Dict[str, Any]],
    default_compress: str = "deflate",
    default_zlevel: int = 2,
) -> Dict[str, Any]:
    if isinstance(compression, bool):
        if compression:
            return {"compress": default_compress, "zlevel": default_zlevel}
        return {"compress": None}
    if isinstance(compression, str):
        compression = {"compress": compression}
    return compression


def _write_cog(
    pix: np.ndarray,
    geobox: GeoBox,
    fname: Union[Path, str],
    nodata: MaybeNodata = None,
    overwrite: bool = False,
    blocksize: Optional[int] = None,
    overview_resampling: Optional[str] = None,
    overview_levels: Optional[List[int]] = None,
    ovr_blocksize: Optional[int] = None,
    use_windowed_writes: bool = False,
    intermediate_compression: Union[bool, str, Dict[str, Any]] = False,
    **extra_rio_opts,
) -> Union[Path, bytes]:
    if blocksize is None:
        blocksize = 512
    if ovr_blocksize is None:
        ovr_blocksize = blocksize
    if overview_resampling is None:
        overview_resampling = "nearest"

    intermediate_compression = _norm_compression_opts(intermediate_compression)

    if pix.ndim == 2:
        h, w = pix.shape
        nbands = 1
        band = 1  # type: Any
    elif pix.ndim == 3:
        if pix.shape[:2] == geobox.shape:
            pix = pix.transpose([2, 0, 1])
        elif pix.shape[-2:] != geobox.shape:
            raise ValueError("GeoBox shape does not match image shape")

        nbands, h, w = pix.shape  # type: ignore
        band = tuple(i for i in range(1, nbands + 1))
    else:
        raise ValueError("Need 2d or 3d ndarray on input")

    assert geobox.shape == (h, w)

    if overview_levels is None:
        if min(w, h) < 512:
            overview_levels = []
        else:
            overview_levels = [2**i for i in range(1, 6)]

    if fname != ":mem:":
        path = check_write_path(
            fname, overwrite
        )  # aborts if overwrite=False and file exists already

    resampling = resampling_s2rio(overview_resampling)

    if (blocksize % 16) != 0:
        warnings.warn("Block size must be a multiple of 16, will be adjusted")

    rio_opts = {
        "width": w,
        "height": h,
        "count": nbands,
        "dtype": pix.dtype.name,
        "crs": str(geobox.crs),
        "transform": geobox.transform,
        **_default_cog_opts(
            blocksize=blocksize, shape=wh_(w, h), is_float=pix.dtype.kind == "f"
        ),
    }
    if nodata is not None:
        rio_opts.update(nodata=nodata)

    rio_opts.update(extra_rio_opts)

    def _write(pix, band, dst):
        if not use_windowed_writes:
            dst.write(pix, band)
            return

        for _, win in dst.block_windows():
            if pix.ndim == 2:
                block = pix[win.toslices()]
            else:
                block = pix[(slice(None),) + win.toslices()]

            dst.write(block, indexes=band, window=win)

    # Deal efficiently with "no overviews needed case"
    if len(overview_levels) == 0:
        if fname == ":mem:":
            with rasterio.MemoryFile() as mem:
                with mem.open(driver="GTiff", **rio_opts) as dst:
                    _write(pix, band, dst)
                return bytes(mem.getbuffer())
        else:
            with rasterio.open(path, mode="w", driver="GTiff", **rio_opts) as dst:
                _write(pix, band, dst)
            return path

    # copy re-compresses anyway so skip compression for temp image
    tmp_opts = _without(rio_opts, "compress", "predictor", "zlevel")
    tmp_opts.update(intermediate_compression)

    with rasterio.Env(GDAL_TIFF_OVR_BLOCKSIZE=ovr_blocksize):
        with rasterio.MemoryFile() as mem:
            with mem.open(driver="GTiff", **tmp_opts) as tmp:
                _write(pix, band, tmp)
                tmp.build_overviews(overview_levels, resampling)

                if fname == ":mem:":
                    with rasterio.MemoryFile() as mem2:
                        rio_copy(
                            tmp,
                            mem2.name,
                            driver="GTiff",
                            copy_src_overviews=True,
                            **_without(
                                rio_opts,
                                "width",
                                "height",
                                "count",
                                "dtype",
                                "crs",
                                "transform",
                                "nodata",
                            ),
                        )
                        return bytes(mem2.getbuffer())

                rio_copy(tmp, path, driver="GTiff", copy_src_overviews=True, **rio_opts)

    return path


def write_cog(
    geo_im: xr.DataArray,
    fname: Union[str, Path],
    *,
    overwrite: bool = False,
    blocksize: Optional[int] = None,
    ovr_blocksize: Optional[int] = None,
    overviews: Optional[Iterable[xr.DataArray]] = None,
    overview_resampling: Optional[str] = None,
    overview_levels: Optional[List[int]] = None,
    use_windowed_writes: bool = False,
    intermediate_compression: Union[bool, str, Dict[str, Any]] = False,
    **extra_rio_opts,
) -> Union[Path, bytes]:
    """
    Save ``xarray.DataArray`` to a file in Cloud Optimized GeoTiff format.

    :param geo_im: ``xarray.DataArray`` with crs
    :param fname: Output path or ``":mem:"`` in which case compress to RAM and return bytes
    :param overwrite: True -- replace existing file, False -- abort with IOError exception
    :param blocksize: Size of internal tiff tiles (512x512 pixels)
    :param ovr_blocksize: Size of internal tiles in overview images (defaults to blocksize)
    :param overviews: Write pre-computed overviews if supplied
    :param overview_resampling: Use this resampling when computing overviews
    :param overview_levels: List of shrink factors to compute overiews for: [2,4,8,16,32],
                            to disable overviews supply empty list ``[]``
    :param nodata: Set ``nodata`` flag to this value if supplied, by default ``nodata`` is
                   read from the attributes of the input array (``geo_im.attrs['nodata']``).
    :param use_windowed_writes: Write image block by block (might need this for large images)
    :param intermediate_compression: Configure compression settings for first pass write, default is no compression
    :param extra_rio_opts: Any other option is passed to ``rasterio.open``

    :returns: Path to which output was written
    :returns: Bytes if ``fname=":mem:"``

    .. note ::

       **memory requirements**

       This function generates a temporary in memory tiff file without
       compression to speed things up. It then adds overviews to this file and
       only then copies it to the final destination with requested compression
       settings. This is necessary to produce a compliant COG, since the COG standard
       demands overviews to be placed before native resolution data and double
       pass is the only way to achieve this currently.

       This means that this function will use about 1.5 to 2 times memory taken by ``geo_im``.
    """
    if overviews is not None:
        layers = [geo_im, *overviews]
        result = write_cog_layers(
            layers,
            fname,
            overwrite=overwrite,
            blocksize=blocksize,
            ovr_blocksize=ovr_blocksize,
            use_windowed_writes=use_windowed_writes,
            intermediate_compression=intermediate_compression,
            **extra_rio_opts,
        )
        assert result is not None
        return result

    pix = geo_im.data
    geobox = geo_im.odc.geobox
    nodata = extra_rio_opts.pop("nodata", None)
    if nodata is None:
        nodata = geo_im.attrs.get("nodata", None)

    if geobox is None:
        raise ValueError("Need geo-registered array on input")

    return _write_cog(
        pix,
        geobox,
        fname,
        nodata=nodata,
        overwrite=overwrite,
        blocksize=blocksize,
        ovr_blocksize=ovr_blocksize,
        overview_resampling=overview_resampling,
        overview_levels=overview_levels,
        use_windowed_writes=use_windowed_writes,
        intermediate_compression=intermediate_compression,
        **extra_rio_opts,
    )


def to_cog(
    geo_im: xr.DataArray,
    blocksize: Optional[int] = None,
    ovr_blocksize: Optional[int] = None,
    overviews: Optional[Iterable[xr.DataArray]] = None,
    overview_resampling: Optional[str] = None,
    overview_levels: Optional[List[int]] = None,
    use_windowed_writes: bool = False,
    intermediate_compression: Union[bool, str, Dict[str, Any]] = False,
    **extra_rio_opts,
) -> bytes:
    """
    Compress ``xarray.DataArray`` into Cloud Optimized GeoTiff bytes in memory.

    This function doesn't write to disk, it compresses in RAM, which is useful
    for saving data to S3 or other cloud object stores.

    :param geo_im: ``xarray.DataArray`` with crs
    :param blocksize: Size of internal tiff tiles (512x512 pixels)
    :param ovr_blocksize: Size of internal tiles in overview images (defaults to blocksize)
    :param overviews: Write pre-computed overviews if supplied
    :param overview_resampling: Use this resampling when computing overviews
    :param overview_levels: List of shrink factors to compute overiews for: [2,4,8,16,32]
    :param nodata: Set ``nodata`` flag to this value if supplied, by default ``nodata`` is
                   read from the attributes of the input array (``geo_im.attrs['nodata']``).
    :param use_windowed_writes: Write image block by block (might need this for large images)
    :param intermediate_compression: Configure compression settings for first pass write, default is no compression
    :param extra_rio_opts: Any other option is passed to ``rasterio.open``

    :returns: In-memory GeoTiff file as bytes

    """
    bb = write_cog(
        geo_im,
        ":mem:",
        blocksize=blocksize,
        ovr_blocksize=ovr_blocksize,
        overviews=overviews,
        overview_resampling=overview_resampling,
        overview_levels=overview_levels,
        use_windowed_writes=use_windowed_writes,
        intermediate_compression=intermediate_compression,
        **extra_rio_opts,
    )

    assert isinstance(
        bb, (bytes,)
    )  # for mypy sake for :mem: output it bytes or delayed bytes
    return bb


@contextmanager
def _memfiles_ovr(nlevels) -> Generator[Tuple[rasterio.MemoryFile, ...], None, None]:
    tt = str(uuid4())
    dirname, fname = tt.split("-", 1)
    fname = fname + ".tif"
    mems = []
    try:
        for i in range(nlevels):
            mems.append(
                rasterio.MemoryFile(dirname=dirname, filename=fname + ".ovr" * i)
            )
        yield tuple(mems)
    finally:
        for mem in mems[::-1]:
            if not mem.closed:
                mem.close()
        del mems


def write_cog_layers(
    layers: Iterable[xr.DataArray],
    dst: Union[str, Path] = ":mem:",
    overwrite: bool = False,
    blocksize: Optional[int] = None,
    ovr_blocksize: Optional[int] = None,
    intermediate_compression: Union[bool, str, Dict[str, Any]] = False,
    use_windowed_writes: bool = False,
    **extra_rio_opts,
) -> Union[Path, bytes, None]:
    """
    Write COG from externally computed overviews.

    Generates in-memory image with multiple side-car overviews, then re-encodes to destination with
    copy overviews flag set.
    """
    xx = list(layers)
    if len(xx) == 0:
        return None

    if dst != ":mem:":
        _ = check_write_path(dst, overwrite)

    if blocksize is None:
        blocksize = 512

    if ovr_blocksize is None:
        ovr_blocksize = blocksize

    pix = xx[0]
    gbox: GeoBox = pix.odc.geobox
    rio_opts = _default_cog_opts(
        blocksize=blocksize,
        shape=gbox.shape,
        is_float=pix.dtype.kind == "f",
        nodata=pix.attrs.get("nodata", None),
    )
    rio_opts.update(extra_rio_opts)

    first_pass_cfg: Dict[str, Any] = {
        "num_threads": "ALL_CPUS",
        "blocksize": blocksize,
        "nodata": rio_opts.get("nodata", None),
        "use_windowed_writes": use_windowed_writes,
        **_norm_compression_opts(intermediate_compression),
    }

    with _memfiles_ovr(len(xx)) as mm:
        temp_fname = mm[0].name

        # write each layer into mem image
        for img, m in zip(xx, mm):
            _write_cog(
                img.data,
                img.odc.geobox,
                m.name,
                overview_levels=[],
                **first_pass_cfg,
            )

        # copy with recompression
        with rasterio.Env(
            GDAL_TIFF_OVR_BLOCKSIZE=ovr_blocksize,
            GDAL_DISABLE_READDIR_ON_OPEN=False,  # force finding of .ovr(s)
            NUM_THREADS="ALL_CPUS",
            GDAL_NUM_THREADS="ALL_CPUS",
        ):
            if dst == ":mem:":
                with rasterio.MemoryFile() as _dst:
                    rio_copy(temp_fname, _dst.name, copy_src_overviews=True, **rio_opts)
                    return bytes(_dst.getbuffer())  # makes a copy of compressed data
            else:
                rio_copy(temp_fname, dst, copy_src_overviews=True, **rio_opts)
                return Path(dst)
