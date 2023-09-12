"""
Tools for interop with other libraries.

Check if libraries available without importing them which can be slow.
"""
import importlib.util
from typing import Any, Callable, Optional

# pylint: disable=import-outside-toplevel


class _LibChecker:
    @property
    def rasterio(self) -> bool:
        return self._check("rasterio")

    @property
    def xarray(self) -> bool:
        return self._check("xarray")

    @property
    def geopandas(self) -> bool:
        return self._check("geopandas")

    @property
    def dask(self) -> bool:
        return self._check("dask")

    @property
    def folium(self) -> bool:
        return self._check("folium")

    @property
    def ipyleaflet(self) -> bool:
        return self._check("ipyleaflet")

    @property
    def datacube(self) -> bool:
        return self._check("datacube")

    @property
    def tifffile(self) -> bool:
        return self._check("tifffile")

    @staticmethod
    def _check(lib_name: str) -> bool:
        return importlib.util.find_spec(lib_name) is not None

    def check_or_error(self, *libs: str, msg: Optional[str] = None) -> None:
        for lib in libs:
            if not self._check(lib):
                if msg is None:
                    msg = f"Missing library: {lib}"
                raise RuntimeError(msg)


have = _LibChecker()
__all__ = ("have",)


def __dir__():
    return [*__all__, "is_dask_collection"]


def __getattr__(name):
    if name == "is_dask_collection":
        if have.dask:
            import dask

            return dask.is_dask_collection

        # pylint: disable=redefined-outer-name
        def is_dask_collection(_: Any) -> bool:
            return False

        return is_dask_collection

    raise AttributeError(f"module {__name__} has no attribute {name}")


is_dask_collection: Callable[[Any], bool]
