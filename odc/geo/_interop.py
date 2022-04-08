"""
Tools for interop with other libraries.

Check if libraries available without importing them which can be slow.
"""
import importlib.util


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

    @staticmethod
    def _check(lib_name):
        return importlib.util.find_spec(lib_name) is not None


have = _LibChecker()
