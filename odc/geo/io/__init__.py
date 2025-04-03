"""
Data loading methods
"""

from .._interop import have

__all__ = []

if have.laspy:
    from ._las import load_las

    __all__.append("load_las")
