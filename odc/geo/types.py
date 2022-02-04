"""Basic types."""
from typing import (
    Generic,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

MaybeInt = Optional[int]
MaybeFloat = Optional[float]
T = TypeVar("T")


class XY(Generic[T]):
    """
    Immutable container for anything X/Y.

    This class is used as a replacement for a plain tuple of two values that could be in X/Y or Y/X
    order.

    :param x: Value of type ``T`` for x
    :param y: Value of type ``T`` for y
    """

    __slots__ = ("_xy",)

    def __init__(self, x: T, y: T) -> None:
        self._xy = x, y

    def __eq__(self, other) -> bool:
        if not isinstance(other, XY):
            return False
        return self._xy == other._xy

    def __str__(self) -> str:
        return f"XY(x={self._xy[0]}, y={self._xy[1]})"

    def __repr__(self) -> str:
        return f"XY(x={self._xy[0]}, y={self._xy[1]})"

    def __hash__(self) -> int:
        return hash(self._xy)

    @property
    def x(self) -> T:
        """Access X value."""
        return self._xy[0]

    @property
    def y(self) -> T:
        """Access Y value."""
        return self._xy[1]

    @property
    def xy(self) -> Tuple[T, T]:
        """Convert to tuple in X,Y order."""
        return self._xy

    @property
    def yx(self) -> Tuple[T, T]:
        """Convert to tuple in Y,X order."""
        return self._xy[1], self._xy[0]

    @property
    def lon(self) -> T:
        """Access Longitude value (X)."""
        return self._xy[0]

    @property
    def lat(self) -> T:
        """Access Latitude value (Y)."""
        return self._xy[1]

    @property
    def lonlat(self) -> Tuple[T, T]:
        """Convert to tuple in Longitude,Latitude order."""
        return self._xy

    @property
    def latlon(self) -> Tuple[T, T]:
        """Convert to tuple in Latitude,Longitude order."""
        return self._xy[1], self._xy[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Interpret as ``shape`` (Y, X) order.

        Only valid for ``XY[int]`` case.

        :raises ValueError: when tuple contains anything but integer values
        :return: ``(y, x)``
        """
        x, y = self._xy
        if isinstance(x, int) and isinstance(y, int):
            return y, x
        raise ValueError("Expect (int, int) for shape")

    @property
    def wh(self) -> Tuple[int, int]:
        """
        Interpret as ``width, height``, (X, Y) order.

        Only valid for ``XY[int]`` case.

        :raises ValueError: when tuple contains anything but integer values
        :return: ``(x, y)``
        """
        x, y = self._xy
        if isinstance(x, int) and isinstance(y, int):
            return (x, y)
        raise ValueError("Expect (int, int) for wh")


class Resolution(XY[float]):
    """
    Resolution for X/Y dimensions.

    """

    def __init__(self, x: float, y: Optional[float] = None) -> None:
        if y is None:
            y = -x
        super().__init__(float(x), float(y))

    def __repr__(self) -> str:
        return f"Resolution(x={self.x:g}, y={self.y:g})"

    def __str__(self) -> str:
        return f"Resolution(x={self.x:g}, y={self.y:g})"


class Index2d(XY[int]):
    """
    2d index.
    """

    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"Index2d(x={self.x}, y={self.y})"

    def __str__(self) -> str:
        return f"Index2d(x={self.x}, y={self.y})"


# fmt: off
@overload
def xy_(x: T, y: T, /) -> XY[T]: ...
@overload
def xy_(x: Iterable[T], y: Literal[None] = None, /) -> XY[T]: ...
@overload
def xy_(x: XY[T], y: Literal[None] = None, /) -> XY[T]: ...
# fmt: on


def xy_(x: Union[T, XY[T], Iterable[T]], y: Optional[T] = None) -> XY[T]:
    """
    Construct from X,Y order.

    .. code-block:: python

       xy_(0, 1)
       xy_([0, 1])
       xy_(tuple([0, 1]))
       assert xy_(1, 3).x == 1
       assert xy_(1, 3).y == 3
    """
    if y is not None:
        return XY(x=cast(T, x), y=y)

    if isinstance(x, XY):
        return x

    if not isinstance(x, Iterable):
        raise ValueError("Expect 2 arguments or a single iterable.")

    x, y = cast(Iterable[T], x)
    return XY(x=x, y=y)


# fmt: off
@overload
def yx_(y: T, x: T, /) -> XY[T]: ...
@overload
def yx_(y: Iterable[T], x: Literal[None] = None, /) -> XY[T]: ...
@overload
def yx_(y: XY[T], x: Literal[None] = None, /) -> XY[T]: ...
# fmt: on


def yx_(y: Union[T, XY[T], Iterable[T]], x: Optional[T] = None, /) -> XY[T]:
    """
    Construct from Y,X order.

    .. code-block:: python

       yx_(0, 1)
       yx_([0, 1])
       yx_(tuple([0, 1]))
       assert yx_(1, 3).x == 3
       assert yx_(1, 3).y == 1
    """
    if x is not None:
        return XY(x=x, y=cast(T, y))

    if isinstance(y, XY):
        return y

    if not isinstance(y, Iterable):
        raise ValueError("Expect 2 arguments or a single iterable.")

    y, x = cast(Iterable[T], y)
    return XY(x=x, y=y)


def res_(x: Union[Resolution, float, int], /) -> Resolution:
    """Resolution for square pixels with inverted Y axis."""
    if isinstance(x, Resolution):
        return x
    if isinstance(x, (int, float)):
        return Resolution(float(x))
    raise ValueError("Unsupported input type: res_(x: {type(x)})")


def resxy_(x: float, y: float, /) -> Resolution:
    """Construct resolution from X,Y order."""
    return Resolution(x=x, y=y)


def resyx_(y: float, x: float, /) -> Resolution:
    """Construct resolution from Y,X order."""
    return Resolution(x=x, y=y)


# fmt: off
@overload
def ixy_(x: int, y: int, /) -> Index2d: ...
@overload
def ixy_(x: Tuple[int, int], y: Literal[None] = None, /) -> Index2d: ...
@overload
def ixy_(x: Index2d, y: Literal[None] = None, /) -> Index2d: ...
@overload
def ixy_(x: XY[int], y: Literal[None] = None, /) -> Index2d: ...
# fmt: on


def ixy_(
    x: Union[int, Tuple[int, int], XY[int], Index2d], y: Optional[int] = None, /
) -> Index2d:
    """Construct 2d index in X,Y order."""
    if y is not None:
        assert isinstance(x, int)
        return Index2d(x=x, y=y)
    if isinstance(x, tuple):
        x, y = x
        return Index2d(x=x, y=y)
    if isinstance(x, Index2d):
        return x
    if isinstance(x, XY):
        x, y = x.xy
        return Index2d(x=x, y=y)
    raise ValueError("Expect 2 values or a single tuple/XY/Index2d object")


# fmt: off
@overload
def iyx_(y: int, x: int, /) -> Index2d: ...
@overload
def iyx_(y: Tuple[int, int], x: Literal[None] = None, /) -> Index2d: ...
@overload
def iyx_(y: Index2d, x: Literal[None] = None, /) -> Index2d: ...
@overload
def iyx_(y: XY[int], x: Literal[None] = None, /) -> Index2d: ...
# fmt: on


def iyx_(
    y: Union[int, Tuple[int, int], XY[int], Index2d], x: Optional[int] = None, /
) -> Index2d:
    """Construct 2d index in Y,X order."""
    if x is not None:
        assert isinstance(y, int)
        return Index2d(x=x, y=y)
    if isinstance(y, tuple):
        y, x = y
        return Index2d(x=x, y=y)
    if isinstance(y, Index2d):
        return y
    if isinstance(y, XY):
        y, x = y.yx
        return Index2d(x=x, y=y)
    raise ValueError("Expect 2 values or a single tuple/XY/Index2d object")


SomeShape = Union[Tuple[int, int], XY[int], Index2d]
SomeIndex2d = Union[Tuple[int, int], XY[int], Index2d]
SomeResolution = Union[float, int, Resolution]
