# pylint: disable=import-outside-toplevel


def test_version() -> None:
    from odc.geo import __version__

    assert __version__ is not None
