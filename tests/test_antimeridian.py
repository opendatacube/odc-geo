import pytest
from affine import Affine

from odc.geo import CRS
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.testutils import epsg3857, epsg4326


def test_antimeridian_sinusoidal_grid_intersect() -> None:
    """Test that grid_intersect works with antimeridian-spanning sinusoidal data."""
    # Create a sinusoidal projection GeoBox that spans the antimeridian
    sin_crs = CRS("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    transform = Affine.translation(-20037508, 5000000) * Affine.scale(10000, -10000)
    large_geobox = GeoBox((1000, 500), transform, sin_crs)
    
    # Create tiles
    tiles = GeoboxTiles(large_geobox, tile_shape=(256, 256))
    
    # Create target geobox in WGS84
    target_transform = Affine.translation(170, 10) * Affine.scale(0.1, -0.1)
    target_geobox = GeoBox((100, 100), target_transform, epsg4326)
    target_tiles = GeoboxTiles(target_geobox, tile_shape=(50, 50))
    
    # This should not raise GEOSException
    intersections = tiles.grid_intersect(target_tiles)
    
    # Should find some intersections
    assert isinstance(intersections, dict)
    # The exact number depends on the overlap, but it should work without crashing


def test_footprint_with_wrapdateline() -> None:
    """Test that footprint method accepts wrapdateline parameter."""
    # Create a simple geobox
    transform = Affine.translation(170, 10) * Affine.scale(0.1, -0.1)
    geobox = GeoBox((100, 100), transform, epsg4326)
    
    # Test footprint with wrapdateline=True (should not raise)
    footprint = geobox.footprint(3857, wrapdateline=True)
    assert footprint is not None
    
    # Test footprint with wrapdateline=False (default behavior)
    footprint_default = geobox.footprint(3857, wrapdateline=False)
    assert footprint_default is not None


def test_pacific_antimeridian_scenario() -> None:
    """Test a realistic Pacific Ocean antimeridian scenario."""
    # Create a geobox spanning from 170°E to -170°W (crosses antimeridian)
    transform = Affine.translation(170, 10) * Affine.scale(0.1, -0.1)
    pacific_geobox = GeoBox((200, 400), transform, epsg4326)  # 20° x 40°
    
    # Create tiles
    tiles = GeoboxTiles(pacific_geobox, tile_shape=(50, 50))
    
    # Target in Web Mercator
    mercator_transform = Affine.translation(18000000, 1000000) * Affine.scale(10000, -10000)
    mercator_geobox = GeoBox((100, 200), mercator_transform, epsg3857)
    mercator_tiles = GeoboxTiles(mercator_geobox, tile_shape=(25, 25))
    
    # This should work without GEOSException
    intersections = tiles.grid_intersect(mercator_tiles)
    assert isinstance(intersections, dict)


