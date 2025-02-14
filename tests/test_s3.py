"""Tests for odc.geo.cog._s3."""

import unittest

from odc.geo.cog._s3 import S3MultiPartUpload

# Conditional import for S3 support
try:
    from odc.geo.cog._s3 import S3MultiPartUpload

    HAVE_S3 = True
except ImportError:
    S3MultiPartUpload = None
    HAVE_S3 = False


def require_s3(test_func):
    """Decorator to skip tests if s3 dependencies are not installed."""
    return unittest.skipUnless(HAVE_S3, "s3 dependencies are not installed")(test_func)


@require_s3
def test_s3_mpu():
    """Test S3MultiPartUpload class initialization."""
    mpu = S3MultiPartUpload("bucket", "file.dat")
    if mpu.bucket != "bucket":
        raise ValueError("Invalid bucket")
    if mpu.key != "file.dat":
        raise ValueError("Invalid key")
