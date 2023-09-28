from odc.geo.cog._s3 import MultiPartUpload

# TODO: moto


def test_s3_mpu():
    mpu = MultiPartUpload("bucket", "file.dat")
    assert mpu.bucket == "bucket"
    assert mpu.key == "file.dat"
