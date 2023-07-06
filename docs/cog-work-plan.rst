*************************
COG to S3 from Dask Graph
*************************

This document describes a work plan for implementing Dask version of
:py:meth:`~odc.geo.xr.write_cog` capable of streaming result directly to an
`S3`_ compatible object store.

Expected Features
=================

- Generate Cloud Optimized GeoTIFF `COG`_ compliant with the `COG spec`_
- Compress COG tiles concurrently
- Stream compressed tile data directly to S3 in a single pass using multi-part upload
- Support local disk or RAM output without recompression of tiles
- Work with local and distributed Dask clusters
- Work with images bigger than the total cluster memory
- Compute raster statistics and store it in the `GDAL_METADATA`_ tag

.. raw:: html

  <br style="page-break-after: always" />


Work units
==========

Overview Generation
-------------------

Shrink Dask image by an integer factor using configured resampling strategy.

For every ``NxN`` Dask blocks on input, produce a single output block of the
same shape as input blocks. Assume all blocks are of the same size except for
the right and bottom edges of the image. Handle edge pixel problem, e.g.
shrinking ``9x7`` by ``2`` will produce ``5x4`` output, as if input was ``10x8``
with extra columns/rows being set to ``nodata`` or ``nan``.

**Estimate**: 2-4 days


Distributed Tile Compression
----------------------------

Given a single tile of pixels and compression configuration, produce compressed
bytes for that tile and compute pixel statistics (optional). Plan is to use
:py:mod:`rasterio` to generate a single tile TIFF image in memory, then extract
compressed bytes for that tile.

- Should support interleaved pixels, useful for webp and JPEG compression options
- For JPEG compression, only support ``JPEGTABLESMODE=0``, i.e. no shared
  quantization or Huffman tables across tiles, each compressed block is a
  standalone jpeg file.

Use single tile compressor from above to map Dask image to a list of delayed
compressed bytes and list of delayed pixel statistics.

Pixel level statistics per tile should include

- Number of valid pixels, excluding ``nodata`` and ``nan``
- Range of pixel values (``min`` and ``max``)
- Sum of pixel values (for computing overall mean)
- Sum of pixel values squared (for computing overall standard deviation)

**Estimate**: 2-3 days

.. raw:: html

  <br style="page-break-after: always" />


Generate TIFF header
--------------------

Inputs
^^^^^^

- Image metadata: width, height, number of channels, :py:class:`~odc.geo.geobox.GeoBox`
- Compression settings, see `GDAL GeoTIFF documentation`_
- Size in bytes for each compressed tile for the full resolution and the overview images
- Extra TIFF tags to set (pixel statistics and whatever else user supplies)

Given all of the above produce bytes containing TIFF header, such that
``<HEADER>|<TILES>`` forms a valid COG.

Implementation plan
^^^^^^^^^^^^^^^^^^^

Use :py:mod:`rasterio` to generate in memory TIFF compatible with the input
configuration. We can use ``SPARSE_OK=TRUE`` option and do not write any pixel
values. This will take care of the most complex parts of the header,
geo-referencing of the full resolution image(s). Headers for overview images are
much simpler and we should be able to construct those easily in python using
:py:func:`struct.pack`.

Once we have the header with all the tiles set to empty (offset and size set to
0), we can compute final byte offsets for each tile, since we now know the size
of the header. Then it's a matter of patching offsets and sizes in the header
with the real values.

**Estimate**: 3-5 days

.. raw:: html

  <br style="page-break-after: always" />


Stream to S3
------------

Given a list of delayed byte chunks, write them out in order to an object in S3
using multi-part upload functionality using configured "part range". Report back
sizes of observed byte chunks, and parts that were written to S3.

AWS S3 multi-part upload must be done in chunks of at least 5Mb (except the very
last chunk) and at most 5Gb. One can have up to 10,000 chunks, with chunk id
determining the order of chunks.

Inputs and configuration
^^^^^^^^^^^^^^^^^^^^^^^^

- Destination path
- Number of sub-streams

For each sub-stream

- List of Dask delayed objects that evaluate to bytes

  - These need to be independent of each other, one should be able to compute
    any one of them without forcing evaluation of others
  - In the case of COG, tiles for each overview will need to be in a separate sub-stream
  - Size of each chunk is unknown at Dask construction time

- Part range per sub-stream, e.g. ``[2000, 3000)``
- Desired part size, e.g. ``200MB``, aim to write out parts of at least that size
- Expected approximate chunk size in bytes (used to estimate number of blocks to
  process concurrently). This is needed to support ordered writes without
  running out of RAM or performing second pass over the data for re-ordering.

Outputs
^^^^^^^

Size in bytes for each chunk written.

Implementation plan
^^^^^^^^^^^^^^^^^^^

For each sub-stream create Dask graph in a form of a list.

- Pick number of chunks (``N``) to process concurrently, used for forcing on
  Dask approximate evaluation order

  - Ideally this should be chosen such that
  - ``sum(size_in_bytes(c) for c in chunks[0:N]) ~= desired_part_size)``

- Create Dask graph in a shape of a list, with chunks ``i*N:(i+1)*N`` having a
  dependency on the result of chunks ``(i-1)*N:i*N``

Essentially we are implementing a reduction operation within each sub-stream.
Each step of the reduction takes the state so far and the next ``N`` byte
chunks. Chunks are concatenated to the cache from the state, and if the cache
has enough bytes, new part is written out to S3. New state is then returned and
feeds into the next step of the reduction operation. We'll need to ensure that
there is at least 5Mb worth of data in the cache left over after writing a part,
as we do not know how many more bytes we will receive in the future, and 5Mb is
the minimum allowed part size.

This one is the most uncertain and will require non-trivial test harness setup.
There is some non-zero probability of me missing some important constraint on
the S3 part. There is a high chance of needing to review interface as
implementation progresses.

**Estimate**: 4+ days


Integration and Testing
-----------------------

Put all the parts from above together, add docs, make a release.

**Estimate**: ~5 days

Summary
=======

Assume 1 day is 8hr

======================  ========
Task                    Estimate
======================  ========
Overviews                   2-4d
Tile Compression            2-3d
TIFF Header                 3-5d
Stream to S3                 4d+
Integration                  ~5d
----------------------  --------
**Total**                 16-25d
======================  ========



.. _`COG spec`: https://github.com/cogeotiff/cog-spec/blob/master/spec.md
.. _`COG`: https://www.cogeo.org/
.. _`S3`: https://aws.amazon.com/s3/
.. _`GDAL_METADATA`: https://www.awaresystems.be/imaging/tiff/tifftags/gdal_metadata.html
.. _`GDAL GeoTIFF documentation`: https://gdal.org/drivers/raster/gtiff.html#creation-options