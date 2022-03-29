GeoBox Model
############

Probably the main abstraction in :py:mod:`odc.geo` is a |gbx|. It defines a geo-registered bounded
image plane. Image width and height, CRS and an affine matrix fully define geo-registered pixel
plane. From this, one can compute footprint on the surface of the earth for the whole or part of the
image.

Affine matrix defines linear mapping from pixel coordinates :math:`X_p` to the world :math:`X_w`. In
the most generic form affine matrix has 6 independent parameters:

.. math::
   :nowrap:

    \begin{eqnarray*}
       X_w &=& A X_p \\
       \begin{pmatrix}x_w \\ y_w \\ 1\end{pmatrix} &=&
       \begin{pmatrix}
         a & b & c \\
         d & e & f \\
         0 & 0 & 1
       \end{pmatrix}
       \begin{pmatrix}x_p \\ y_p \\ 1\end{pmatrix}
    \end{eqnarray*}

However most data sources use square pixels aligned to :math:`x_w`, :math:`y_w` world coordinates with
image :math:`y_p` axis flipped to match image coordinate convention. This form looks like this:

.. math::
   :nowrap:

   \begin{equation}
    \begin{pmatrix}x_w \\ y_w \\ 1\end{pmatrix} =
    \begin{pmatrix}
       s  &   0 & t_x \\
       0  &  -s & t_y \\
       0  &   0 & 1
    \end{pmatrix}
    \begin{pmatrix}x_p \\ y_p \\ 1\end{pmatrix}
   \end{equation}


Image coordinate system we use is the same as used by GDAL: point :math:`(0, 0)` is a top left
corner of the top left pixel of the image. The center of the top-left pixel is at :math:`(0.5, 0.5)`.
Valid range of pixel coordinates spans :math:`x_p \in [0, width], y_p \in [0, height]`.

.. jupyter-execute::

   from odc.geo.geobox import GeoBox
   GeoBox.from_bbox(
      (-2_000_000, -5_000_000,
        2_250_000, -1_000_000),
      "epsg:3577", resolution=1000)

In the sample above we have constructed a |gbx| in the Australia Albers projection with 1km pixels.
It describes an image plane 4,250 pixels wide by 4,000 pixels tall. Mapping from pixel coordinates
to the world coordinates in EPSG:3577 is defined by the affine matrix listed below:

.. math::
   :nowrap:

   \begin{equation}
    \begin{pmatrix}x_w \\ y_w \\ 1\end{pmatrix} =
    \begin{pmatrix}
       1000 &  0    & -2,000,000 \\
       0    & -1000 & -1,000,000 \\
       0    &  0    & 1
    \end{pmatrix}
    \begin{pmatrix}x_p \\ y_p \\ 1\end{pmatrix}
   \end{equation}

Here :math:`1000` is a size of the pixel in meters, and :math:`(-2,000,000, -1,000,000)` is the
location of the top left corner of the top left pixel (point :math:`(0, 0)` in the pixel coordinate system).
Top left corner is marked by a circle on the map above.


.. |gbx| replace:: :py:class:`~odc.geo.geobox.GeoBox`
