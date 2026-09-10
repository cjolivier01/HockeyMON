# Scoreboard selection

The browser selector displays a reduced panorama when the source exceeds 8192
pixels on either axis or a 96 MiB RGBA display budget. The Python
`max_display_height` option can reduce it further. Zoom, selection, initial
points, and saved polygons always use the original image's pixel coordinates.

Only the reduced RGB image is retained and PNG-encoded for the browser. Tensor
inputs are resized on their current device before host transfer. File inputs
still use Pillow's full PNG decoder, so source decoding needs memory for the
original image and remains subject to Pillow's decompression-bomb limits. The
selector additionally rejects sources larger than 65535 pixels per axis or
256 MiPixels before requesting pixel data; this is a display proxy, not a
streaming PNG decoder.
