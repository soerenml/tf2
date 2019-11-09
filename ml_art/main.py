from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib

data_dir = pathlib.Path("/Users/sorenprivat/data/art_data/resized")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)