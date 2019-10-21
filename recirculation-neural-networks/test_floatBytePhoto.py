from unittest import TestCase

import numpy as np
from ArrayPhoto import ArrayPhoto
from FloatBytePhoto import FloatBytePhoto, RestoredFloatBytePhoto
from Gallery import Gallery


class TestFloatBytePhoto(TestCase):
  def test_content(self):
    gallery = Gallery("photoes")
    photo = next(gallery.photos())

    restored_photo = RestoredFloatBytePhoto(FloatBytePhoto(photo))
    np.testing.assert_array_equal(restored_photo.content(), photo.content(), verbose=True)

  def test_byte_by_byte(self):
    gallery = Gallery("photoes")
    photo = next(gallery.photos())

    restored_photo = RestoredFloatBytePhoto(FloatBytePhoto(photo))
    for left, right in zip(np.nditer(restored_photo.content()), np.nditer(photo.content())):
      if left != right:
        print(left, right)

  def test_(self):
    photo = ArrayPhoto([49, 26, 41, 10])

    restored_photo = RestoredFloatBytePhoto(FloatBytePhoto(photo))
    np.testing.assert_array_equal(restored_photo.content(), photo.content(), verbose=True)
    print(restored_photo.content())
    print(photo.content())
