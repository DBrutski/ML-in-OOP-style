from unittest import TestCase

import numpy as np
from ArrayPhoto import ArrayPhoto
from BlockedPhoto import BlockedPhoto

class TestBlockedPhoto(TestCase):

  def test_np_array_shape(self):
    self.assertEqual(np.array([1, 1, 1]).shape, (3,))
    self.assertEqual(np.array([[1], [1], [1]]).shape, (3, 1,))
    self.assertEqual(np.array([
      [[[1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0]],
       [[2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0]],
       [[3, 1, 0], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0]],
       [[4, 1, 0], [4, 2, 0], [4, 4, 0], [4, 4, 0], [4, 5, 0]],
       [[5, 1, 0], [5, 2, 0], [5, 5, 0], [5, 5, 0], [5, 5, 0]]]]).shape,
                     (1, 5, 5, 3,))

  def test_2_2(self):
    photo = ArrayPhoto([[1, 2, 3, 4], [5, 6, 7, 8]])
    cut = BlockedPhoto(photo, (2, 2))
    expected = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]])
    np.testing.assert_array_equal(cut.content(), expected)

  def test_3_2(self):
    photo = ArrayPhoto([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    cut = BlockedPhoto(photo, (3, 2))
    expected = np.array([[[1, 2], [5, 6], [9, 10]], [[3, 4], [7, 8], [11, 12]]])
    np.testing.assert_array_equal(cut.content(), expected)

  def test_3_2_nested(self):
    photo = ArrayPhoto([
      [[1, 1], [1, 2], [1, 3], [1, 4]],
      [[2, 1], [2, 2], [2, 3], [2, 4]],
      [[3, 1], [3, 2], [3, 3], [3, 4]]])

    array = photo.content()
    array.swapaxes(0, 1).reshape(-1, )
    cut = BlockedPhoto(photo, (3, 2))
    expected = np.array([
      [[[1, 1], [1, 2]],
       [[2, 1], [2, 2]],
       [[3, 1], [3, 2]]],
      [[[1, 3], [1, 4]],
       [[2, 3], [2, 4]],
       [[3, 3], [3, 4]]]])
    np.testing.assert_array_equal(cut.content(), expected)

  def test_content_10_5(self):
    photo = ArrayPhoto(
      [[[1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0], [1, 8, 0], [1, 9, 0], [1, 10, 0]],
       [[2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0], [2, 6, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0], [2, 10, 0]],
       [[3, 1, 0], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0], [3, 6, 0], [3, 7, 0], [3, 8, 0], [3, 9, 0], [3, 10, 0]],
       [[4, 1, 0], [4, 2, 0], [4, 4, 0], [4, 4, 0], [4, 5, 0], [4, 6, 0], [4, 7, 0], [4, 8, 0], [4, 9, 0], [4, 10, 0]],
       [[5, 1, 0], [5, 2, 0], [5, 5, 0], [5, 5, 0], [5, 5, 0], [5, 6, 0], [5, 7, 0], [5, 8, 0], [5, 9, 0], [5, 10, 0]]])

    cut = BlockedPhoto(photo, (5, 5))

    expected = np.array([
      [1, 1, 0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0,
       2, 1, 0, 2, 2, 0, 2, 3, 0, 2, 4, 0, 2, 5, 0,
       3, 1, 0, 3, 2, 0, 3, 3, 0, 3, 4, 0, 3, 5, 0,
       4, 1, 0, 4, 2, 0, 4, 4, 0, 4, 4, 0, 4, 5, 0,
       5, 1, 0, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0],
      [1, 6, 0, 1, 7, 0, 1, 8, 0, 1, 9, 0, 1, 10, 0,
       2, 6, 0, 2, 7, 0, 2, 8, 0, 2, 9, 0, 2, 10, 0,
       3, 6, 0, 3, 7, 0, 3, 8, 0, 3, 9, 0, 3, 10, 0,
       4, 6, 0, 4, 7, 0, 4, 8, 0, 4, 9, 0, 4, 10, 0,
       5, 6, 0, 5, 7, 0, 5, 8, 0, 5, 9, 0, 5, 10, 0]
    ])
    np.testing.assert_array_equal(cut.content(), expected)
    np.testing.assert_array_equal(cut.Back(cut).content(), photo.content())
