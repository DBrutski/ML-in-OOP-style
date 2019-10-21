import argparse
import cv2
from BlockedPhoto import BlockedPhoto
from FloatBytePhoto import FloatBytePhoto

from Gallery import Gallery

from RecirculationNN import RecirculationNetwork
from ResizedPhoto import ResizedPhoto
from TensorBoard import TensorBoard
from ZippedPhoto import ZippedPhoto


def cycle(chunck_size, array):
  i = 0
  while i + chunck_size < len(array):
    result = array[i:i + chunck_size]
    i = i + chunck_size
    yield result


def blocked_photo(photo, photo_shape, block_shape):
  return BlockedPhoto(
    FloatBytePhoto(
      ResizedPhoto(
        photo,
        photo_shape
      )
    ),
    block_shape
  )


def zipped_photo(photo, photo_shape, block_shape, output_size):
  content = photo.content()
  return ZippedPhoto(
    blocked_photo(photo, photo_shape, block_shape),
    RecirculationNetwork(
      block_shape,
      output_size,
      tensor_board=TensorBoard("logs/")
    ),
    content.shape
  )


def train_network(photo, photo_shape, block_shape, output_size):
  photo = blocked_photo(photo, photo_shape, block_shape)
  network = RecirculationNetwork(block_shape, output_size)
  network.fit(photo.content())
  return network


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-g", "--gallery", required=True, help="path to the photoes gallery")
  args = ap.parse_args()

  gallery = Gallery(args.gallery)
  photos = gallery.photos()
  photo = next(photos)

  photo_shape = (500, 400)
  input_shape = (4, 4)
  output = 16 * 3
  archived_photo = zipped_photo(photo, photo_shape, input_shape, output)
  content = archived_photo.Back().content()
  cv2.imshow("Image", content)
  cv2.waitKey(0)


if __name__ == "__main__":
  main()
