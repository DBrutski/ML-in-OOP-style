import numpy as np


class RestoredFloatBytePhoto:
  def __init__(self, float_byte_photo):
    self.original = float_byte_photo

  @staticmethod
  def restore_float(f):
    if f < -1:
      return -1
    elif f > 1:
      return 1
    else:
      return f

  def content(self):
    float_bytes = self.original.content()
    float_bytes = np.vectorize(self.restore_float)(float_bytes)
    return ((float_bytes + 1) * 255 / 2 + 0.5).astype("uint8")

class FloatBytePhoto:
  def __init__(self, photo):
    self.original = photo

  def content(self):
    rgb = self.original.content()
    return (rgb.astype(float) / 255 * 2) - 1

  def Back(self, photo):
    back = RestoredFloatBytePhoto(photo)
    return self.original.Back(back)
