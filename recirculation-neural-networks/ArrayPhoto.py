import numpy as np


class ArrayPhoto:
  def __init__(self, content):
    self._content = np.array(content)

  def content(self):
    return self._content

  def Back(self, photo):
    return photo
