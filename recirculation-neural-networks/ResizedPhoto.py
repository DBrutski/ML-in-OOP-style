import cv2

class ResizedPhoto:
  def __init__(self, photo, shape):
    self.original = photo
    self.target_shape = shape

  def content(self):
    content = self.original.content()
    return cv2.resize(content, (self.target_shape[1], self.target_shape[0]))

  def Back(self, photo):
    back = ResizedPhoto(photo, self.original.content().shape)
    return self.original.Back(back)
