import cv2


class Photo:
  def __init__(self, path):
    self.file_path = path

  def content(self):
    return cv2.imread(self.file_path)

  def Back(self, photo):
    return photo
