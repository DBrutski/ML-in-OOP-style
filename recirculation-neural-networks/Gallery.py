import os

from Photo import Photo


def directory_files(walk_triple):
  (directory, directories, files) = walk_triple
  for file in files:
    yield os.path.join(directory, file)

class Gallery:
  def __init__(self, gallery_path):
    self.gallery_path = gallery_path

  def photos(self):
    for directory in os.walk(self.gallery_path):
      for file in directory_files(directory):
        yield Photo(file)
