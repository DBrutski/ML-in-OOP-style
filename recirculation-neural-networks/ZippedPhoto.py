class ZippedPhoto:
  def __init__(self, photo, recirculation_network, shape):
    self.shape = shape
    self.original = photo
    self.recirculation_network = recirculation_network

  def content(self):
    blocks = self.original.content()
    self.recirculation_network.fit(blocks)
    return self.recirculation_network.forward(blocks)

  def Back(self):
    back = SelfExtractablePhoto(self, self.recirculation_network.backward)
    if "Back" in dir(self.original):
      return self.original.Back(back)
    else:
      return back

class TrainedZippedPhoto(ZippedPhoto):
  def __init__(self, photo, recirculation_network, shape):
    super().__init__(photo, recirculation_network, shape)
    self.shape = shape
    self.original = photo
    self.recirculation_network = recirculation_network

  def content(self):
    blocks = self.original.content()
    return self.recirculation_network.forward(blocks)

  def Back(self):
    back = SelfExtractablePhoto(self, self.recirculation_network.backward)
    if "Back" in dir(self.original):
      return self.original.Back(back)
    else:
      return back

class SelfExtractablePhoto:
  def __init__(self, zipped_photo, extractor):
    self.original = zipped_photo
    self.extractor = extractor

  def content(self):
    blocks = self.original.content()
    return self.extractor(blocks)
