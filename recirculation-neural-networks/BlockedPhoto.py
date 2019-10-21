class UnblockedPhoto:
  def __init__(self, block_shape, photo_shape, blocked_photo):
    self.original = blocked_photo
    self.photo_shape = photo_shape
    self.block_shape = block_shape

  def unblock(self, blocked):
    h = self.photo_shape[0]
    nrows = self.block_shape[0]
    ncols = self.block_shape[1]
    blocked = blocked.reshape((blocked.shape[0], nrows, ncols) + self.photo_shape[2:])
    return blocked.reshape((h // nrows, -1, nrows, ncols) + self.photo_shape[2:]).swapaxes(1, 2).reshape(
      self.photo_shape)

  def content(self):
    blocked = self.original.content()
    return self.unblock(blocked)

class BlockedPhoto:
  def __init__(self, photo, block_shape):
    self.row_n = block_shape[0]
    self.column_n = block_shape[1]
    self.original = photo

  def block(self, array):
    shape = array.shape
    h = shape[0]
    nrows = self.row_n
    ncols = self.column_n
    blocks = array.reshape((h // nrows, nrows, -1, ncols) + shape[2:]).swapaxes(1, 2).reshape(
      (-1, nrows, ncols) + shape[2:])
    return blocks.reshape(blocks.shape[0], -1)

  def content(self):
    content = self.original.content()
    return self.block(content)

  def Back(self, photo):
    back = UnblockedPhoto((self.row_n, self.column_n), self.original.content().shape, photo)
    return self.original.Back(back)
