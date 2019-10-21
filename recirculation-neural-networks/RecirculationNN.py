from keras.layers import Dense
from keras import Sequential


class RecirculationNetwork:

  def __init__(self, block_shape, output_block_size, tensor_board=None, epochs=100, callbacks=None):
    self._callbacks = callbacks
    self._epochs = epochs
    self._tensor_board = tensor_board
    self._block_shape = block_shape
    self._output_block_size = output_block_size
    input_size = block_shape[0] * block_shape[1] * 3
    self._forward = Dense(output_block_size, input_dim=input_size, activation="linear")
    self._backward = Dense(input_size, activation="linear")
    self._forwardModel = Sequential([self._forward])
    self._backwardModel = Sequential([self._backward])
    self._model = Sequential([self._forward, self._backward])

  def fit(self, blocks):
    self._model.compile(loss='mean_squared_error', optimizer='sgd')
    if self._tensor_board:
      self._model.fit(blocks, blocks, epochs=self._epochs, use_multiprocessing=True,
                      callbacks=[self._tensor_board.callback()])
    else:
      self._model.fit(blocks, blocks, epochs=self._epochs, use_multiprocessing=True)

  def forward(self, blocks):
    return self._forwardModel.predict(blocks)

  def backward(self, blocks):
    return self._backwardModel.predict(blocks)
