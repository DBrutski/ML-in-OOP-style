from datetime import datetime
import keras

class TensorBoard:

  def __init__(self, log_dir):
    self.log_dir = log_dir

  def callback(self):
    log_dir = self.log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
    return keras.callbacks.TensorBoard(log_dir=log_dir)
