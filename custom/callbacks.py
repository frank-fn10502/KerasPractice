import tensorflow as tf
from tensorflow import keras

class TestCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # print(f"epoch: {epoch},\nmodel:{self.model},\ngot log keys:{logs.keys()}")
        
        self.model.layers[1].setNewMagnitude(epoch)
        self.model.layers[1].setResizing(epoch)
        
        print(f"magnitude: {self.model.layers[1].magnitude}")
        print(f"imgSize: {self.model.layers[1].imgSize}")
        pass