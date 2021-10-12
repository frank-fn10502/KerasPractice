import numpy as np
import tensorflow as tf
from tensorflow import keras



# my custom lib
from small_dataset import MNIST
from basicModel import LeNet


# 取得資料集
# 推薦 https://keras.io/api/preprocessing/
#   包含 正規化(未完成) 、one-hot(完成)
dataset = MNIST(info=True).addChannel().tocategorical().resizing().Done()

#取得模型架構
MyLeNet = LeNet()

#訓練
#   compile 

MyLeNet.model.compile(
    optimizer= tf.keras.optimizers.Adam(),
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
)
#   在每層 layer 和 compile 都可自動尋找超參數(keras_tuner)

# fit
history  = \
MyLeNet.model.fit(
    x = dataset.train_x,
    y = dataset.train_y,
    epochs = 20,
    batch_size = 32,
    validation_data = (dataset.test_x ,dataset.test_y)
)

#取得訓練結果
#   save the entire model as a single file
#   model = keras.models.load_model("path_to_my_model")
# model.save("path_to_my_model")
MyLeNet.outputHelper.saveModel()

#   包含 訓練中的loss、acc的圖形(畫圖)
MyLeNet.outputHelper.drawTrainProcess(history.history)