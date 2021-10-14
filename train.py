
import tensorflow as tf
from keras import backend as K
# my custom lib
from frankModel import LeNet
from frankModel import AlexNet
from frankModel import VGG16
from frankModel import InceptionV1
from frankModel import ResNet50


from small_dataset import MNIST
from small_dataset import CIFAR10
from small_dataset import CIFAR100

try:
    dataset = CIFAR10(info=True).addChannel().tocategorical().Done()


    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)

    #訓練
    # compile  #在每層 layer 和 compile 都可自動尋找超參數
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        #取得模型架構
        MyNet = ResNet50(datasetName=dataset.className,input_shape=(32,32,3) ,classes=len(dataset.train_y[0]))
        
        MyNet.model.compile(
            #learning_rate=0.01
            optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule,epsilon=1e-09),
            loss= 'categorical_crossentropy',
            metrics=['accuracy']
        )

    # fit
    history  = \
    MyNet.model.fit(
        x = dataset.train_x,
        y = dataset.train_y,
        epochs = 30,
        batch_size = 64,
        validation_data = (dataset.test_x ,dataset.test_y)
    )


    MyNet.outputHelper.saveModel()

    # print(history.history)
    r = history.history
    MyNet.outputHelper.drawTrainProcess(r)

except:
    import sys
    # sys.exc_info()[0] 就是用來取出except的錯誤訊息的方法
    print("Unexpected error:", sys.exc_info()[0])

finally:
    K.clear_session()