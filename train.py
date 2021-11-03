import tensorflow as tf
from keras import backend as K
# my custom lib
from frankModel import LeNet
from frankModel import AlexNet
from frankModel import VGG16
from frankModel import InceptionV1
from frankModel import ResNet50
from frankModel import EfficientNetV2_S


from small_dataset import MNIST, Dataset
from small_dataset import CIFAR10
from small_dataset import CIFAR100


from utils.outputs import ModelOuputHelper

class Train:
    def __init__(self ,net) -> None:
        self.initial_learning_rate = 1e-4
        self.mainDirectory = 'result'
        self.epoch = 30
        self.batchSize = 64
        self.net = net

    def process(self ,dataset):
        myNet = self.__prepareTrain()
        outputHelper = ModelOuputHelper(myNet.model ,myNet.verMark ,dataset.className ,main_directory=self.mainDirectory)

        outputHelper.seveModelArchitecture() # 儲存架構

        history = self.__train(myNet) #訓練

        outputHelper.saveModel()
        outputHelper.saveTrainProcessImg(history)
        outputHelper.saveTrainHistory()

    def __prepareTrain(self) -> Dataset:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)

        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            #取得模型架構
            myNet = self.net(input_shape=(32,32,3) ,classes=len(dataset.train_y[0]))
            
            myNet.model.compile(
                #learning_rate=0.01
                optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule,epsilon=1e-09),
                loss= 'categorical_crossentropy',
                metrics=['accuracy']
            )
        return myNet

    def __train(self ,myNet) -> dict:
        return myNet.model.fit(
                    x = dataset.train_x,
                    y = dataset.train_y,
                    epochs = self.epoch,
                    batch_size = self.batchSize,
                    validation_data = (dataset.test_x ,dataset.test_y)
                )

dataset = CIFAR10(info=True).addChannel().tocategorical().Done()

train = Train(EfficientNetV2_S)
train.process(dataset)