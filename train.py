import tensorflow as tf

# my custom lib
from custom.frankModel import BasicModel, LeNet, AlexNet, VGG16, InceptionV1, ResNet50, EfficientNetV2_S
from custom.interface.Dataset import Dataset
from custom.small_dataset import MNIST, SmallDataset ,CIFAR10 ,CIFAR100
from custom.fileDataset import Flowers

from custom.callbacks import TestCallback
from custom.layer import DistortImage
from utils.outputs import ModelOuputHelper

class Train:
    def __init__(self ,net) -> None:
        self.initial_learning_rate = 1e-4
        self.mainDirectory = 'result'
        self.epoch = 30
        self.batchSize = 64
        self.net = net

    def process(self ,dataset):
        myNet = self.__prepareTrain(dataset)
        outputHelper = ModelOuputHelper(myNet.model ,myNet.verMark ,dataset.className ,main_directory=self.mainDirectory)

        outputHelper.seveModelArchitecture() # 儲存架構

        history = self.__train(myNet ,dataset ,self.__callback(outputHelper)) #訓練

        #outputHelper.saveModel() #有 callback 狀況下也許不是很有必要?
        outputHelper.saveTrainProcessImg(history)
        outputHelper.saveTrainHistory()

    def __prepareTrain(self ,dataset) -> BasicModel:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)

        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            #取得模型架構
            myNet = self.net(input_shape=(*dataset.imgSize,3) ,classes=len(dataset.labels[0]) ,image_preProcess = DistortImage(self.epoch))
            
            myNet.model.compile(
                #learning_rate=0.01
                optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule,epsilon=1e-09),
                loss= 'categorical_crossentropy',
                metrics=['accuracy']
            )
        return myNet

    def __train(self ,myNet ,dataset ,callbacks) -> dict:
        if not dataset.isTfDataset:
            return myNet.model.fit(
                        x = dataset.train_x,
                        y = dataset.train_y,
                        epochs = self.epoch,
                        batch_size = self.batchSize,
                        validation_data = (dataset.test_x ,dataset.test_y),
                        callbacks = callbacks
                    )
        else:
            return myNet.model.fit(
                dataset.train,
                epochs = self.epoch,
                validation_data = dataset.validation,
                callbacks = callbacks
            )

    def __callback(self ,outputHelper):
        return [
            tf.keras.callbacks.ModelCheckpoint(outputHelper.train_result_dir / '{epoch:02d}-{val_accuracy:.2f}.h5', 
                                               save_weights_only=True,
                                               monitor='val_accuracy',
                                               mode='max',
                                               save_best_only=True),
            TestCallback()
        ]


dataset = Flowers(info=True ,labelPath = 'dataset/flowers/imagelabels.mat' ,imagePath = 'dataset/flowers/')
dataset.batchSize = 64
dataset.imgSize = (256 ,256)
dataset = dataset.tocategorical().Done()

train = Train(EfficientNetV2_S)
# train.process(dataset)