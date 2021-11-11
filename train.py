import tensorflow as tf
import os
# my custom lib
from custom.frankModel import BasicModel, LeNet, AlexNet, VGG16, InceptionV1, ResNet50, EfficientNetV2_S
from custom.interface.Dataset import Dataset
from custom.small_dataset import MNIST, SmallDataset ,CIFAR10 ,CIFAR100
from custom.fileDataset import Flowers

from custom.callbacks import TestCallback
from custom.layer import DistortImage
from utils.outputs import ModelOuputHelper
from custom.utils.utils import VersionMark as verMark
from custom.utils.other.learningRate import WarmUpAndCosineDecay

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class Train:
    def __init__(self ,net) -> None:
        self.initial_learning_rate = 1e-4
        self.mainDirectory = 'result'
        self.epoch = 30
        self.batchSize = 64
        self.net = net

    def process(self ,dataset ,noPreProcess = False ,verMark = None):
        myNet = self.__prepareTrain(dataset ,noPreProcess ,verMark)
        outputHelper = ModelOuputHelper(myNet.model ,myNet.verMark ,dataset.className ,main_directory=self.mainDirectory)

        outputHelper.seveModelArchitecture() # 儲存架構

        history = self.__train(myNet ,dataset ,self.__callback(outputHelper)) #訓練

        #outputHelper.saveModel() #有 callback 狀況下也許不是很有必要?
        outputHelper.saveTrainHistory(history)
        outputHelper.saveTrainProcessImg(history)

    def __prepareTrain(self ,dataset ,noPreProcess ,verMark = None) -> BasicModel:
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     self.initial_learning_rate,
        #     decay_steps=1000,
        #     decay_rate=0.96,
        #     staircase=True)
        class Args:
            def __init__(self) -> None:
                self.warmup_epochs = 10
                self.train_batch_size = 512
                self.learning_rate_scaling = 'linear'
                self.train_epochs = 100
                self.train_steps = None

        lr_schedule = WarmUpAndCosineDecay(.2, len(dataset.labels), Args())

        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            #取得模型架構
            myNet = None
            if not noPreProcess:
                myNet = self.net(input_shape=(*dataset.imgSize,3) ,classes=len(dataset.labels[0]) ,image_preProcess = DistortImage(self.epoch) ,verMark = verMark)
            else:
                myNet = self.net(input_shape=(*dataset.imgSize,3) ,classes=len(dataset.labels[0]),verMark = verMark)
            
            myNet.model.compile(
                #learning_rate=0.01
                optimizer= tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                momentum=0.9),
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
dataset.batchSize = 512
dataset.imgSize = (300 ,300)
dataset = dataset.tocategorical().Done()

train = Train(EfficientNetV2_S)
train.epoch = 30
train.batchSize = dataset.batchSize

myVerMark = verMark()
myVerMark.badge = 'withImgScaleLayer'
train.process(dataset ,verMark=myVerMark )