import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np

from .interface.Dataset import Dataset


class SmallDataset(Dataset):
    '''
    最上層的 SmallDataset ，以下繼承都是使用 keras 內建再包裝
    '''
    def __init__(self, info: bool = False) -> None:
        super().__init__(info)

    def Done(self) -> 'SmallDataset':
        if self.info:
            print(
                f"train_x:{self.train_x.shape} \ntrain_y:{self.train_y.shape} \ntest_x:{self.test_x.shape} \ntest_y:{self.test_y.shape}")
        return self

    def addChannel(self) -> 'SmallDataset':
        return self

    def tocategorical(self) -> 'SmallDataset':
        '''
        label 轉換成 one-hot 編碼(train_y 和 test_y)
        '''
        pre_train_y = self.train_y[0]
        pre_test_y = self.test_y[0]

        self.train_y = tf.keras.utils.to_categorical(self.train_y)
        self.test_y = tf.keras.utils.to_categorical(self.test_y)

        if self.info:
            print("one-hot encoder:")
            print(f"\tindex: 0 ,pre: {pre_train_y} ,after:{self.train_y[0]}")
            print(
                f"\tindex: 0 ,pre: {pre_test_y} ,after:{self.test_y[0]}\n{'-'*10}")

        return self

    def resizing(self, w: int = 32, h: int = 32) -> 'SmallDataset':
        '''
        將圖片轉換成 w * h
        '''
        pre_train_x = self.train_x.shape
        pre_test_x = self.test_x.shape

        layer_resizing = keras.layers.Resizing(w, h, interpolation='bilinear')

        if(len(self.train_x.shape) != 4):
            self.addChannel()

        self.train_x = layer_resizing(self.train_x)
        self.test_x = layer_resizing(self.test_x)

        if self.info:
            print(
                f"resizing from {pre_train_x[1]}*{pre_train_x[2]} to {w}*{h}:")
            print(
                f"\tindex: 0 ,pre: {pre_train_x} ,after:{self.train_x.shape}")
            print(
                f"\tindex: 0 ,pre: {pre_test_x} ,after:{self.test_x.shape}\n{'-'*10}")

        return self


class MNIST(SmallDataset):
    '''
    使用 tensorflow.keras 取得的 MNIST 資料集(https://keras.io/api/datasets/mnist/) \n
    This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
    '''

    def __init__(self, info: bool = False) -> None:
        self.dataset = keras.datasets.mnist
        (self.train_x, self.train_y), (self.test_x,
                                       self.test_y) = self.dataset.load_data()
        super().__init__(info)

    def addChannel(self) -> 'MNIST':
        self.train_x = np.expand_dims(self.train_x, 3)
        self.test_x = np.expand_dims(self.test_x, 3)
        return self


class CIFAR10(SmallDataset):
    '''
    使用 tensorflow.keras 取得的 CIFAR10 資料集(https://keras.io/api/datasets/cifar10/) \n
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. \n
    There are 50000 training images and 10000 test images.
    '''

    def __init__(self, info: bool = False) -> None:
        self.dataset = keras.datasets.cifar10
        (self.train_x, self.train_y), (self.test_x,
                                       self.test_y) = self.dataset.load_data()
        super().__init__(info)


class CIFAR100(SmallDataset):
    '''
    使用 tensorflow.keras 取得的 CIFAR10 資料集(https://keras.io/api/datasets/cifar100/) \n
    This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. \n
    There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 super()classes. \n
    Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).\n
    EX: Superclass:(aquatic mammals) \n
        Classes: (beaver, dolphin, otter, seal, whale)
    '''

    def __init__(self, label_mode: str = "fine", info: bool = False) -> None:
        self.dataset = keras.datasets.cifar100
        (self.train_x, self.train_y), (self.test_x,
                                       self.test_y) = self.dataset.load_data(label_mode=label_mode)
        super().__init__(info)
