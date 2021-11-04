from interface.Dataset import Dataset
import tensorflow as tf
import scipy.io


class FileDataset(Dataset):
    def __init__(self ,info: bool = False) -> None:
        super().__init__(info)
        self.isTfDataset = True

    def Done(self):
        pass


class Flowers(FileDataset):
    '''
    ** 使用 102 個 class 的版本 **
    採用 flowers 資料集(https://www.robots.ox.ac.uk/~vgg/data/flowers/)
        1. Consisting of 102 different categories of flowers common to the UK
        2. Each class consists of between 40 and 258 images
    '''
    def __init__(self ,info: bool = False ,labelPath = '' ,imagePath = '') -> None:
        self.labelPath = labelPath
        self.imagePath = imagePath

        #目前下載的資料集是這樣處理
        self.labels = scipy.io.loadmat(self.labelPath)['labels'][0].tolist()
        self.labelMode = 'int'
        self.batchSize = 32
        self.imgSize = (256,256)
        self.seed = 2021
        self.split = .2

        #target
        self.train = None
        self.validation = None

        super().__init__(info)

    def tocategorical(self) -> 'FileDataset':
        '''
        label 轉換成 one-hot 編碼(train_y 和 test_y)
        '''
        pre_labels = self.labels[0]

        # self.labels = list( map(lambda x: [int(i) for i in x.tolist()] ,tf.keras.utils.to_categorical(self.labels)) )
        self.labels = list( map(lambda x: [int(i) for i in x ] ,tf.keras.utils.to_categorical(self.labels)) )
        # self.labelMode = 'categorical'

        if self.info:
            print("one-hot encoder:")
            print(f"\tindex: 0 ,pre: {pre_labels} ,after:{self.labels[0]}")

        return self

    def Done(self):
        # get dataset from dir
        d_train = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.imagePath,
            labels=self.labels,
            batch_size=self.batchSize,
            label_mode=self.labelMode,
            image_size=self.imgSize,
            # Set seed to ensure the same split when loading testing data.
            seed=self.seed,
            validation_split=self.split,
            subset='training',
            shuffle=True,
            interpolation="nearest",
            crop_to_aspect_ratio=True)
            
        d_validation = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.imagePath,
            labels=self.labels,
            batch_size=self.batchSize,
            label_mode=self.labelMode,
            image_size=self.imgSize,
            # Set seed to ensure the same split when loading testing data.
            seed=self.seed,
            validation_split=self.split,
            subset='validation',
            shuffle=True,
            interpolation="nearest",
            crop_to_aspect_ratio=True)

        self.train, self.validation = d_train, d_validation

        return self
