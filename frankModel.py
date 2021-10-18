import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.python.keras.layers.merge import add

from utils.outputs import ModelOuputHelper


class BasicModel:
    def __init__(self, datasetName="None", input_shape=(None, None, 3), classes=10 ,resize=(224, 224) ,main_directory = None) -> None:
        self.model = None
        self.outputHelper = None
        self.datasetName = datasetName
        self.input_shape = input_shape
        self.classes = classes
        self.main_directory = main_directory

        self.layer_scale = layers.Rescaling(scale=1./255)
        self.layer_resizing = layers.Resizing(*resize, interpolation='bilinear')

    def getModel(self): return self.model

    def __createOutputHelper(self ,preStr:str = None):
        '''
        需要先建立 model 再呼叫此方法
            self._BasicModel__createOutputHelper()
        
        '''
        if(self.model == None): raise Exception("please check model")

        self.outputHelper = ModelOuputHelper(self.model, self.datasetName ,preStr ,self.main_directory)
        self.outputHelper.drawModelImg()
        self.outputHelper.saveModelTxT()


class LeNet(BasicModel):
    def __init__(self, input_shape=(32, 32, 1), classes=10, datasetName='MNIST' ,main_directory = None) -> None:
        super().__init__(datasetName, input_shape, classes ,main_directory=main_directory)

        self.model = self.__build()
        self._BasicModel__createOutputHelper()

    def __build(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_shape)

        # 數值縮成 0 ~ 1 之間
        x = self.layer_scale(inputs)

        # ( [(32 - 5) / 1] + 1 ) * 6 = 28 * 6 ==> 輸出 28 * 28 * 6
        x = layers.Conv2D(kernel_size=(5, 5), filters=6,
                          strides=1, activation='tanh')(inputs)
        # ( [(28 - 2) / 2] + 1 ) * 6 = 14 * 6 ==> 輸出 14 * 14 * 6
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # ( [(14 - 5) / 1] + 1 ) * 16 = 10 * 16 ==> 輸出 10 * 10 * 16
        x = layers.Conv2D(kernel_size=(5, 5), filters=16,
                          strides=1, activation='tanh')(x)
        # ( [(10 - 2) / 2] + 1 ) * 16 = 5 * 16 ==> 輸出 5 * 5 * 16
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # 2D --> 1D
        x = layers.Flatten()(x)

        x = layers.Dense(120, activation='tanh')(x)
        x = layers.Dense(84, activation='tanh')(x)

        outputs = layers.Dense(self.classes, activation=activations.softmax)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="frank_LeNet")

        model.summary()

        return model


class AlexNet(BasicModel):
    def __init__(self, input_shape=(None, None, 3), classes=10, datasetName='MNIST' ,resize=(227, 227) ,main_directory = None) -> None:
        super().__init__(datasetName, input_shape, classes,resize ,main_directory=main_directory)

        self.model = self.__build()
        self._BasicModel__createOutputHelper()

    def __build(self):
        inputs = keras.Input(shape=self.input_shape)

        #preprocessing layer
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.__buildConv(x)
        outputs = self.__buildFC(x)


        model = keras.Model(inputs=inputs, outputs=outputs,name="frank_AlexNet")
        model.summary()

        return model

    def __buildConv(self, x):
        # conv1
        #   (227 - 11) / 4 + 1 --> 55 * 55 * 96
        x = layers.Conv2D(kernel_size=(11,11) ,filters = 96 ,strides=4 ,activation='relu')(x)

        #   (55 - 3) / 2 + 1  --> 27 * 27 * 96
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = layers.BatchNormalization()(x) #本來應該要用 LRN


        # conv2
        #   27 * 27 * 256
        x = layers.Conv2D(kernel_size=(5, 5), filters=256,
                          strides=2, padding="same", activation='relu')(x)

        #   (27 - 3) / 2 + 1 --> 13 * 13 * 256
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = layers.BatchNormalization()(x) #本來應該要用 LRN


        #conv3 - 4 - 5
        x = layers.Conv2D(kernel_size=(3, 3), filters=384,
                          strides=1, padding="same", activation='relu')(x)
        x = layers.Conv2D(kernel_size=(3, 3), filters=384,
                          strides=1, padding="same", activation='relu')(x)
        x = layers.Conv2D(kernel_size=(3, 3), filters=256,
                          strides=1, padding="same", activation='relu')(x)

        # (13 - 3) / 2 + 1 --> 6 * 6 * 256
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)

        return x

    def __buildFC(self, x):
        x = layers.Flatten()(x)
        # FC6
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # FC7
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # FC8
        # x = layers.Dense(1000)(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.classes, activation=activations.softmax)(x)

        return x


class VGG16(BasicModel):
    def __init__(self, input_shape=(None, None, 3), classes=10, datasetName='MNIST' ,flexImgSize = False ,main_directory = None) -> None:
        super().__init__(datasetName, input_shape, classes ,main_directory=main_directory)

        self.model = self.__build(flexImgSize)
        self._BasicModel__createOutputHelper(preStr='flexImgSize' if flexImgSize else 'fixedImgSize')

    def __build(self ,flexImgSize):
        inputs = keras.Input(shape=self.input_shape)        

       
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.__buildConv(x)

        if flexImgSize: x = self.__buildLastConv(x)
        else: x = self.__buildFC(x)
        

        outputs = layers.Dense(self.classes, activation=activations.softmax)(x)

        model = keras.Model(inputs=inputs, outputs=outputs , name="frank_VGG16")
        model.summary()

        return model

    def __buildConv(self, x):
        #conv1
        x = layers.Conv2D(kernel_size=(3, 3), filters=64, strides=1, padding="same", activation='relu')(x)
        #conv2
        x = layers.Conv2D(kernel_size=(3, 3), filters=64, strides=1, padding="same", activation='relu')(x)
        #(224 - 2) / 2 + 1 --> 112 * 112 * 64
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)

        #conv3
        x = layers.Conv2D(kernel_size=(3, 3), filters=128, strides=1, padding="same", activation='relu')(x)
        #conv4
        x = layers.Conv2D(kernel_size=(3, 3), filters=128, strides=1, padding="same", activation='relu')(x)
        #(112 - 2) / 2 + 1 --> 56 * 56 * 128
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)

        #conv5
        x = layers.Conv2D(kernel_size=(3, 3), filters=256, strides=1, padding="same", activation='relu')(x)
        #conv6
        x = layers.Conv2D(kernel_size=(3, 3), filters=256, strides=1, padding="same", activation='relu')(x)
        #conv7
        x = layers.Conv2D(kernel_size=(3, 3), filters=256, strides=1, padding="same", activation='relu')(x)
        #(56 - 2) / 2 + 1 --> 28 * 28 * 256
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)

        #conv8
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #conv9
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #conv10
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #(28 - 2) / 2 + 1 --> 14 * 14 * 512
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)

        #conv11
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #conv12
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #conv13
        x = layers.Conv2D(kernel_size=(3, 3), filters=512, strides=1, padding="same", activation='relu')(x)
        #(14 - 2) / 2 + 1 --> 7 * 7 * 512
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)

        return x

    def __buildFC(self, x):
        # FC1
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # FC2
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        #FC3
        x = layers.Dense(1000)(x)
        x = layers.Dropout(0.5)(x)

        return x

    def __buildLastConv(self ,x):
        # conv1
        x = layers.Conv2D(kernel_size=(7, 7), filters=4096, strides=1 )(x)
        x = layers.Dropout(0.5)(x)

        # conv2
        x = layers.Conv2D(kernel_size=(1, 1), filters=4096, strides=1 )(x)
        x = layers.Dropout(0.5)(x)

        # conv3
        x = layers.Conv2D(kernel_size=(1, 1), filters=1000, strides=1 )(x)
        x = layers.Dropout(0.5)(x)

        #https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
        x = layers.GlobalAveragePooling2D(data_format='channels_last')(x)

        return x


class InceptionV1(BasicModel):
    def __init__(self, input_shape=(None, None, 3), classes=10, datasetName='MNIST' ,main_directory = None) -> None:
        super().__init__(datasetName, input_shape, classes ,main_directory=main_directory)

        self.model = self.__build()
        self._BasicModel__createOutputHelper()   

    def __build(self):
        inputs = keras.Input(shape=self.input_shape)   
        
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)
        
        x = layers.Conv2D(kernel_size=7 ,strides=2 ,filters=64 ,activation='relu' ,padding='same')(x)
        #https://keras.io/api/layers/pooling_layers/max_pooling2d/
        x = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')(x)
        x = layers.Conv2D(kernel_size=3, strides=1, filters=192, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')(x)


        x = self.__Inception(x ,64 ,(96,128) ,(16 ,32) ,32)
        x = self.__Inception(x ,128 ,(128,192) ,(32 ,96) ,64)


        x = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')(x)


        x = self.__Inception(x ,192 ,(96,208) ,(16 ,48) ,64)
        x = self.__Inception(x ,160 ,(112,224) ,(24 ,64) ,64)
        x = self.__Inception(x ,128 ,(128,256) ,(24 ,64) ,64)
        x = self.__Inception(x ,112 ,(144,288) ,(32 ,64) ,64)
        x = self.__Inception(x ,256 ,(160,320) ,(32 ,128) ,128)


        x = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')(x)


        x = self.__Inception(x ,256 ,(160,320) ,(32 ,128) ,128)
        x = self.__Inception(x ,384 ,(192,384) ,(48 ,128) ,128)


        x = layers.GlobalAveragePooling2D(data_format='channels_last')(x)
        x = layers.Dropout(.4)(x)
    
        outputs = layers.Dense(self.classes, activation=activations.softmax)(x)

        model = keras.Model(inputs=inputs, outputs=outputs , name="frank_InceptionV1")
        model.summary()
        return model

    def __Inception(self, x, *depth):
        '''
        *depth: [conv1x1 ,(c3x3_r ,c3x3) ,(c5x5_r,c5x5) ,c1x1_pool]
        '''
        conv1x1 ,(c3x3_r ,c3x3) ,(c5x5_r,c5x5) ,c1x1_pool = depth

        b1x1 = layers.Conv2D(kernel_size=(1, 1), filters=conv1x1, padding="same", activation='relu')(x)

        b3x3 = layers.Conv2D(kernel_size=(1, 1), filters=c3x3_r, padding="same", activation='relu')(x)
        b3x3 = layers.Conv2D(kernel_size=(3, 3), filters=c3x3, padding="same", activation='relu')(b3x3)

        b5x5 = layers.Conv2D(kernel_size=(1, 1), filters=c5x5_r, padding="same", activation='relu')(x)
        b5x5 = layers.Conv2D(kernel_size=(5, 5), filters=c5x5, padding="same", activation='relu')(b5x5)

        pool = layers.MaxPool2D(pool_size=3 ,strides=1, padding='same')(x)
        pool = layers.Conv2D(kernel_size=(1, 1), filters=c1x1_pool, padding="same", activation='relu' )(pool)

        return layers.concatenate([b1x1, b3x3, b5x5 ,pool],axis=3)


class ResNet50(BasicModel):
    def __init__(self, input_shape=(None, None, 3), classes=10, datasetName='MNIST' ,main_directory = None) -> None:
        super().__init__(datasetName, input_shape, classes ,main_directory=main_directory)

        self.model = self.__build()
        self._BasicModel__createOutputHelper()  

    def __build(self):
        inputs = keras.Input(shape=self.input_shape)   
        
        x = self.layer_scale(inputs) 
        x = self.layer_resizing(x)

        x = layers.Conv2D(kernel_size=(7, 7), filters=64, strides=2, padding='same')(x)
        x = layers.MaxPool2D(pool_size=(3,3),strides=2 ,padding='same')(x)

        x = self.__residualBottleneckBlock(x, 64, 64, 256, changeShortcutChannel=True)
        for i in range(2): x = self.__residualBottleneckBlock(x ,64,64,256)

        x = self.__residualBottleneckBlock(x, 128, 128, 512, needDownSample=True)
        for i in range(3): x = self.__residualBottleneckBlock(x ,128, 128, 512)

        x = self.__residualBottleneckBlock(x, 256, 256, 1024, needDownSample=True)
        for i in range(5): x = self.__residualBottleneckBlock(x ,256, 256, 1024)

        x = self.__residualBottleneckBlock(x, 512, 512, 2048, needDownSample=True)
        for i in range(3): x = self.__residualBottleneckBlock(x ,512, 512, 2048)

        x = layers.GlobalAveragePooling2D()(x)


        outputs = layers.Dense(self.classes, activation=activations.softmax)(x)      

        model = keras.Model(inputs=inputs, outputs=outputs , name="frank_ResNet50")
        model.summary()
        return model

    def __residualBottleneckBlock(self, pre_x, *f, needDownSample=False, changeShortcutChannel=False):
        '''
        *f : conv1x1_filter ,conv3x3_filter ,conv1x1_filter
        '''
        (cf1 ,cf2 ,cf3) = f
        x = self.__BN_relu_conv(pre_x, kernel_size=(1,1) , filters=cf1 
                                     , stride=2 if needDownSample else 1)
        x = self.__BN_relu_conv(x,filters=cf2)
        x = self.__BN_relu_conv(x, kernel_size=(1,1) ,filters=cf3)

        if needDownSample or changeShortcutChannel:
            #filters 要使用最終的輸出才可相加
            pre_x = self.__BN_relu_conv(pre_x, kernel_size=(1,1), filters=cf3, 
                                        stride=2 if needDownSample else 1)

        return layers.Add()([pre_x ,x])

    def __BN_relu_conv(self, x, filters, kernel_size=(3, 3), stride=1):
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(kernel_size=kernel_size,
                          filters=filters, strides=stride ,padding='same')(x)

        return x
