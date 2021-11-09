import scipy.io
path = 'dataset/flowers/imagelabels.mat'

labels = scipy.io.loadmat(path)

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from custom.layer import DistortImage
import numpy as np

# 需要自行處理 label 的 one-hot encode
d = tf.keras.preprocessing.image_dataset_from_directory(directory='dataset/flowers/' ,labels=labels['labels'][0].tolist()
            ,batch_size=1 ,label_mode='int',image_size=(128, 128) ,validation_split = .2 ,subset = 'training' ,shuffle = False ,interpolation="nearest",crop_to_aspect_ratio = True)#labels['labels'][0] 

# print(list(d.take(1).as_numpy_iterator())[0][0])

# one_data = list(d.skip(500).take(1).as_numpy_iterator())
# one_data
# d.take(1)
x = DistortImage()
x.train = True
x.magnitude = 15
# for datas ,labels in d.take(1):
#     for data ,label in zip(datas ,labels):
#         img = Image.fromarray(data.numpy(), 'RGB')
#         plt.imshow(img)


        # print(label)
for datas ,labels in d.take(1):
    for data ,label in zip(datas ,labels):
        data = x.testFunc([data])[0]

        img = Image.fromarray(np.asarray(data), 'RGB')
        plt.imshow(img)
        # print(data)

        # print(label)
