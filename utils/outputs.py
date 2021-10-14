import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import datetime
import numpy as np
import sys

class ModelOuputHelper:
    '''
    處理神經網路模型的各種資料輸出，包含圖形、模型和文檔
    '''

    def __init__(self, model=None, datasetName='None' ,preStr :str = None) -> None:
        if(model == None):
            raise Exception("please check model")
        self.model = model
        self.preStr = preStr

        self.main_directory = f"./result/{model.name}"
        if not os.path.exists(self.main_directory):
            os.makedirs(self.main_directory)

        self.save_train_result_dir = f"{self.main_directory}{'' if datasetName == '' else '/' + datasetName}/{self.preStr + '/'  if self.preStr != None else ''}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not os.path.exists(self.save_train_result_dir):
            os.makedirs(self.save_train_result_dir)

        self.model_image_dir = f"{self.save_train_result_dir}/model-images"
        if not os.path.exists(self.model_image_dir):
            os.makedirs(self.model_image_dir)

        self.save_model_dir = f"{self.save_train_result_dir}/model"
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

    def drawTrainProcess(self, history=None) -> None:
        '''
        將訓練過程用 matplotlib.pyplot 畫成圖表
        :param history  傳入 model.fit() 的回傳值
        '''
        if(history == None):
            return
        plt.figure(figsize = (15,5))
        
        
        self.__pltOnePlot('loss' ,(1,2,1),
        [
            [history['loss'] ,'-'],
            [history['val_loss'] ,'--'],
        ],loc_mini='upper right')
        self.__pltOnePlot('accuracy' ,(1,2,2),
        [
            [history['accuracy'] ,'-'],
            [history['val_accuracy'] ,'--'],
        ])

        plt.savefig(f'{self.save_train_result_dir}/train-progress.jpg')
        plt.show()

        print('drawTrainProcess... Done')

        self.__saveTrainHistory(history)

        print('saveTrainHistory... Done')

    def __pltOnePlot(self,title, pos, plotDatas: list,loc_mini:str = 'upper left'):
        '''
        pos: 
            ex: (1,2,1)

        plotDatas:
            ex:
            [
                [[...] ,'--'],
                [[...] ,'-'],
            ]

        loc_mini: 'upper left' or 'upper right'
        '''

        plt.subplot(*pos)
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5')
        xticks_start = 0
        xticks_end = 0
        yticks_start = sys.maxsize
        yticks_end = 0
        
        for datas ,sign in plotDatas:
            xticks_end = max(xticks_end ,len(datas))
            yticks_end = max(max(datas) ,yticks_end)
            yticks_start = min(min(datas) ,yticks_start)

            plt.plot(datas ,sign)
        
        plt.legend(['train', 'test'], loc=loc_mini)
        plt.xlim([xticks_start,xticks_end])
        plt.ylim([yticks_start,yticks_end])
        
        x_range = max(10 ,(xticks_start + xticks_end) / 20)
        x_tick_list = np.arange(xticks_start ,xticks_end ,x_range)
        x_tick_list = np.append(x_tick_list ,xticks_end)
        plt.xticks(x_tick_list,rotation=90)

        y_range = (yticks_start + yticks_end) / 10
        y_tick_list = np.arange(yticks_start ,yticks_end - y_range ,y_range)
        y_tick_list = np.append(y_tick_list ,yticks_end)
        plt.yticks( y_tick_list )
        
    def drawModelImg(self) -> None:
        '''
        使用 keras.utils.plot_model 畫出模型架構圖
        '''
        keras.utils.plot_model(
            self.model,
            to_file=f"{self.model_image_dir}/simple-model.png",
            show_shapes=False,
        )
        keras.utils.plot_model(
            self.model,
            to_file=f"{self.model_image_dir}/complete-model.png",
            show_shapes=True,
        )
        print('drawModelImg... Done')

    def saveModel(self):
        '''
        儲存 model 到預設位置(目前是儲存所有的資料)
        '''
        self.model.save(f"{self.save_model_dir}")

        print('saveModel... Done')

    def saveModelTxT(self):
        path = f'{self.save_train_result_dir}/model-arc.txt'
        with open(path,'w') as f:
            self.model.summary(print_fn=lambda x: print(x, file=f))

        print('saveModelTxT... Done')

    def __saveTrainHistory(self ,history:dict):
        path = f'{self.save_train_result_dir}/trainHistory.txt'
        with open(path,'w') as f:
            for k ,v in history.items():
                print(f'{k}: {v} \n\r', file=f)
