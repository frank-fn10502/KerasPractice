import datetime
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from pathlib import Path
import json
import numpy as np
import sys

from custom.utils.utils import VersionMark 

class ModelOuputHelper:
    '''
    處理神經網路模型的各種資料輸出，包含圖形、模型和文檔
    '''

    def __init__(self, model, verMark: VersionMark, datasetName='None', main_directory=None) -> None:
        if(model == None):
            raise Exception("please check model")

        self.model = model

        self.main_directory = Path(main_directory or f'result') / model.name / Path(*verMark.getMarkList())

        
        self.model_architecture_dir = self.main_directory 
        self.train_result_dir = self.main_directory / datasetName / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.model_architecture_dir.mkdir(parents=True, exist_ok=True)
        self.train_result_dir.mkdir(parents=True, exist_ok=True)

    def saveTrainProcessImg(self, history=None) -> None:
        '''
        將訓練過程用 matplotlib.pyplot 畫成圖表
        :param history  傳入 model.fit() 的回傳值
        '''
        modelHistory = history
        history = history.history
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

        plt.savefig( (self.train_result_dir/'train-progress.jpg').__str__() )
        plt.show()

        print('drawTrainProcess... Done')

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
        
    def saveTrainHistory(self ,history):
        '''
        儲存 train 產生的 history 以備不時之需
        '''
        modelHistory = history
        history = history.history
        path = self.train_result_dir / 'trainHistory.json'
        with path.open('w') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
        
        print('saveTrainHistory... Done')

    def saveModel(self):
        '''
        儲存 model 到預設位置(目前是儲存所有的資料)
        '''
        self.model.save(self.train_result_dir.__str__())

        print('saveModel... Done')
    
    def seveModelArchitecture(self) -> None:
        '''
        儲存 model:
            1. 文字 summary
            2. 圖片 keras.utils.plot_model(simple & complete)
        '''
        self.__drawModelImg()
        self.__saveModelTxT()

    def __drawModelImg(self):
        '''
        使用 keras.utils.plot_model 畫出模型架構圖
        '''
        keras.utils.plot_model(
            self.model,
            to_file=(self.model_architecture_dir / 'simple-model-architecture.png').__str__(),
            show_shapes=False,
        )
        keras.utils.plot_model(
            self.model,
            to_file=(self.model_architecture_dir / 'complete-model-architecture.png').__str__(),
            show_shapes=True,
        )
        print('saveModelImg... Done')
    
    def __saveModelTxT(self):
        '''
        儲存 model.summary()
        '''
        path = self.model_architecture_dir / 'model-architecture.txt'
        with path.open('w') as f:
            self.model.summary(print_fn=lambda x: print(x, file=f))

        print('saveModelTxT... Done')

