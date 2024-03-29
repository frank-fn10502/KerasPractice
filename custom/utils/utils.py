class VersionMark:
    '''
    用來標註 
        1. model 的版本
        2. 同一 model 的不同變種
    可由呼叫者自行增加 attr
    '''
    def __init__(self) -> None:
        self.ver = '1'
        self.badge = None

    def getMarkList(self) -> list: 
        '''
        取得不為 None 的 attr list
        '''
        return list( map(lambda y : f'{y[0]}_{y[1]}' , filter(lambda x: x[1], self.__dict__.items()) ))
