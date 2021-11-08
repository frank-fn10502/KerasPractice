


class Dataset:
    def __init__(self , info: bool = False) -> None:
        self.isTfDataset = False
        self.info = info
        self.className = type(self).__name__
        if(info):
            print(f"dataset: {self.__doc__} \n")