from data import Dataset


class GenVideo(Dataset):

    def __init__(self, path: str):
        super().__init__(name='GenVideo')
        self.path = path


    def __iter__(self):
        pass