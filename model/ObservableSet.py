

class ObservableSet:
    def __init__(self, data):
        self.data = data.sort()

    def __getitem__(self, key):
        lower_value = self.data



