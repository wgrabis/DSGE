import numpy as np


class EstimationData:
    def __init__(self, data, observable_len, observable_names):
        self.estimation_time = len(data)
        self.measurements = data
        self.observable_len = observable_len
        self.observable_names = observable_names

    def __getitem__(self, key):
        return np.array(self.measurements[key], dtype='float')

    def size(self):
        # len(self.measurements[0])
        return self.observable_len, len(self.measurements)
