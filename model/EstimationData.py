import numpy as np


class EstimationData:
    def __init__(self, data):
        self.estimation_time = len(data)
        self.measurements = data

    def __getitem__(self, key):
        return np.array(self.measurements[key])
