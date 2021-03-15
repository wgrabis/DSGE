class EstimationData:
    def __init__(self, data):
        self.estimation_time = len(data)
        self.measurements = data