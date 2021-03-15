class LikelihoodSequence:
    def __init__(self):
        self.distributions = []
        self.measurements = []

    def insert_distribution(self, distribution):
        self.distributions.append(distribution)

    def insert_measurement(self, measurement, updated_distribution):
        self.measurements.append(measurement)

    def count_previous_probability(self):
        pass

    def count_next_probability(self, measurement):
        pass

    def last_distribution(self):
        return self.distributions[-1]
