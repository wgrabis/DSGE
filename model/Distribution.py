from abc import ABC, abstractmethod
import scipy.stats as stats
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def probability_of(self, value):
        pass

    @abstractmethod
    def get_mean(self):
        pass

    @abstractmethod
    def get_variance(self):
        pass


class NormalDistribution(Distribution):
    def __init__(self, mean, variance):
        self.distribution = stats.norm(mean, variance)
        self.mean = mean
        self.variance = variance

    def probability_of(self, value):
        return self.distribution.pdf(value)

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.variance


class NormalVectorDistribution(Distribution):
    def __init__(self, mean_vector, variance_vector):
        self.distributions = []
        for i in range(len(mean_vector)):
            self.distributions.append(NormalDistribution(mean_vector[i], variance_vector[i]))

    def probability_of(self, value):
        probability = 1

        for i in range(len(self.distributions)):
            probability *= self.distributions[i].probability_of(value[i])

        return probability

    def get_mean(self):
        mean = []

        for distribution in self.distributions:
            mean.append(distribution.get_mean())

        return mean

    def get_variance(self):
        variance = []

        for distribution in self.distributions:
            variance.append(distribution.get_variance())

        return variance

    def get_vectors(self):
        return np.array(self.get_mean()), np.array(self.get_variance())




