from abc import ABC, abstractmethod
import scipy.stats as stats
import numpy as np
from numpy import dot, linalg


class Distribution(ABC):
    @abstractmethod
    def probability_of(self, value):
        pass

    @abstractmethod
    def get_mean(self):
        pass

    @abstractmethod
    def get_covariance(self):
        pass


class NormalDistribution(Distribution):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def probability_of(self, value):
        print("probability_of")
        # todo fix
        return self.distribution.pdf(value)

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.covariance


class NormalVectorDistribution(Distribution):
    def __init__(self, mean_vector, covariance):
        print("NormalVector - distribution")
        print(mean_vector)
        # print(variance_vector)
        # self.distributions = []
        self.covariance = covariance
        self.mean = mean_vector

        # for i in range(len(mean_vector)):
        #     self.distributions.append(NormalDistribution(mean_vector[i], variance_vector[i]))

    # log value
    def probability_of(self, value):
        # probability = 1.0

        print("probability_of[]start")
        print(value)

        # for i in range(len(self.distributions)):
        #     probability *= self.distributions[i].probability_of(value[i])

        residual = value - self.mean

        probability = 0.5 * dot(residual.T, dot(linalg.inv(self.covariance), residual))
        probability += 0.5 * value.shape[0] * np.log(2 * np.pi)
        probability += 0.5 * np.log(linalg.det(self.covariance))

        print(probability)

        return probability

    def get_mean(self):
        return np.array(self.mean)

    def get_covariance(self):
        return np.array(self.covariance)

    def get_vectors(self):
        return np.array(self.get_mean()), np.array(self.get_covariance())




