from abc import ABC, abstractmethod
import scipy.stats as stats
from numpy import dot, linalg
import numpy as np
from math import log, pi, exp


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
    def __init__(self, mean, variance):
        self.distribution = stats.norm(mean, variance)
        self.mean = mean
        self.variance = variance

    def probability_of(self, value):
        print("probability_of")
        print(value)
        print(self.distribution)
        print(self.mean)
        print(self.variance)
        print(self.distribution.pdf(value))
        return self.distribution.pdf(value)

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.variance


class NormalVectorDistribution(Distribution):
    def __init__(self, mean_vector, covariance_matrix):
        print("NormalVector - distribution")
        print(mean_vector)
        print(covariance_matrix)
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

        # self.distributions = []
        # for i in range(len(mean_vector)):
        #     self.distributions.append(NormalDistribution(mean_vector[i], variance_vector[i]))

    def probability_of(self, value):
        residual = value - self.mean_vector

        probability = 0.5 * dot(residual.T, dot(linalg.inv(self.covariance_matrix), residual))
        probability += 0.5 * residual.shape[0] * log(2 * np.pi)
        probability += 0.5 * log(linalg.det(self.covariance_matrix))

        # print("probability_of[]start")
        # print(value)
        #
        # for i in range(len(self.distributions)):
        #     probability *= self.distributions[i].probability_of(value[i])
        #
        # print("probability_of[]")
        # print(probability)

        return probability

    def get_mean(self):
        return np.array(self.mean_vector)

    def get_covariance(self):
        return np.array(self.covariance_matrix)

    def get_vectors(self):
        return self.get_mean(), self.get_covariance()




