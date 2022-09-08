from abc import ABC, abstractmethod
import scipy.stats as stats
from numpy import dot, linalg
import numpy as np
from math import log, pi, exp
from numpy import dot, linalg

import logging

logger = logging.getLogger(__name__)


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
        pass
        # return self.distribution.pdf(value)

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.covariance


class NormalVectorDistribution(Distribution):
    def __init__(self, mean_vector, covariance_matrix):
        # logger.debug("NormalVector - distribution")
        # logger.debug(mean_vector)
        # logger.debug(covariance_matrix)
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    # log value
    def probability_of(self, value):
        residual = value - self.mean_vector

        ft_diff = np.linalg.solve(self.covariance_matrix, residual)

        probability = - 0.5 * np.dot(residual, ft_diff)
        # probability = 0.5 * dot(residual.T, dot(linalg.inv(self.covariance_matrix), residual))

        # probability += 0.5 * residual.shape[0] * np.log(2 * np.pi)
        # probability += 0.5 * log(linalg.det(self.covariance_matrix))

        return probability

    def get_mean(self):
        return np.array(self.mean_vector)

    def get_covariance(self):
        return np.array(self.covariance_matrix)

    def get_vectors(self):
        return self.get_mean(), self.get_covariance()




