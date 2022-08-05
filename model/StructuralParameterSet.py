import logging

from model.Distribution import Distribution, NormalVectorDistribution
import numpy as np

logger = logging.getLogger(__name__)


class StructuralParameterSet:
    def __init__(self, calibrated_params, random_params,  calibrated_vector, mean_vector, covariance_matrix):
        self.calibrated_vector = calibrated_vector
        self.ordered_params = calibrated_params + random_params
        self.calibrated_len = len(calibrated_params)
        self.random_len = len(random_params)

        self.random_distribution = NormalVectorDistribution(mean_vector, covariance_matrix)

    def probability_of(self, value):
        random_vector_part = value[self.calibrated_len:]

        return self.random_distribution.probability_of(random_vector_part)

    def get_mean(self):
        return self.random_distribution.get_mean()

    def get_covariance(self):
        return self.random_distribution.get_covariance()

    def get_prior_vector(self):
        return self.calibrated_vector + self.random_distribution.get_mean()

    def get_posterior_vector(self, random_params_value):
        return self.calibrated_vector + random_params_value
