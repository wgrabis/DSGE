import logging

from model.Distribution import Distribution, NormalVectorDistribution
import numpy as np

from model.PosteriorVector import PosteriorVector

logger = logging.getLogger(__name__)


class StructuralParameterSet:
    def __init__(self, calibrated_params, random_params, calibrated_vector, mean_vector, covariance_matrix, name_map):
        self.calibrated_vector = np.array(calibrated_vector).astype(np.float64)
        self.ordered_params = calibrated_params + [x.name for x in random_params]
        self.calibrated_len = len(calibrated_params)
        self.random_len = len(random_params)
        self.random_params = random_params
        self.name_map = name_map

        self.random_distribution = NormalVectorDistribution(mean_vector, covariance_matrix)

    def probability_of(self, value):
        random_vector_part = value[self.calibrated_len:]

        return self.random_distribution.probability_of(random_vector_part)

    def get_mean(self):
        return self.random_distribution.get_mean()

    def get_param_covariance(self):
        unbound_covariance = np.zeros((self.random_len, self.random_len))
        covariance = self.random_distribution.get_covariance()

        for i in range(self.random_len):
            param = self.random_params[i]
            for j in range(self.random_len):
                unbound_covariance[i, j] = param.scale_variable(covariance[i, j], True)

        logger.debug(unbound_covariance)
        logger.debug(covariance)

        return covariance

    def scale_seed(self, pre_vars):
        result = np.zeros(self.random_len)
        for i in range(self.random_len):
            param = self.random_params[i]
            value = pre_vars[i]

            scaled_value = param.scale_variable(value)
            result[i] = scaled_value

        return result

    def rescale_seed(self, post_seed):
        result = np.zeros(self.random_len)
        for i in range(self.random_len):
            param = self.random_params[i]
            seed = post_seed[i]

            scaled_value = param.rescale_variable(seed)
            result[i] = scaled_value

        return result

    def get_prior_vector(self):
        # result = np.zeros(self.calibrated_len + self.random_len)
        # result[:self.calibrated_len] = self.calibrated_vector
        # result[self.calibrated_len:] = self.random_distribution.get_mean()

        random_mean = self.random_distribution.get_mean()
        unbound_seed = self.scale_seed(random_mean)

        return PosteriorVector(unbound_seed, self, self.random_distribution.get_mean())

    # def get_random_part(self, posterior):
    #     return posterior[self.calibrated_len:]

    def get_move_vector(self, unbound_seed):
        return PosteriorVector(unbound_seed, self, self.rescale_seed(unbound_seed))

    # accepts seed in scaled version not unbound seed
    def get_bound_posterior_vector(self, random_params_scaled):
        result = np.zeros(self.calibrated_len + self.random_len)

        result[:self.calibrated_len] = self.calibrated_vector
        result[self.calibrated_len:] = random_params_scaled

        return result

    # def check_bounds(self, random_params_value):
    #     for i in range(self.random_len):
    #         param = self.random_params[i]
    #         value = random_params_value[i]
    #         if not param.check_value(value):
    #             return False
    #
    #     return True
