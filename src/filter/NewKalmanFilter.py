from numpy import dot, linalg
import numpy as np
from math import log, pi, exp
import logging

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution

logger = logging.getLogger(__name__)


class NewKalmanFilter(Filter):
    def filter(self, time, prev_distribution, measurement_vector):
        x_prev, p_prev = prev_distribution

        x_k = dot(self.transition, x_prev)
        p_k = dot(dot(self.transition, p_prev), self.transition.T) \
                 + self.shock_matrix @ self.shock_covariance @ self.shock_matrix.transpose()

        logger.debug("kalman-predict-from")
        logger.debug(x_prev)
        # logger.debug(p_k)
        logger.debug("kalman-predict-to")
        logger.debug(x_k)
        # logger.debug(p_next)

        assert x_prev.shape == x_k.shape
        assert p_prev.shape == p_k.shape

        y = measurement_vector
        y_hat = self.measurement_function(x_k, time)

        assert y_hat.shape == measurement_vector.shape

        y_diff = y - y_hat
        y_cov = dot(dot(self.measurement_matrix, p_k), self.measurement_matrix.transpose()) \
                + self.measurement_noise_covariance

        y_cov = 0.5 * (y_cov.transpose() + y_cov)

        logger.debug("Measurement")
        logger.debug(y)
        logger.debug(y_hat)
        logger.debug("Diff")
        logger.debug(y_diff)
        logger.debug(y_cov)

        pre_dft = np.linalg.det(y_cov)
        dFt = np.log(abs(pre_dft))

        # iFtnut = np.linalg.solve(y_cov, y_diff)
        iFtnut = np.linalg.inv(y_cov)

        kalman_gain = dot(self.transition, dot(p_k, self.measurement_matrix.transpose()))

        # x_updated_k = x_k + dot(kalman_gain, iFtnut)

        x_updated_k = x_k + dot(kalman_gain, y_diff)
        p_updated_k = p_k - kalman_gain @ np.linalg.solve(y_cov, kalman_gain.transpose())
        # p_updated_k =

        assert x_updated_k.shape == x_k.shape, "Wrong shape of updated vector"
        assert p_k.shape == p_updated_k.shape

        likelihood = - 0.5 * dFt
        likelihood -= 0.5 * y_hat.shape[0] * np.log(2 * np.pi)
        likelihood -= 0.5 * np.dot(y_diff, iFtnut)

        logger.debug("Likelihood-status")
        logger.debug(pre_dft)
        logger.debug(y_cov)
        logger.debug(dFt)
        logger.debug(np.dot(y_diff, iFtnut))
        logger.debug(y_hat.shape[0])
        logger.debug("Y_HAT, Y:")
        logger.debug(y_hat)
        logger.debug(y)
        logger.debug("Diffrence of value, probability:")
        logger.debug(y_diff)
        logger.debug(likelihood)

        return NormalVectorDistribution(x_updated_k, p_updated_k), likelihood