from numpy import dot, linalg
import numpy as np
from math import log, pi, exp
import logging

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution

logger = logging.getLogger(__name__)


class NewKalmanFilter(Filter):
    def predict(self, prev_distribution, transition, shock_matrix, shock_covariance, noise_covariance):
        x_k, p_k = prev_distribution

        x_next = dot(transition, x_k)
        p_next = dot(dot(transition, p_k), transition.T) + shock_matrix @ shock_covariance @ shock_matrix.transpose()

        logger.debug("kalman-predict-from")
        logger.debug(x_k)
        # logger.debug(p_k)
        logger.debug("kalman-predict-to")
        logger.debug(x_next)
        # logger.debug(p_next)

        assert x_next.shape == x_k.shape
        assert p_k.shape == p_next.shape

        return NormalVectorDistribution(x_next, p_next)

    def update(self, transition, time, prev_distribution, measurement_matrix, measurement_function, measurement_noise_covariance,
               measurement_vector):
        x_k, p_k = prev_distribution

        y = measurement_vector
        y_hat = measurement_function(x_k, time)

        assert y_hat.shape == measurement_vector.shape

        y_diff = y - y_hat
        y_cov = dot(dot(measurement_matrix, p_k), measurement_matrix.transpose()) + measurement_noise_covariance

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

        kalman_gain = dot(transition, dot(p_k, measurement_matrix.transpose()))

        # x_updated_k = x_k + dot(kalman_gain, iFtnut)

        x_updated_k = x_k + dot(kalman_gain, y_diff)
        # p_updated_k = p_k - kalman_gain @ np.linalg.solve(y_cov, kalman_gain.transpose())
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