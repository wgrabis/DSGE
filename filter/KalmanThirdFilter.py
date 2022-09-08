from numpy import dot, linalg
import numpy as np
from math import log, pi, exp
import logging

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution

logger = logging.getLogger(__name__)


class KalmanThirdFilter(Filter):
    def __init__(self, transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function,
                 measurement_noise_covariance):
        super().__init__(transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function,
                         measurement_noise_covariance)

    def filter(self, time, prev_distribution, y_t):
        x_prev, p_prev = prev_distribution
        # x_temp = dot(self.transition, x_prev)

        # logger.debug("kalman-predict-from")
        # logger.debug(x_prev)
        # # logger.debug(p_k)
        # logger.debug("kalman-predict-to")
        # logger.debug(x_next)
        #
        # assert x_next.shape == x_prev.shape

        y_hat = self.measurement_function(x_prev, time)

        assert y_hat.shape == y_t.shape

        y_diff = y_t - y_hat
        y_cov = dot(dot(self.measurement_matrix, p_prev), self.measurement_matrix.transpose()) + \
              self.measurement_noise_covariance

        pre_dft = np.linalg.det(y_cov)
        dFt = np.log(pre_dft)

        iFtnut = np.linalg.inv(y_cov)

        kalman_gain = dot(self.transition, dot(p_prev, self.measurement_matrix.transpose())) @ iFtnut

        x_next = self.transition @ x_prev + kalman_gain @ y_diff

        p_transition = (self.transition - kalman_gain @ self.measurement_matrix)

        rqr = self.shock_matrix @ self.shock_covariance @ self.shock_matrix.transpose()

        p_next = p_transition @ p_prev @ p_transition.transpose() \
                 + kalman_gain @ self.measurement_noise_covariance @ kalman_gain.transpose() + rqr

        assert x_next.shape == x_prev.shape, "Wrong shape of updated vector"
        assert p_next.shape == p_prev.shape

        likelihood = - 0.5 * dFt
        likelihood -= 0.5 * y_hat.shape[0] * np.log(2 * np.pi)
        likelihood -= 0.5 * y_diff @ iFtnut @ y_diff.transpose()

        logger.debug("Likelihood-status")
        # logger.debug(pre_dft)
        # logger.debug(y_cov)
        logger.debug(dFt)
        logger.debug(y_diff @ iFtnut @ y_diff.transpose())
        logger.debug(y_hat.shape[0])
        logger.debug("Diffrence of value, probability:")
        logger.debug(y_diff)
        logger.debug(likelihood)
        logger.debug("Y_HAT, Y:")
        logger.debug(y_hat)
        logger.debug(y_t)
        logger.debug("X_PREV, X_NEXT")
        logger.debug(x_prev)
        logger.debug(x_next)
        logger.debug("P_PREV, P_NEXT")
        logger.debug(p_prev)
        logger.debug(p_next)
        logger.debug("P_DIFF")
        logger.debug(x_prev - x_next)
        logger.debug(p_prev - p_next)

        return NormalVectorDistribution(x_next, p_next), likelihood
