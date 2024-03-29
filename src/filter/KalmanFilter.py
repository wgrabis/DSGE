import logging

from numpy import dot, linalg
import numpy as np
from math import log, pi, exp

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution

logger = logging.getLogger(__name__)


class KalmanFilter_Old(Filter):
    def filter(self, time, prev_distribution, measurement_vector):
        # y(k) = z(k) - Hx(k) - measurement residual
        # S(k) = HP(k)Ht - measurement prediction covariance
        # K (k) = P(k)*Ht*S(k)^(-1) - kalman gain
        # x_u(k) = x(k) + K(k)y(k) - updated mean
        # P_u(k) = (I - K(k)H)P(k) - updated covariance

        # x(k) = Fx(k-1) + Bu(k-1)
        # P(k_ = FP(k-1)Ft + Q
        x_prev, p_prev = prev_distribution

        logger.debug("kalman-predict")
        logger.debug(x_prev)
        logger.debug(p_prev)

        x_k = dot(self.transition, x_prev)

        p_k = dot(dot(self.transition, p_prev), self.transition.T) \
              + self.shock_matrix @ self.shock_covariance @ self.shock_matrix.transpose()

        logger.debug("kalman-next")
        logger.debug(x_k)
        logger.debug(p_k)

        assert x_prev.shape == x_k.shape
        assert p_prev.shape == p_k.shape

        measurement_predict = self.measurement_function(x_k, time)

        logger.debug("Measurement")
        logger.debug(measurement_predict)
        logger.debug(measurement_vector)
        assert measurement_predict.shape == measurement_vector.shape

        measurement_residual = measurement_vector - measurement_predict
        measurement_prediction_covariance = dot(dot(self.measurement_matrix, p_k), self.measurement_matrix.transpose()) + \
                                            self.measurement_noise_covariance

        # todo if matrix is singular
        measurement_prediction_covariance = \
            0.5 * (measurement_prediction_covariance.transpose() + measurement_prediction_covariance)

        logger.debug(measurement_prediction_covariance)

        p_k_tt = p_k

        kalman_gain = dot(p_k, dot(self.measurement_matrix.transpose(), linalg.inv(measurement_prediction_covariance)))

        x_updated_k = x_k + dot(kalman_gain, measurement_residual)
        p_updated_k = p_k - dot(kalman_gain, dot(self.measurement_matrix, p_k))

        assert x_updated_k.shape == x_k.shape
        assert p_k.shape == p_updated_k.shape

        likelihood = 0.5 * np.log(linalg.det(measurement_prediction_covariance))

        likelihood += 0.5 * np.dot(measurement_residual, linalg.solve(measurement_prediction_covariance, measurement_residual))

        # ok
        likelihood += 0.5 * measurement_predict.shape[0] * np.log(2 * np.pi)

        logger.debug("Likelihood-status")
        logger.debug(measurement_prediction_covariance)
        logger.debug(linalg.det(measurement_prediction_covariance))

        logger.debug(likelihood)

        return NormalVectorDistribution(x_updated_k, p_updated_k), -likelihood
