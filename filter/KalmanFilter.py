from numpy import dot, linalg
import numpy as np
from math import log, pi, exp

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution


class KalmanFilter(Filter):
    def predict(self, prev_distribution, transition, control_input_matrix, control_input, noise_covariance):
        # x(k) = Fx(k-1) + Bu(k-1)
        # P(k_ = FP(k-1)Ft + Q
        x_k, p_k = prev_distribution

        x_next = dot(transition, x_k) + dot(control_input_matrix, control_input)
        p_next = dot(dot(transition, p_k), transition.T) + noise_covariance

        assert x_next.shape == x_k.shape
        assert p_k.shape == p_next.shape

        return NormalVectorDistribution(x_next, p_next)

    def update(self, time, prev_distribution,
               measurement_matrix, measurement_function,
               measurement_noise_covariance,
               measurement_vector):
        # y(k) = z(k) - Hx(k) - measurement residual
        # S(k) = HP(k)Ht - measurement prediction covariance
        # K (k) = P(k)*Ht*S(k)^(-1) - kalman gain
        # x_u(k) = x(k) + K(k)y(k) - updated mean
        # P_u(k) = (I - K(k)H)P(k) - updated covariance
        x_k, p_k = prev_distribution

        measurement_predict = measurement_function(x_k, time)

        print("Measurement")
        print(x_k)
        print(measurement_predict)
        print(measurement_vector)
        assert measurement_predict.shape == measurement_vector.shape

        measurement_residual = measurement_vector - measurement_predict
        measurement_prediction_covariance = dot(dot(measurement_matrix, p_k), measurement_matrix.transpose()) + \
                                            measurement_noise_covariance

        # todo if matrix is singular

        kalman_gain = dot(p_k, dot(measurement_matrix.transpose(), linalg.inv(measurement_prediction_covariance)))

        x_updated_k = x_k + dot(kalman_gain, measurement_residual)
        p_updated_k = p_k - dot(kalman_gain, dot(measurement_matrix, p_k))

        assert x_updated_k.shape == x_k.shape
        assert p_k.shape == p_updated_k.shape

        # todo rework into separate object

        likelihood = 0.5 * dot(measurement_residual.T,
                               dot(linalg.inv(measurement_prediction_covariance), measurement_residual))

        likelihood += 0.5 * measurement_predict.shape[0] * log(2 * np.pi)

        print(linalg.det)

        # todo rework
        if linalg.det(measurement_prediction_covariance) < 0:
            print("Likelihood-error")
            print(measurement_prediction_covariance)
            print(linalg.det(measurement_prediction_covariance))
        else:
            likelihood += 0.5 * log(linalg.det(measurement_prediction_covariance))

        print(likelihood)

        return NormalVectorDistribution(x_updated_k, p_updated_k), likelihood
