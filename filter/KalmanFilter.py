from numpy import dot, linalg

from filter.Filter import Filter
from model.Distribution import NormalVectorDistribution
from model.Wrappers import TDistribution


# todo rework to distributions
class KalmanFilter(Filter):
    def predict(self, prev_distribution, transition, control_input_matrix, control_input, noise_covariance):
        # x(k) = Fx(k-1) + Bu(k-1)
        # P(k_ = FP(k-1)Ft + Q
        x_k, p_k = prev_distribution

        x_k = dot(transition, x_k) + dot(control_input_matrix, control_input)
        p_next = dot(dot(transition, p_k), transition.T) + noise_covariance

        return NormalVectorDistribution(x_k, p_next)

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

        measurement_residual = measurement_vector - measurement_predict
        measurement_prediction_covariance = dot(dot(measurement_matrix, p_k), measurement_matrix.transpose()) + \
            measurement_noise_covariance

        kalman_gain = dot(p_k, dot(measurement_matrix.transpose(), linalg.inv(measurement_prediction_covariance)))

        print("Update:")
        print(kalman_gain)
        print(measurement_residual)
        print(x_k)

        x_updated_k = x_k + dot(kalman_gain, measurement_residual)
        p_updated_k = p_k - dot(kalman_gain, dot(measurement_matrix, p_k))

        return NormalVectorDistribution(x_updated_k, p_updated_k)

