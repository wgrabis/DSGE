from numpy import dot, linalg

from filter.Filter import Filter


class KalmanFilter(Filter):
    def predict(self, x_k, p_k, transition, control_input_matrix, control_input, noise_covariance):
        # x(k) = Fx(k-1) + Bu(k-1)
        # P(k_ = FP(k-1)Ft + Q
        x_k = dot(transition, x_k) + dot(control_input_matrix, control_input)
        p_next = dot(dot(transition, p_k), transition.T) + noise_covariance

        return x_k, p_next

    def update(self, x_k, p_k, measurement_matrix, measurement_noise_covariance, measurement_vector):
        # y(k) = z(k) - Hx(k) - measurement residual
        # S(k) = HP(k)Ht - measurement prediction covariance
        # K(k) = O(k)*Ht*S(k)^(-1) - kalman gain
        # x_u(k) = x(k) + K(k)y(k) - updated mean
        # P_u(k) = (I - K(k)H)P(k) - updated covariance
        measurement_residual = measurement_vector - dot(measurement_matrix, x_k)
        measurement_prediction_covariance = dot(dot(measurement_matrix, p_k), measurement_matrix.transpose()) + \
            measurement_noise_covariance

        kalman_gain = dot(p_k, dot(measurement_matrix.transpose(), linalg.inv(measurement_prediction_covariance)))

        x_updated_k = x_k + dot(kalman_gain, measurement_residual)
        p_updated_k = p_k - dot(kalman_gain, dot(measurement_matrix, p_k))

        return x_updated_k, p_updated_k

