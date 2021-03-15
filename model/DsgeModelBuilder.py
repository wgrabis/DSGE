from filter.KalmanFilter import KalmanFilter
from model.DsgeModel import DsgeModel
from numpy import dot


class DsgeModelBuilder:
    def build(self, name, equations, structural, shocks, priors):
        transition_matrix, shock_matrix = self.prepare_state_matrices(equations.model)

        measurement_state_matrix, measurement_time_matrix, measurement_base_matrix \
            = self.prepare_measurement_matrices(equations.observables)

        noise_covariance = self.build_noise_covariance(shock_matrix,)

        measurement_noise_covariance = self.build_measurement_noise_covariance()

        return DsgeModel(
            name,
            transition_matrix, shock_matrix,
            measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
            noise_covariance, measurement_noise_covariance,
            structural, priors,
            self.build_filter()
        )


    @staticmethod
    def prepare_state_matrices(model_equations):
        # todo
        transition_matrix, shock_matrix = None, None
        return transition_matrix, shock_matrix

    @staticmethod
    def prepare_measurement_matrices(observable_equations):
        # todo
        measurement_state_matrix, measurement_time_matrix, measurement_base_matrix = None, None
        return measurement_state_matrix, measurement_time_matrix, measurement_base_matrix

    @staticmethod
    def build_noise_covariance(shock_matrix, shock_variances):
        return dot(shock_matrix, dot(shock_variances, shock_matrix.transpose()))

    @staticmethod
    def build_measurement_noise_covariance():
        # todo
        pass

    @staticmethod
    def build_filter():
        return KalmanFilter()
