from numpy import dot
import numpy

from model.LikelihoodSequence import LikelihoodSequence


class DsgeModel:
    def __init__(self, name,
                 transition_matrix, shock_matrix,
                 measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
                 noise_covariance, measurement_noise_covariance,
                 likelihood_filter):
        self.name = name
        self.transition_matrix = transition_matrix
        self.shock_matrix = shock_matrix

        self.measurement_state_matrix = measurement_state_matrix
        self.measurement_time_matrix = measurement_time_matrix
        self.measurement_base_matrix = measurement_base_matrix

        self.noise_covariance = noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance

        self.likelihood_filter = likelihood_filter

        self.time = 0

        self.likelihood_sequence = LikelihoodSequence()

    def build_matrices(self, posterior):
        return self.transition_matrix.fill_values(posterior), self.shock_matrix.fill_values(posterior)

    # def predict_likelihood_distribution(self):
    #     prev_distribution = self.likelihood_sequence.last_distribution()
    #
    #     control_input_matrix = numpy.eye(self.transition_matrix.shape[0])
    #     control_input = numpy.zeros((self.transition_matrix.shape[0], 1))
    #
    #     next_distribution = self.likelihood_filter.predict(
    #         prev_distribution,
    #         self.transition_matrix,
    #         control_input_matrix,
    #         control_input,
    #         self.noise_covariance
    #     )
    #
    #     return next_distribution
    #
    # def update_likelihood_distribution(self, measurement):
    #     prev_distribution = self.likelihood_sequence.last_distribution()
    #
    #     updated_distribution = self.likelihood_filter.update(
    #         prev_distribution,
    #         self.measurement_state_matrix,
    #         self.measurement_function,
    #         self.measurement_noise_covariance,
    #         measurement
    #     )
    #
    #     return updated_distribution
    #
    # def iteration_update(self, measurement):
    #     self.time += 1
    #
    #     updated_distribution = self.update_likelihood_distribution()
    #
    #     self.likelihood_sequence.insert_measurement(measurement, updated_distribution)
    #
    #     self.likelihood_sequence.insert_distribution(self.predict_likelihood_distribution())
    #
    # def measurement_function(self, state):
    #     return self.measurement_base_matrix + dot(self.measurement_time_matrix, self.time) + \
    #            dot(self.measurement_state_matrix, state)
