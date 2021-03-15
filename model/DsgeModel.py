from numpy import dot
import numpy as np

from model.LikelihoodSequence import LikelihoodSequence


class DsgeModel:
    def __init__(self, name,
                 transition_matrix, shock_matrix,
                 measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
                 noise_covariance, measurement_noise_covariance,
                 structural, priors,
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

        self.structural = structural
        self.priors = priors

        self.likelihood_sequence = LikelihoodSequence()

    def build_matrices(self, posterior):
        return self.transition_matrix.fill_values(posterior), self.shock_matrix.fill_values(posterior)

    def measurement_function(self, state):
        return self.measurement_base_matrix + dot(self.measurement_time_matrix, self.time) + \
               dot(self.measurement_state_matrix, state)

    def prior_probability(self, posterior):
        #todo
        pass

    def get_prior_posterior(self):
        prior_vector = []

        for variable in self.structural:
            prior_vector.append(self.priors[variable].mean)

        return np.array(prior_vector)
