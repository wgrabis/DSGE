from numpy import dot
import numpy as np

from model.MeasurementFunction import MeasurementFunction


class DsgeModel:
    def __init__(self, name,
                 transition_matrix, shock_matrix,
                 measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
                 noise_covariance, measurement_noise_covariance,
                 structural, shocks,
                 structural_prior,
                 shock_prior,
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

        self.structural = structural
        self.structural_prior = structural_prior
        self.shock_prior = shock_prior
        # self.priors = priors

    def build_matrices(self, posterior):
        return self.transition_matrix(posterior), self.shock_matrix(posterior)

    def measurement_matrices(self, posterior):
        m_base_matrix = self.measurement_base_matrix(posterior)
        m_time_matrix = self.measurement_time_matrix(posterior)
        m_state_matrix = self.measurement_state_matrix(posterior)

        return MeasurementFunction(m_base_matrix, m_time_matrix, m_state_matrix), m_state_matrix

    def prior_probability(self, posterior):
        return self.structural_prior.probability_of(posterior)

    def get_prior_posterior(self):
        return np.array(self.structural_prior.get_mean())
