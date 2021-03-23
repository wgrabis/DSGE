from numpy import dot
import numpy as np

from model.Distribution import NormalVectorDistribution
from model.Wrappers import TDistribution
import scipy.linalg as scipy


class LikelihoodAlgorithm:
    def get_likelihood_probability(self, model, data, posterior):
        likelihood_filter = model.likelihood_filter

        # setup
        transition_matrix, shock_matrix = model.build_matrices(posterior)
        noise_covariance = model.noise_covariance(posterior)

        measurement_function, measurement_matrix = model.measurement_matrices(posterior)

        distribution = self.get_invariant_distribution(transition_matrix, shock_matrix)

        # run algorithm
        control_input_matrix = np.eye(transition_matrix.shape[0])
        control_input = np.zeros((transition_matrix.shape[0], 1))

        distributions_history = []

        for t in range(data.estimation_time):
            next_distribution = likelihood_filter.predict(
                distribution.get_vectors(),
                transition_matrix,
                control_input_matrix,
                control_input,
                noise_covariance
            )

            distributions_history.append(next_distribution)
            measurement = data[t]

            updated_distribution = likelihood_filter.update(
                t,
                next_distribution.get_vectors(),
                measurement_matrix,
                measurement_function,
                model.measurement_noise_covariance,
                measurement
            )

            distribution = updated_distribution

        # distribution(posterior) * distribution(data)
        # (distribution(data)  = distribution1(y1)*distribution2(y2)*... )

        posterior_probability = model.prior_probability(posterior)

        for t in range(1, data.estimation_time):
            posterior_probability *= distributions_history[t - 1].probability_of(data.measurements[t - 1])

        return posterior_probability

    def get_invariant_distribution(self, transition_matrix, shock_matrix):
        A = transition_matrix - np.eye(transition_matrix.shape[0])
        B = np.zeros(transition_matrix.shape[0], dtype='float')

        # np.zeros(transition_matrix.shape[0])
        print("---")
        print(A)
        print(B)
        s_0 = np.linalg.solve(A, B)
        p_0 = scipy.solve_discrete_lyapunov(transition_matrix, shock_matrix)
        return NormalVectorDistribution(s_0, p_0)
