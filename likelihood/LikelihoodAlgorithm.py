from numpy import dot
import numpy

from model.Wrappers import Distribution


class LikelihoodAlgorithm:
    def get_likelihood_probability(self, model, data, posterior):
        likelihood_filter = model.likelihood_filter

        # setup
        transition_matrix, shock_matrix = model.build_matrices(posterior)

        distribution = self.get_invariant_distribution(transition_matrix, shock_matrix)

        # run algorithm

        control_input_matrix = numpy.eye(transition_matrix.shape[0])
        control_input = numpy.zeros((transition_matrix.shape[0], 1))

        for t in range(1, data.estimation_time):
            next_distribution = likelihood_filter.predict(
                distribution,
                transition_matrix,
                control_input_matrix,
                control_input,
                model.noise_covariance
            )

            measurement = data.measurements[t - 1]

            updated_distribution = likelihood_filter.update(
                next_distribution,
                model.measurement_state_matrix,
                model.measurement_function,
                model.measurement_noise_covariance,
                measurement
            )

            distribution = updated_distribution

        # todo calculate P(O)*P(Y|0)
        return

    def get_invariant_distribution(self, transition_matrix, shock_matrix):
        # todo
        pass
