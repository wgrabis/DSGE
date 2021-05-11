import numpy as np

from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from model.StateTransition import StateTransition


class ForecastAlgorithm:
    def __init__(self, model):
        self.model = model

    def walk_state_variables(self, x_mean, p_cov, posterior, time):
        model = self.model

        print("walk state")
        print(x_mean)
        print(p_cov)

        x_next = np.random.multivariate_normal(x_mean, p_cov)
        x_s = []

        state_transition = StateTransition(model, posterior)

        for i in range(time):
            covariances = model.shock_prior.get_covariance()

            shocks_draw = np.random.multivariate_normal(np.zeros(model.shock_prior.get_mean().shape), covariances)
            x_next = state_transition(x_next, shocks_draw)

            x_s.append(x_next)

        return x_s

    def walk_measurements(self, x_s, posterior, time, starting_time):
        # todo add measurement errors
        model = self.model

        print("walk measurement")
        print(x_s)

        measurement_function, _ = model.measurement_matrices(posterior)
        observables = []

        for i in range(time):
            x_k = x_s[i]

            observables.append(measurement_function(x_k, starting_time + i))

        print("walk measurement -result")
        print(observables)

        return observables

    def calculate(self, posteriors, rounds, time, starting_time, data):
        measure_len, _ = data.size()

        average_sum = [np.zeros(measure_len) for _ in range(time)]

        # todo refactor for matrix average sum

        posterior_count = len(posteriors)

        for posterior, distribution in posteriors:
            x_mean, p_cov = distribution.get_vectors()
            for j in range(rounds):
                x_s = self.walk_state_variables(x_mean, p_cov, posterior, time)

                observables = self.walk_measurements(x_s, posterior, time, starting_time)

                for i in range(time):
                    average_sum[i] += observables[i]

        for i in range(time):
            average_sum[i] /= rounds * posterior_count
        return average_sum

