import numpy as np
import logging

from model.forecast.PathForecastData import PathForecastData
from solver.BlanchardKahnPolicyFactory import BlanchardKahnPolicyFactory

logger = logging.getLogger(__name__)


class RandomPathForecast:
    def __init__(self, model, posterior_story, prev_estimations):
        self.model = model
        self.posterior_story = posterior_story
        self.policy_factory = BlanchardKahnPolicyFactory(model)
        self.prev_estimations = prev_estimations

        self.shock_covariance = self.model.shock_prior.get_covariance()
        self.shock_mean = np.zeros(self.model.shock_prior.get_mean().shape)

    def gen_shock(self, time):
        shocks_draw = np.random.multivariate_normal(self.shock_mean, self.shock_covariance)

        logger.debug("Shock:")
        logger.debug(shocks_draw)

        return shocks_draw

    def walk_state_variables(self, x_mean, policy_function, time):
        logger.debug("walk state")
        logger.debug(x_mean)

        x_curr = x_mean
        x_s = []

        # policy_function = self.policy_factory.create_policy(posterior)     # StateTransition(model, posterior)

        logger.debug("Start:")
        logger.debug(x_curr)

        x_curr = policy_function.get_x(x_curr)

        logger.debug(x_curr)
        logger.debug(policy_function.get_start_x())

        for i in range(time):
            shocks_draw = self.gen_shock(time)
            var_vector, x_next = policy_function.predict(x_curr, shocks_draw)

            x_s.append(var_vector)
            x_curr = x_next

        return x_s

    def walk_measurements(self, x_s, measurement_function, time, starting_time):
        logger.debug("walk measurement")
        logger.debug(x_s)

        observables = []

        for i in range(time):
            x_k = x_s[i]

            observables.append(measurement_function(x_k, starting_time + i))

        logger.debug("walk measurement-result")
        logger.debug(observables)

        return observables

    def calculate(self, rounds, time, starting_time):
        measure_len = len(self.model.observable_names)

        forecast_data = PathForecastData(self.prev_estimations, self.model.observable_names)
        average_sum = [np.zeros(measure_len) for _ in range(time)]

        posteriors = self.posterior_story.get_post_burnout()
        posterior_count = len(posteriors)

        logger.info("Running forecast")

        for posterior, distribution in posteriors:
            x_mean, p_cov = distribution.get_vectors()
            policy_function = self.policy_factory.create_policy(posterior)
            measurement_function, _ = self.model.measurement_matrices(posterior)
            for j in range(rounds):
                x_s = self.walk_state_variables(x_mean, policy_function, time)

                observables = self.walk_measurements(x_s, measurement_function, time, starting_time)

                for i in range(time):
                    average_sum[i] += observables[i]

                forecast_data.add_posterior_forecast(observables)

        for i in range(time):
            average_sum[i] /= rounds * posterior_count

        forecast_data.add_main_forecast(average_sum)
        return forecast_data
