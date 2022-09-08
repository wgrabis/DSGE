import numpy as np
from sympy import pprint

from solver.BlanchardKahnPolicyFactory import BlanchardKahnPolicyFactory
from model.forecast.CalibrationForecastData import CalibrationForecastData

debug_level = 0 # 0 - full 1 - important matrices 2 - only result policy


def mprint(matrix):
    pprint(matrix, wrap_line=False)


class ImpulseResponseForecast:
    def __init__(self, model, parameters):
        self.parameters = parameters
        self.model = model

    def prepare_policy(self):
        policy_factory = BlanchardKahnPolicyFactory(self.model)

        return policy_factory.create_policy(self.parameters)

    def __gen_shock(self, time):
        if time == 0:
            return self.model.shock_prior.get_mean()
        return np.zeros(len(self.model.shocks))

    def predict_observables(self, time):
        policy_function = self.prepare_policy()

        measurement_function, _ = self.model.measurement_matrices(self.parameters)

        full_forecast = []
        x_curr = policy_function.get_start_x()

        for time in range(time):
            curr_shock = self.__gen_shock(time)
            var_vector, x_next = policy_function.predict(x_curr, curr_shock)
            observables = measurement_function(np.array(var_vector).flatten(), time + 1)
            full_forecast.append(observables)

            x_curr = x_next

        return CalibrationForecastData(full_forecast, self.model.observable_names), policy_function
