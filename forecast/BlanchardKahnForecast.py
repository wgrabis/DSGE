import numpy as np
from sympy import Matrix, pprint
import sympy as sym
from scipy import linalg

from forecast.BlanchardKahnPolicyFactory import BlanchardKahnPolicyFactory
from model.EstimationData import EstimationData
from model.forecast.CalibrationForecastData import CalibrationForecastData
from model.forecast.ForecastData import ForecastData
from model.PolicyFunction import PolicyFunction

debug_level = 0 # 0 - full 1 - important matrices 2 - only result policy


def mprint(matrix):
    pprint(matrix, wrap_line=False)


class BlanchardKahnForecast:
    def calculate_policy(self, model):
        parameters = model.get_prior_posterior()

        policy_factory = BlanchardKahnPolicyFactory(model)

        return policy_factory.create_policy(parameters)

    def predict_observables(self, model, policy_function, time):
        parameters = model.get_prior_posterior()
        measurement_function, _ = model.measurement_matrices(parameters)

        full_forecast = []

        for var_vector in policy_function.predict(time):
            observables = measurement_function(np.array(var_vector).flatten(), time + 1)
            full_forecast.append(observables)

        return CalibrationForecastData(full_forecast, model.observable_names)