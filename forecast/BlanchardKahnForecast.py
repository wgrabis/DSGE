import numpy as np
from sympy import Matrix, pprint
import sympy as sym

from forecast.BlanchardRaw import BlanchardRaw
from model.EstimationData import EstimationData
from model.ForecastData import ForecastData

debug_blanchard = True


class BlanchardKahnForecast:
    def calculate(self, model, time):
        parameters = model.get_prior_posterior()

        A, B, C = model.blanchard_raw_representation(parameters)
        measurement_function, _ = model.measurement_matrices(parameters)
        dummy_data = EstimationData([], 0)

        forecast_data = ForecastData(dummy_data)

        shock = model.shock_prior.get_mean()

        state_count = model.state_var_count

        x0 = np.zeros(state_count)

        x, y = BlanchardRaw().non_singular_calculate(A, B, C, x0, shock, state_count,
                                                     len(model.variables) - state_count, time)
        full_forecast = []

        for i in range(time):
            x_i = x[i]
            y_i = y[i]

            full_state_vector = Matrix(np.zeros(len(model.variables)))
            full_state_vector[:state_count, :] = x_i
            full_state_vector[state_count:, :] = y_i

            print("---STATE")
            print(np.array(full_state_vector).T)
            print("---")

            observables = measurement_function(np.array(full_state_vector).flatten(), time + 1)
            print("---obs")
            print(observables)
            full_forecast.append(observables)

        forecast_data.add_main_forecast(full_forecast)
        return forecast_data

# class BlanchardKahnForecast:
#     def __init__(self, model):
#         self.model = model
#
#         self.start_state_vector = []
#         self.H11 = None
#         self.H12 = None
#         self.H21 = None
#         self.H22 = None
#
#         self.H11R = None
#         self.H12R = None
#         self.H21R = None
#         self.H22R = None
#
#         self.G = None
#         self.G1 = None
#         self.G2 = None
#
#         self.F11 = None
#         self.F12 = None
#
#         self.J1 = None
#         self.J2 = None
#
#         self.state_count = model.state_var_count
#
#         self.measurement_function = None
#
#         self.prepare()
#
#     def prepare(self):
#         state_count = self.state_count
#         parameters = self.model.get_prior_posterior()
#
#         transition_m, shock_m = self.model.build_matrices(parameters)
#         measurement_function, _ = self.model.measurement_matrices(parameters)
#
#         self.measurement_function = measurement_function
#
#         sympy_transition = Matrix(transition_m)
#
#         H, J = sympy_transition.diagonalize()
#
#         HR = H.inv()
#
#         self.H11 = H[:state_count, :state_count]
#         self.H12 = H[:state_count, state_count:]
#         self.H21 = H[state_count:, :state_count]
#         self.H22 = H[state_count:, state_count:]
#
#         self.H11R = HR[:state_count, :state_count]
#         self.H12R = HR[:state_count, state_count:]
#         self.H21R = HR[state_count:, :state_count]
#         self.H22R = HR[state_count:, state_count:]
#
#         self.G = Matrix(shock_m)
#         self.G1 = self.G[:state_count, :]
#         self.G2 = self.G[state_count:, :]
#
#         self.F11 = sympy_transition[:state_count, :state_count]
#         self.F12 = sympy_transition[:state_count, state_count:]
#
#         self.J1 = J[0:state_count, 0:state_count]
#         self.J2 = J[state_count:, state_count:]
#
#         if debug_blanchard:
#             print("Debug matrices")
#             pprint(sympy_transition, wrap_line=False)
#             pprint(self.G, wrap_line=False)
#             print("H:")
#             pprint(H, wrap_line=False)
#             pprint(self.H11, wrap_line=False)
#             pprint(self.H12, wrap_line=False)
#             pprint(self.H21, wrap_line=False)
#             pprint(self.H22, wrap_line=False)
#             pprint(self.G, wrap_line=False)
#             pprint(self.G1, wrap_line=False)
#             pprint(self.G2, wrap_line=False)
#             pprint(self.F11, wrap_line=False)
#             pprint(self.F12, wrap_line=False)
#             pprint(J, wrap_line=False)
#             pprint(self.J1, wrap_line=False)
#             pprint(self.J2, wrap_line=False)
#             print("H check:")
#             pprint(sympy_transition, wrap_line=False)
#             pprint(H * J * H.inv(), wrap_line=False)
#
#         self.start_state_vector = Matrix(np.zeros(state_count))
#
#     def draw_shock(self, iteration):
#         shock_len = len(self.model.shocks)
#         result = Matrix(np.zeros(shock_len))
#         if iteration == 0:
#             result = Matrix(self.model.shock_prior.get_mean())
#
#         print("draw-shock")
#         print(result)
#         return result
#
#     def predict_control(self, iteration, state_vector):
#         state_part = self.H22R.pinv() @ self.H21R
#         inner_shock = (self.H21R @ self.G1 + self.H22R @ self.G2)
#         outer_shock = self.H22R.pinv() @ self.J2.inv()
#
#         if debug_blanchard:
#             print("predict control")
#             pprint(state_part)
#             print("state vector")
#             pprint(state_vector)
#             print("state part")
#             pprint(state_part * state_vector)
#             print("shock control")
#             pprint(inner_shock)
#             pprint(outer_shock)
#             pprint(outer_shock * inner_shock)
#
#         shock_part = outer_shock * inner_shock
#
#         return -1 * state_part @ state_vector - shock_part @ self.draw_shock(iteration)
#
#     def predict_state(self, iteration, state_vector, control_vector):
#         state_progress = self.F11 @ state_vector
#         control_progress = self.F12 @ control_vector
#         shock_progress = self.G1 @ self.draw_shock(iteration)
#
#         if debug_blanchard:
#             print("predict state")
#             pprint(state_progress)
#             print("=============")
#             pprint(control_progress)
#             print("=============")
#             pprint(shock_progress)
#             print("=============")
#
#         return state_progress + control_progress + shock_progress
#
#     def calculate(self, time):
#         # todo refactor
#         dummy_data = EstimationData([], 0)
#
#         forecast_data = ForecastData(dummy_data)
#
#         curr_state = self.start_state_vector
#         curr_control = []
#
#         full_forecast = []
#
#         for i in range(time):
#             curr_control = self.predict_control(i, curr_state)
#             next_state = self.predict_state(i, curr_state, curr_control)
#
#             print('Iteration, {}'.format(i))
#             print(curr_state.shape)
#             print(curr_control.shape)
#
#             full_state_vector = Matrix(np.zeros(len(self.model.variables)))
#             print(full_state_vector.shape)
#             full_state_vector[:self.state_count, :] = curr_state
#             full_state_vector[self.state_count:, :] = curr_control
#
#             print("---STATE")
#             print(np.array(full_state_vector).T)
#             print("---")
#
#             curr_state = next_state
#
#             observables = self.measurement_function(np.array(full_state_vector).flatten(), time + 1)
#             print("---obs")
#             print(observables)
#
#             if i != 0:
#                 full_forecast.append(observables)
#
#         forecast_data.add_main_forecast(full_forecast)
#         return forecast_data




