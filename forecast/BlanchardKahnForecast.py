import numpy as np
from sympy import Matrix, pprint
import sympy as sym
from scipy import linalg

from forecast.BlanchardRaw import BlanchardRaw
from model.EstimationData import EstimationData
from model.ForecastData import ForecastData

debug_blanchard = True


class BlanchardKahnForecast:

    def substitute_static_vars(self, model, parameters):
        state_count = model.state_var_count
        static_count = model.static_var_count
        control_count = model.control_vars_count

        A, B, C = model.blanchard_raw_representation(parameters)

        A = Matrix(A)
        B = Matrix(B)
        C = Matrix(C)

        S = A[:, :static_count]

        Q, R = linalg.qr(S)

        Q = Matrix(Q)
        R = Matrix(R)

        FY_plus = A[:, (static_count + state_count):]
        FY_minus = B[:, static_count:(static_count + state_count)]
        # F_zero = A[:, (static_count + state_count):]

        var_count, _ = B.shape

        FY_zero = sym.zeros(var_count, var_count)

        FY_zero[:, :(static_count + state_count)] = A[:, :(static_count + state_count)]
        FY_zero[:, (static_count + state_count):] = B[:, (static_count + state_count):]

        # FY_plus[:, :(static_count + state_count)] = F_zero[:, :(static_count + state_count)]
        # FY_minus[:, (static_count + state_count):] = F_zero[:, (static_count + state_count):]

        print("Q, R")
        pprint(S, wrap_line=False)
        pprint(Q, wrap_line=False)
        pprint(R, wrap_line=False)

        pprint(S, wrap_line=False)
        pprint(Q @ R, wrap_line=False)

        print("B, FY+, FY-")
        pprint(B)
        pprint(FY_plus)
        pprint(FY_minus)

        Aplus = Q.T @ FY_plus
        Azero = Q.T @ FY_zero
        Aminus = Q.T @ FY_minus

        print("A+, A0, A-")
        pprint(Aplus)
        pprint(Azero)
        pprint(Aminus)

        Aplus_p = Aplus[static_count:,:]
        Azero_p = Azero[static_count:,:]
        Aminus_p = Aminus[static_count:,:]

        print("A+', A0', A-'")
        pprint(Aplus_p, wrap_line=False)
        pprint(Azero_p, wrap_line=False)
        pprint(Aminus_p, wrap_line=False)

        Azero_p_plus = Azero_p[:, (static_count + state_count):]
        Azero_p_minus = Azero_p[:, static_count:(static_count + state_count)]

        print("A0+', A0-'")
        pprint(Azero_p_plus, wrap_line=False)
        pprint(Azero_p_minus, wrap_line=False)

        A_p = sym.zeros(state_count + control_count, state_count + control_count)
        B_p = sym.zeros(state_count + control_count, state_count + control_count)

        pprint(A_p, wrap_line=False)
        pprint(B_p, wrap_line=False)

        A_p[:, :state_count] = Azero_p_minus[:, :]
        A_p[:, state_count:] = Aplus_p[:, :]

        B_p[:, :state_count] = Aminus_p[:, :]
        B_p[:, state_count:] = Azero_p_plus[:, :]

        C_t = Q.T @ C

        C_p = C_t[static_count:, :]

        return A_p, B_p, C_p

    def calculate(self, model, time, enable_static = False):
        print("Blanchard-cast-forecasting")
        parameters = model.get_prior_posterior()

        # self.substitute_static_vars(model, parameters)
        # return

        A, B, C = model.blanchard_raw_representation(parameters)

        state_count = model.state_var_count
        static_count = model.static_var_count
        control_count = model.control_vars_count

        shock = model.shock_prior.get_mean()

        # blanchard pre

        BlanchardRaw().singular_calculate(A, B, C, np.zeros(state_count + static_count), shock, state_count + static_count,
                                          control_count, time)

        if model.static_var_count != 0 and enable_static:
            A, B, C = self.substitute_static_vars(model, parameters)

        measurement_function, _ = model.measurement_matrices(parameters)
        dummy_data = EstimationData([], 0)

        forecast_data = ForecastData(dummy_data)

        endogenous_count = state_count

        if not enable_static:
            endogenous_count = state_count + static_count

        # todo static_var_count
        x0 = np.zeros(endogenous_count)

        x, y = [], []

        print("Starting blanchard: A, B, C")
        pprint(A, wrap_line=False)
        pprint(B, wrap_line=False)
        pprint(C, wrap_line=False)

        # Ap = A[static_count:, static_count:]
        # Bp = B[static_count:, static_count:]
        # Cp = C[static_count:, :]
        #
        # print("Starting blanchard: A, B, C")
        # pprint(Ap, wrap_line=False)
        # pprint(Bp, wrap_line=False)
        # pprint(Cp, wrap_line=False)

        # todo fix after
        # if A.det() == 0:
        #     x, y = BlanchardRaw().singular_calculate(A, B, C, x0, shock, state_count,
        #                                              len(model.variables) - state_count - static_count, time)
        # else:
        #     x, y = BlanchardRaw().non_singular_calculate(A, B, C, x0, shock, state_count,
        #                                              len(model.variables) - state_count - static_count, time)

        x, y = BlanchardRaw().singular_calculate(A, B, C, x0, shock, endogenous_count,
                                                 control_count, time)
        #

        full_forecast = []

        for i in range(time):
            x_i = x[i]
            y_i = y[i]

            full_state_vector = Matrix(np.zeros(len(model.variables)))

            if enable_static:
                full_state_vector[static_count:(static_count + state_count), :] = x_i
            else:
                full_state_vector[:endogenous_count, :] = x_i
            full_state_vector[(static_count + state_count):, :] = y_i

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




