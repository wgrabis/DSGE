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

        FY_plus = -1 * A[:, (static_count + state_count):]
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

    def calculate(self, model, time, enable_static=False):
        parameters = model.get_prior_posterior()

        # self.substitute_static_vars(model, parameters)
        # return

        A, B, C = model.blanchard_raw_representation(parameters)

        Ap, Bp, Cp = A, B, C

        print("Blanchard-cast-forecasting")
        print(parameters)
        pprint(A, wrap_line=False)
        pprint(B, wrap_line=False)
        pprint(C, wrap_line=False)

        state_count = model.state_var_count
        static_count = model.static_var_count
        control_count = model.control_vars_count

        shock = model.shock_prior.get_mean()

        # blanchard pre

        X1, Y1 = BlanchardRaw().singular_calculate(model, A, B, C, np.zeros(state_count + static_count), shock, state_count + static_count,
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
        pprint(Ap, wrap_line=False)
        pprint(B, wrap_line=False)
        pprint(Bp, wrap_line=False)
        pprint(C, wrap_line=False)
        pprint(Cp, wrap_line=False)

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

        x, y = BlanchardRaw().singular_calculate(model, A, B, C, x0, shock, endogenous_count,
                                                 control_count, time)
        #

        full_forecast = []

        x, y = X1, Y1

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


