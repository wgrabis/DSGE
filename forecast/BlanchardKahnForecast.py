import numpy as np
from sympy import Matrix, pprint
import sympy as sym
from scipy import linalg

from forecast.BlanchardRaw import BlanchardRaw
from model.EquationNormalize import EquationNormalize
from model.EstimationData import EstimationData
from model.ForecastData import ForecastData
from model.PolicyFunction import PolicyFunction

debug_level = 0 # 0 - full 1 - important matrices 2 - only result policy


def mprint(matrix):
    pprint(matrix, wrap_line=False)


class BlanchardKahnForecast:
    def calculate_policy(self, model):
        parameters = model.get_prior_posterior()

        fy_plus, fy_zero, fy_minus, fu = model.build_bh_form(parameters)

        no_static = len(model.static_vars)
        no_state = len(model.state_vars)
        no_mixed = len(model.mixed_vars)
        no_control = len(model.control_vars)

        if no_static > 0:
            S = fy_zero[:, :no_static]
            S = Matrix(S)

            Q, R = linalg.qr(S)

            Q = Matrix(Q)
            R = Matrix(R)

            if debug_level == 0:
                print("S, Q, R:")
                print("S:")
                mprint(S)
                print("Q:")
                mprint(Q)
                print("R:")
                mprint(R)

            a_plus = Q.T @ fy_plus
            a_zero = Q.T @ fy_zero
            a_minus = Q.T @ fy_minus
        else:
            a_plus = fy_plus
            a_zero = fy_zero
            a_minus = fy_minus

        if debug_level == 0:
            print("A+, A0, A-")
            print("A+")
            mprint(a_plus)
            print("A0:")
            mprint(a_zero)
            print("A-:")
            mprint(a_minus)

        ap_plus = a_plus[no_static:, :]
        ap_zero = a_zero[no_static:, no_static:]
        ap_minus = a_minus[no_static:, :]

        if debug_level == 0:
            print("A'-, A'0, A'-")
            print("A'+")
            mprint(ap_plus)
            print("A'0:")
            mprint(ap_zero)
            print("A'-:")
            mprint(ap_minus)

        #todo mixed are currently ignored
        ap_zero_plus = ap_zero[:, no_state:]
        ap_zero_minus = ap_zero[:, :no_state]

        if debug_level == 0:
            print("A0'-, A0'-")
            print("A0'+")
            mprint(ap_zero_plus)
            print("A0'-:")
            mprint(ap_zero_minus)

        left_matrix = sym.zeros(no_state + no_control, no_state + no_control)
        right_matrix = sym.zeros(no_state + no_control, no_state + no_control)

        left_matrix[:, :no_state] = ap_zero_minus[:, :]
        left_matrix[:, no_state:] = ap_plus[:, :]

        right_matrix[:, :no_state] = -1 * ap_minus[:, :]
        right_matrix[:, no_state:] = -1 * ap_zero_plus[:, :]

        # left_matrix, right_matrix,  = EquationNormalize.normalize(A, B, C)

        if debug_level <= 1:
            print("Left matrix, right matrix:")
            print("Left:")
            mprint(left_matrix)
            print("Right:")
            mprint(right_matrix)

        def eigen_sort(a, b):
            return abs(a/b) < 1 + 1e-6

        s, t, s_eigen, t_eigen, q, z = linalg.ordqz(right_matrix, left_matrix, sort=eigen_sort, output="complex")

        t = Matrix(t)
        s = Matrix(s)
        z = Matrix(z)

        if debug_level <= 1:
            print("EigenValues:")
            print(s_eigen/t_eigen)

        z_21 = z.T[no_state:, :no_state]
        z_22 = z.T[no_state:, no_state:]

        zt_11 = z[:no_state, :no_state]

        t_11 = t[:no_state, :no_state]

        s_11 = s[:no_state, :no_state]

        g_y_plus = -1 * z_22.inv() @ z_21
        g_y_minus = zt_11 @ t_11.inv() @ s_11 @ zt_11.inv()

        if debug_level == 0:
            print("Construct G_Y+")
            print("Z22")
            mprint(z_22)
            print("Z22^-1")
            mprint(z_22.inv())
            print("Z21")
            mprint(z_21)

        if debug_level <= 1:
            print("G_Y+:")
            mprint(g_y_plus)

        if debug_level == 0:
            print("Construct G_Y-")
            print("T11^-1 @ S11")
            mprint(t_11.inv() @ s_11)
            print("ZT11")
            mprint(zt_11)
            print("ZT'11")
            mprint(zt_11.inv())

        if debug_level <= 1:
            print("G_Y-:")
            mprint(g_y_minus)

        j_minus = sym.eye(no_state)

        for i in range(no_control):
            j_minus = j_minus.col_insert(no_state + i, sym.zeros(no_state, 1))

        for i in range(no_static):
            j_minus = j_minus.col_insert(0, sym.zeros(no_state, 1))

        g_u = None

        if no_state == 0:
            g_u = -1 * fy_zero.inv() @ fu
        else:
            g_u = -1 * (fy_plus @ g_y_plus @ j_minus + fy_zero).inv() @ fu

        if debug_level == 0:
            print("Construct G_U")
            print("FY_Plus")
            mprint(fy_plus)
            print("J_Minus")
            mprint(j_minus)
            print("GY+ part")
            mprint(fy_plus @ g_y_plus @ j_minus)
            print("FY_zero")
            mprint(fy_zero)
            print("Inverse part")
            if not no_state == 0:
                mprint((fy_plus @ g_y_plus @ j_minus + fy_zero).inv())
            else:
                mprint(fy_zero.inv())
            print("FU")
            mprint(fu)

        if debug_level <= 1:
            print("GU")
            mprint(g_u)

        ad_plus = a_plus[:no_static, :]
        ad_minus = a_minus[:no_static, :]
        ad_zero_s = a_zero[:no_static, :no_static]
        ad_zero_d = a_zero[:no_static, no_static:]

        g_y_d = np.zeros((no_state + no_control, no_state))

        g_y_d[:no_state, :] = g_y_minus[:, :]
        g_y_d[no_state:, :] = g_y_plus[:, :]

        g_y_static = -1 * ad_zero_s.inv() @ (ad_plus @ g_y_plus @ g_y_minus + ad_zero_d @ g_y_d + ad_minus)

        policy_function = PolicyFunction(model, g_y_minus, g_y_plus, g_y_static, g_u)

        return policy_function

    def predict_observables(self, model, policy_function, time, ):
        parameters = model.get_prior_posterior()
        measurement_function, _ = model.measurement_matrices(parameters)
        dummy_data = EstimationData([], 0)

        forecast_data = ForecastData(dummy_data)

        full_forecast = []

        for var_vector in policy_function.predict(time):
            observables = measurement_function(np.array(var_vector).flatten(), time + 1)
            full_forecast.append(observables)

        forecast_data.add_main_forecast(full_forecast)
        return forecast_data