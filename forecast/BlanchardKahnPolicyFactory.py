import logging
import numpy as np
from sympy import Matrix, pprint
import sympy as sym
from scipy import linalg

from model.EstimationData import EstimationData
from model.forecast.CalibrationForecastData import CalibrationForecastData
from model.forecast.ForecastData import ForecastData
from model.PolicyFunction import PolicyFunction
from util.NpUtils import to_np

debug_level = 2 # 0 - full 1 - important matrices 2 - only result policy

logger = logging.getLogger(__name__)


def mprint(matrix):
    pprint(matrix, wrap_line=False)


class BlanchardKahnPolicyFactory:
    def __init__(self, model):
        self.model = model

    def create_policy(self, posterior):
        fy_plus, fy_zero, fy_minus, fu = self.model.build_bh_form(posterior)

        no_static = len(self.model.static_vars)
        no_state = len(self.model.state_vars)
        no_mixed = len(self.model.mixed_vars)
        no_control = len(self.model.control_vars)

        if no_static > 0:
            S = fy_zero[:, :no_static]
            S = Matrix(S)

            Q, R = linalg.qr(S)

            Q = Matrix(Q)
            R = Matrix(R)

            if debug_level == 0:
                logger.debug("S, Q, R:")
                logger.debug("S:")
                logger.debug(to_np(S))
                logger.debug("Q:")
                logger.debug(to_np(Q))
                logger.debug("R:")
                logger.debug(to_np(R))

            a_plus = Q.T @ fy_plus
            a_zero = Q.T @ fy_zero
            a_minus = Q.T @ fy_minus
        else:
            a_plus = fy_plus
            a_zero = fy_zero
            a_minus = fy_minus

        if debug_level == 0:
            logger.debug("A+, A0, A-")
            logger.debug("A+")
            logger.debug(to_np(a_plus))
            logger.debug("A0:")
            logger.debug(to_np(a_zero))
            logger.debug("A-:")
            logger.debug(to_np(a_minus))

        ap_plus = a_plus[no_static:, :]
        ap_zero = a_zero[no_static:, no_static:]
        ap_minus = a_minus[no_static:, :]

        if debug_level == 0:
            logger.debug("A'-, A'0, A'-")
            logger.debug("A'+")
            logger.debug(to_np(ap_plus))
            logger.debug("A'0:")
            logger.debug(to_np(ap_zero))
            logger.debug("A'-:")
            logger.debug(to_np(ap_minus))

        ap_zero_plus = ap_zero[:, no_state:]
        ap_zero_minus = ap_zero[:, :no_state]

        if debug_level == 0:
            logger.debug("A0'-, A0'-")
            logger.debug("A0'+")
            logger.debug(to_np(ap_zero_plus))
            logger.debug("A0'-:")
            logger.debug(to_np(ap_zero_minus))

        left_matrix = sym.zeros(no_state + no_control, no_state + no_control)
        right_matrix = sym.zeros(no_state + no_control, no_state + no_control)

        left_matrix[:, :no_state] = ap_zero_minus[:, :]
        left_matrix[:, no_state:] = ap_plus[:, :]

        right_matrix[:, :no_state] = -1 * ap_minus[:, :]
        right_matrix[:, no_state:] = -1 * ap_zero_plus[:, :]

        # left_matrix, right_matrix,  = EquationNormalize.normalize(A, B, C)

        if debug_level <= 1:
            logger.debug("Left matrix, right matrix:")
            logger.debug("Left:")
            logger.debug(to_np(left_matrix))
            logger.debug("Right:")
            logger.debug(to_np(right_matrix))

        def eigen_sort(a, b):
            return abs(a / b) < 1 + 1e-6

        s, t, s_eigen, t_eigen, q, z = linalg.ordqz(right_matrix, left_matrix, sort=eigen_sort, output="complex")

        t = Matrix(t)
        s = Matrix(s)
        z = Matrix(z)

        logger.info("Eigenvalues")
        logger.info(s_eigen / t_eigen)

        real_eigenvalues = np.array(s_eigen / t_eigen, dtype="complex").astype(np.float32)

        logger.debug(real_eigenvalues)

        non_explosive = sum(abs(x) < 1 for x in real_eigenvalues)

        if non_explosive != no_state:
            logger.warning("Policy cannot be constructed for posterior")
            logger.warning(posterior)
            logger.warning("Got no_non_explosive/no_state: " + str(non_explosive) + "/" + str(no_state))

        assert non_explosive == no_state

        z_21 = z.T[no_state:, :no_state]
        z_22 = z.T[no_state:, no_state:]

        zt_11 = z[:no_state, :no_state]

        t_11 = t[:no_state, :no_state]

        s_11 = s[:no_state, :no_state]

        g_y_plus = -1 * z_22.inv() @ z_21
        g_y_minus = zt_11 @ t_11.inv() @ s_11 @ zt_11.inv()

        if debug_level == 0:
            logger.debug("Construct G_Y+")
            logger.debug("Z22")
            logger.debug(to_np(z_22))
            logger.debug("Z22^-1")
            logger.debug(to_np(z_22.inv()))
            logger.debug("Z21")
            logger.debug(to_np(z_21))

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

        g_y_d = np.zeros((no_state + no_control, no_state), dtype=complex)

        g_y_d[:no_state, :] = g_y_minus[:, :]
        g_y_d[no_state:, :] = g_y_plus[:, :]

        g_y_static = -1 * ad_zero_s.inv() @ (ad_plus @ g_y_plus @ g_y_minus + ad_zero_d @ g_y_d + ad_minus)

        policy_function = PolicyFunction(self.model, g_y_minus, g_y_plus, g_y_static, g_u)

        return policy_function
