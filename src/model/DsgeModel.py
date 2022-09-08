import numpy as np
import logging
import sympy as sym

from sympy import Matrix

from solver.BlanchardKahnPolicyFactory import BlanchardKahnPolicyFactory
from model.MeasurementFunction import MeasurementFunction
from util.NpUtils import to_np

logger = logging.getLogger(__name__)


def c_inv(Z):
    if Z.det() == 0:
        logger.error("Inversing matrix with pinv! Z.det = 0")
        return None
    return Z.inv()


class DsgeModel:
    def __init__(self, name,
                 fy_plus, fy_zero, fy_minus, fu,
                 measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
                 measurement_noise_covariance,
                 observable_names,
                 structural, shocks,
                 structural_prior,
                 shock_prior,
                 order_variables,
                 variables,
                 static_vars,
                 state_vars,
                 mixed_vars,
                 control_vars,
                 likelihood_filter_factory
                 ):
        self.name = name

        # self.structural = structural
        self.structural_prior = structural_prior
        self.shock_prior = shock_prior

        self.measurement_state_matrix = measurement_state_matrix
        self.measurement_time_matrix = measurement_time_matrix
        self.measurement_base_matrix = measurement_base_matrix
        self.measurement_noise_covariance = measurement_noise_covariance

        self.likelihood_filter_factory = likelihood_filter_factory

        self.variables = variables
        self.ordered_variables = order_variables
        self.shocks = shocks

        # self.transition_matrix = transition_matrix
        # self.shock_matrix = shock_matrix

        # self.blanchardA = left_state_matrix
        # self.blanchardB = right_state_matrix
        # self.blanchardC = shock_bk_matrix

        self.fy_plus = fy_plus
        self.fy_zero = fy_zero
        self.fy_minus = fy_minus
        self.fu = fu

        self.static_vars = static_vars
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.mixed_vars = mixed_vars
        self.observable_names = observable_names

    def build_mh_form(self, posterior):
        # todo refactor into separate file
        policy_factory = BlanchardKahnPolicyFactory(self)

        transition_matrix, shock_matrix = policy_factory.create_policy(posterior).map_to_transition()

        logger.info("MH Form for posterior")
        logger.info(posterior)
        logger.info("Transition matrix:")
        logger.info(transition_matrix)
        logger.info("Shock matrix")
        logger.info(shock_matrix)

        return transition_matrix, shock_matrix

        # left, right, shock = self.build_canonical_form(posterior)
        #
        # inv_left = c_inv(left)
        #
        # logger.debug("Left:")
        # logger.debug(to_np(left))
        # logger.debug("Right:")
        # logger.debug(to_np(right))
        # logger.debug("Shock:")
        # logger.debug(to_np(shock))
        #
        # transition_matrix = to_np(inv_left @ right)
        # shock_matrix = to_np(inv_left @ shock)
        #
        # logger.debug("Transition matrix:")
        # logger.debug(transition_matrix)
        # logger.debug("Shock matrix")
        # logger.debug(shock_matrix)
        #
        #

    def build_canonical_form(self, posterior):
        fy_plus, fy_zero, fy_minus, fu = self.build_bh_form(posterior)

        var_len = len(self.ordered_variables)

        left_matrix = sym.zeros(var_len, var_len)
        right_matrix = sym.zeros(var_len, var_len)

        no_static = len(self.static_vars)
        no_state = len(self.state_vars)
        no_mixed = len(self.mixed_vars)

        assert no_mixed == 0

        left_matrix[:, (no_static + no_state):] = -fy_plus
        left_matrix[:, :(no_static + no_state)] = -fy_zero[:, :(no_static + no_state)]

        right_matrix[:, (no_static + no_state):] = fy_zero[:, (no_static + no_state):]
        right_matrix[:, no_static:(no_static + no_state)] = fy_minus

        logger.debug("Posterior for canonical form")
        logger.debug(posterior)
        logger.debug("Canonical form")
        logger.debug(self.ordered_variables)
        logger.debug(to_np(left_matrix))
        logger.debug(to_np(right_matrix))
        logger.debug(to_np(fu))

        left, right, shock = left_matrix, right_matrix, fu

        return left, right, shock

    def print_debug(self):
        logger.debug("Debug model {}".format(self.name))
        logger.debug("Static vars:")
        logger.debug(self.static_vars)
        logger.debug("State vars:")
        logger.debug(self.state_vars)
        logger.debug("Mixed vars:")
        logger.debug(self.mixed_vars)
        logger.debug("Control vars(pure forward):")
        logger.debug(self.control_vars)
        logger.debug("Full order vector:")
        logger.debug(self.ordered_variables)

        print("fy+")
        self.fy_plus.print()
        print("fy0")
        self.fy_zero.print()
        print("fy-")
        self.fy_minus.print()

        self.fu.print()

    def build_bh_form(self, posterior):
        return Matrix(self.fy_plus(posterior)), \
               Matrix(self.fy_zero(posterior)), \
               Matrix(self.fy_minus(posterior)), \
               Matrix(self.fu(posterior))

    def measurement_matrices(self, posterior):
        m_base_matrix = self.measurement_base_matrix(posterior)
        m_time_matrix = self.measurement_time_matrix(posterior)
        m_state_matrix = self.measurement_state_matrix(posterior)

        return MeasurementFunction(m_base_matrix, m_time_matrix, m_state_matrix), m_state_matrix

    # def prior_probability(self, posterior):
    #     return self.structural_prior.probability_of(posterior)

    # def posterior_covariance(self):
    #     return self.structural_prior.get_covariance()

    def shock_covariance(self):
        return self.shock_prior.get_covariance()

    def get_prior_posterior(self):
        return np.array(self.structural_prior.get_mean())

    def split_variables(self):
        return self.variables

    # todo STRUCTURAL
    def noise_covariance(self, posterior):
        return self.measurement_noise_covariance
