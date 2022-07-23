from numpy import dot
import numpy as np
import logging

from sympy import Matrix

from model.MeasurementFunction import MeasurementFunction

logger = logging.getLogger(__name__)


def c_inv(Z):
    if Z.det() == 0:
        return Z.pinv()
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
                 variables,
                 order_variables,
                 static_vars,
                 state_vars,
                 mixed_vars,
                 control_vars,
                 likelihood_filter
                 ):
        self.name = name

        self.structural = structural
        self.structural_prior = structural_prior
        self.shock_prior = shock_prior

        self.measurement_state_matrix = measurement_state_matrix
        self.measurement_time_matrix = measurement_time_matrix
        self.measurement_base_matrix = measurement_base_matrix
        self.measurement_noise_covariance = measurement_noise_covariance

        self.likelihood_filter = likelihood_filter

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
        left, right, shock = self.build_canonical_form()

        inv_left = c_inv(left)

        return inv_left @ right, inv_left @ shock

    def build_canonical_form(self, posterior):
        fy_plus, fy_zero, fy_minus, fu = self.build_bh_form(posterior)
        left, right, shock = None, None, None

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
        logger.debug(self.variables)

        self.fy_plus.print()
        self.fy_zero.print()
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

    def prior_probability(self, posterior):
        return self.structural_prior.probability_of(posterior)

    def posterior_covariance(self):
        return self.structural_prior.get_covariance()

    def get_prior_posterior(self):
        return np.array(self.structural_prior.get_mean())

    def split_variables(self):
        return self.variables