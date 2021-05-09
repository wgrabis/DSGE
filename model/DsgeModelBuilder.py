from filter.KalmanFilter import KalmanFilter
from model.Distribution import NormalVectorDistribution
from model.DsgeModel import DsgeModel
from numpy import dot
import numpy as np

from model.Equation import EquationParser
from model.VariableMatrix import VariableMatrix, VariableVector, CompVariableMatrix


class DsgeModelBuilder:
    def build(self, name, equations, parameters, variables):
        structural, shocks = self.split_parameters(parameters)

        structural_prior = self.build_prior_distribution(structural, parameters)
        shock_prior = self.build_prior_distribution(shocks, parameters)

        measurement_state_matrix, measurement_time_matrix, measurement_base_matrix \
            = self.prepare_measurement_matrices(equations["observables"], variables, structural)

        transition_matrix, shock_matrix = self.prepare_state_matrices(equations["model"], variables, shocks, structural)

        noise_covariance = self.build_noise_covariance(shock_matrix, shock_prior.get_covariance())

        measurement_noise_covariance = self.build_measurement_noise_covariance(equations["observables"])

        return DsgeModel(
            name,
            transition_matrix, shock_matrix,
            measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
            noise_covariance, measurement_noise_covariance,
            structural, shocks,
            structural_prior,
            self.build_filter()
        )

    @staticmethod
    def prepare_state_matrices(model_equations, variables, shocks, structural):
        lhs, rhs = [], []
        prev_values = [x + x for x in variables]

        for equation in model_equations:
            equation_replace = EquationParser.replace_prev_value(
                equation, variables, lambda parameter: parameter + parameter)

            p_lhs, p_rhs = EquationParser.build_equation(equation_replace)
            rhs.append(p_rhs - p_lhs)
            lhs.append(p_lhs)

        print(rhs)

        left, right = EquationParser.equations_to_matrices(rhs, variables + prev_values + shocks)

        left_state_matrix = left[:, :len(variables)] * (-1)
        right_state_matrix = left[:, len(variables):len(variables) * 2]
        shock_matrix = left[:, len(variables) * 2:]

        print("model builder")
        print(variables + prev_values + shocks)
        print(left)
        print(right)
        print(left_state_matrix)
        print(right_state_matrix)

        inverse_left = left_state_matrix.inv()

        transition_matrix, shock_matrix = inverse_left * right_state_matrix, inverse_left * shock_matrix

        print(transition_matrix)
        print(shock_matrix)
        return VariableMatrix(transition_matrix, structural), VariableMatrix(shock_matrix, structural)

    @staticmethod
    def prepare_measurement_matrices(observable_equations, variables, structural):
        parameters = variables.copy()
        parameters.append("t")

        lhs, rhs = [], []
        for equation in observable_equations:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            lhs.append(p_lhs)
            rhs.append(p_rhs)

        base_left, base_right = EquationParser.equations_to_matrices(rhs, parameters)

        measurement_time_matrix = VariableVector(base_left.col(-1), structural)
        base_left.col_del(-1)

        measurement_base_matrix = VariableVector(base_right * -1, structural)

        measurement_state_matrix = VariableMatrix(base_left, structural)
        return measurement_state_matrix, measurement_time_matrix, measurement_base_matrix

    @staticmethod
    def multiply_noise_covariance(computed_shock, shock_variances):
        return dot(dot(computed_shock, shock_variances), computed_shock.transpose())


    @staticmethod
    def build_noise_covariance(shock_matrix, shock_covariance):
        # todo more options for covariance
        # shock_count = len(shock_variances)
        #
        # shock_covariance = np.zeros((shock_count, shock_count))
        #
        # for i in range(shock_count):
        #     shock_covariance[i, i] = shock_variances[i]

        return CompVariableMatrix(
            shock_matrix,
            lambda computed_shock: DsgeModelBuilder.multiply_noise_covariance(computed_shock, shock_covariance))

    @staticmethod
    def build_measurement_noise_covariance(observables):
        # todo load from model data
        observable_size = len(observables)

        return np.zeros((observable_size, observable_size))

    @staticmethod
    def build_filter():
        return KalmanFilter()

    @staticmethod
    def split_parameters(parameters):
        shocks, structural = [], []
        for key in parameters:
            parameter = parameters[key]
            if parameter["type"] == 'shock':
                shocks.append(key)

            if parameter["type"] == 'structural':
                structural.append(key)
        return structural, shocks

    @staticmethod
    def build_prior_distribution(variables, parameters):
        means, variances = [], []
        for variable in variables:
            means.append(parameters[variable]["mean"])
            variances.append(parameters[variable]["variance"])

        variable_count = len(variables)

        covariance_matrix = np.zeros((variable_count, variable_count))

        for i in range(variable_count):
            covariance_matrix[i, i] = variances[i]

        print("distribution-prior")
        return NormalVectorDistribution(means, covariance_matrix)
