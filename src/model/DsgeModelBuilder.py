from filter.FilterFactory import KalmanFilterFactory
from model.DefinitionSet import DefinitionSet
from model.Distribution import NormalVectorDistribution
from model.DsgeModel import DsgeModel
from numpy import dot
import numpy as np

from model.Equation import EquationParser
from model.RandomParameter import RandomParameter
from model.StructuralParameterSet import StructuralParameterSet
from model.VariableMatrix import VariableMatrix, VariableVector
import logging

log = logging.getLogger(__name__)


class PrintArray:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __repr__(self):
        rpr = ('PrintArray(' +
               ', '.join([f'{name}={value}' for name, value in self._kwargs.items()]) +
               ')')
        return rpr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc != np.floor_divide:
            return NotImplemented
        a = inputs[0]
        with np.printoptions(**self._kwargs):
            print(a)


printer = PrintArray(precision=4, linewidth=150, suppress=True)


class DsgeModelBuilder:
    def build(self, raw_model):
        variables, raw_structural, shocks = raw_model.entities()
        priors = raw_model.priors

        structural_prior = self.build_structural_parameters(priors, raw_structural)

        structural = structural_prior.ordered_params

        definition_set = self.build_definition_set(raw_model.definitions, structural)
        shock_prior = self.build_prior_distribution(shocks, priors)

        fy_plus, fy_zero, fy_minus, fu, static_vars, state_vars, mixed_vars, control_vars \
            = EquationParser.parse_equations_to_functional(raw_model.equations, variables, shocks)

        measurement_noise_covariance = self.build_measurement_noise_covariance(raw_model.observables)

        ordered_variables = static_vars + state_vars + mixed_vars + control_vars

        measurement_state_matrix, measurement_time_matrix, measurement_base_matrix, observable_names \
            = self.prepare_measurement_matrices(raw_model.observables, ordered_variables, structural, definition_set)

        return DsgeModel(
            raw_model.name,
            VariableMatrix(fy_plus, structural, definition_set),
            VariableMatrix(fy_zero, structural, definition_set),
            VariableMatrix(fy_minus, structural, definition_set),
            VariableMatrix(fu, structural, definition_set),
            measurement_state_matrix, measurement_time_matrix, measurement_base_matrix,
            measurement_noise_covariance,
            observable_names,
            structural, shocks,
            structural_prior,
            shock_prior,
            ordered_variables,
            variables,
            static_vars,
            state_vars,
            mixed_vars,
            control_vars,
            self.build_filter_factory()
        )

    @staticmethod
    def prepare_transition_matrix(left_matrix, right_matrix):
        log.debug("Prepare transition")
        log.debug(left_matrix)
        log.debug(right_matrix)

        inverse_left = np.linalg.inv(left_matrix)

        transition_matrix = inverse_left @ right_matrix

        return transition_matrix

    @staticmethod
    def prepare_shock_matrix(left_matrix, shock_matrix):
        log.debug("Prepare shock")
        log.debug(left_matrix)
        log.debug(shock_matrix)

        inverse_left = np.linalg.inv(left_matrix)

        shock_matrix = inverse_left @ shock_matrix

        return shock_matrix

    # @staticmethod
    # def prepare_state_matrices(left, right, shock, structural, definition_set):
    #     left_variable = VariableMatrix(left, structural, definition_set)
    #     right_variable = VariableMatrix(right, structural, definition_set)
    #     shock_variable = VariableMatrix(shock, structural, definition_set)
    #
    #     return (
    #         CompDoubleVariableMatrix(
    #             left_variable,
    #             right_variable,
    #             lambda c_left, c_right: DsgeModelBuilder.prepare_transition_matrix(c_left, c_right)
    #         ),
    #         CompDoubleVariableMatrix(
    #             left_variable,
    #             shock_variable,
    #             lambda c_left, c_shock: DsgeModelBuilder.prepare_shock_matrix(c_left, c_shock)
    #         )
    #     )
    #     # return VariableMatrix(transition_matrix, structural), VariableMatrix(shock_matrix, structural)

    @staticmethod
    def prepare_measurement_matrices(observable_equations, variables, structural, definition_set):
        parameters = variables.copy()
        parameters.append("t")

        observable_names, rhs = [], []
        for equation in observable_equations:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            observable_names.append(str(p_lhs))
            rhs.append(p_rhs)

        base_left, base_right = EquationParser.equations_to_matrices(rhs, parameters)

        log.info("Measurement matrices")
        log.info(parameters)
        log.info(base_left)
        log.info(base_right)
        log.info(observable_names)

        measurement_time_matrix = VariableVector(base_left.col(-1), structural, definition_set)
        base_left.col_del(-1)

        measurement_base_matrix = VariableVector(base_right * -1, structural, definition_set)

        measurement_state_matrix = VariableMatrix(base_left, structural, definition_set)
        return measurement_state_matrix, measurement_time_matrix, measurement_base_matrix, observable_names

    # @staticmethod
    # def multiply_noise_covariance(computed_shock, shock_variances):
    #     log.debug("multiply noise cov")
    #     log.debug(computed_shock)
    #     log.debug(shock_variances)
    #
    #     return dot(dot(computed_shock, shock_variances), computed_shock.transpose())

    @staticmethod
    def build_definition_set(definitions, structural):
        parsed_definitions = []
        for equation in definitions:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            parsed_definitions.append((p_lhs, p_rhs))

        return DefinitionSet(structural, parsed_definitions)

    # @staticmethod
    # def build_noise_covariance(shock_matrix, shock_covariance):
    #     # todo more options for covariance
    #     # shock_count = len(shock_variances)
    #     #
    #     # shock_covariance = np.zeros((shock_count, shock_count))
    #     #
    #     # for i in range(shock_count):
    #     #     shock_covariance[i, i] = shock_variances[i]
    #     return CompVariableMatrix(
    #         shock_matrix,
    #         lambda computed_shock: DsgeModelBuilder.multiply_noise_covariance(computed_shock, shock_covariance))

    @staticmethod
    def build_measurement_noise_covariance(observables):
        # todo load from model data
        observable_size = len(observables)

        return np.zeros((observable_size, observable_size))

    @staticmethod
    def build_filter_factory():
        return KalmanFilterFactory()

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
    def build_structural_parameters(parameters, structural):
        calibrated_names = []
        calibrated_values = []
        means = []
        variances = []
        random_part_names = []
        name_map = {}

        count = len(structural)

        for i in range(count):
            variable = structural[i]

            if isinstance(variable, str):
                name_map[variable] = variable
            else:
                assert "name" in variable
                assert "display" in variable

                var_name = variable["name"]
                name_map[var_name] = variable["display"]

                variable = var_name

            calibration = parameters[variable]

            dist_type = calibration["distribution"]
            if dist_type == "calibration":
                calibrated_values.append(calibration["value"])
                calibrated_names.append(variable)
            elif dist_type == "normal":
                mean = calibration["mean"]
                variance = calibration["variance"]

                upper_bound = calibration['upperBound'] if 'upperBound' in calibration else None
                lower_bound = calibration['lowerBound'] if 'lowerBound' in calibration else None

                means.append(mean)
                variances.append(variance)
                random_part_names.append(RandomParameter(variable, lower_bound, upper_bound))
            else:
                raise Exception("Parameter description not recognized")

        random_size = len(random_part_names)
        covariance = np.zeros((random_size, random_size))

        for i in range(random_size):
            covariance[i, i] = variances[i]

        return StructuralParameterSet(calibrated_names, random_part_names, calibrated_values, means, covariance, name_map)

    @staticmethod
    def build_prior(variables, parameters):
        means = []

        # todo more options for covariance
        count = len(variables)

        covariance = np.zeros((count, count))

        for i in range(count):
            variable = variables[i]
            calibration = parameters[variable]

            dist_type = calibration["distribution"]
            if dist_type == "calibration":
                mean = calibration["value"]
                variance = 0
            elif dist_type == "normal":
                mean = calibration["mean"]
                variance = calibration["variance"]
            else:
                mean = None
                variance = None

            assert mean is not None

            means.append(mean)
            covariance[i, i] = variance

        return means, covariance

    # @staticmethod
    # def build_structural_distribution(variables, parameters):
    #     means, covariance = DsgeModelBuilder.build_prior(variables, parameters)
    #
    #     return StructuralParameterSet(variables, means, covariance)

    @staticmethod
    def build_prior_distribution(variables, parameters):
        means, covariance = DsgeModelBuilder.build_prior(variables, parameters)

        return NormalVectorDistribution(means, covariance)
