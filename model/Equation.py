from sympy import Symbol, sympify, symbols, linear_eq_to_matrix, pprint, pretty
import logging

logger = logging.getLogger(__name__)


mix_prefix = "mx_pre_v{}"


class EquationParser:
    @staticmethod
    def build_equation(equation):
        lhs, rhs = equation.split("=")

        p_lhs = sympify(lhs)
        p_rhs = sympify(rhs)

        return p_lhs, p_rhs

    @staticmethod
    def replace_prev_value(equation, prev_parameters, mapping):
        result = equation
        for parameter in prev_parameters:
            result = result.replace(parameter + "(-1)", mapping(parameter))

        return result

    @staticmethod
    def equations_to_matrices(equations, variables):
        left, right = linear_eq_to_matrix(equations, symbols(" ".join(variables)))
        pprint(left, wrap_line=False)
        pprint(right, wrap_line=False)

        return left, right

    @staticmethod
    def split_variables(equations, variables):
        state_variables = set()
        control_variables = set()
        mixed_variables = set()

        for equation in equations:
            for variable in variables:
                if variable in mixed_variables:
                    continue
                if equation.find('{}(+1)'.format(variable)) != -1:
                    if variable in state_variables:
                        state_variables.remove(variable)
                        mixed_variables.add(variable)
                    else:
                        control_variables.add(variable)
                if equation.find('{}(-1)'.format(variable)) != -1:
                    if variable in control_variables:
                        control_variables.remove(variable)
                        mixed_variables.add(variable)
                    else:
                        state_variables.add(variable)

        assert len(state_variables.intersection(control_variables)) == 0

        static_variables = set(variables)\
            .difference(mixed_variables)\
            .difference(control_variables)\
            .difference(state_variables)

        ordered_state = []
        ordered_control = []
        ordered_static = []
        ordered_mixed = []

        for variable in variables:
            if variable in mixed_variables:
                ordered_mixed.append(variable)
            if variable in state_variables:
                ordered_state.append(variable)
            if variable in control_variables:
                ordered_control.append(variable)
            if variable in static_variables:
                ordered_static.append(variable)

        return ordered_state, ordered_control, ordered_mixed, ordered_static

    @staticmethod
    def generate_mapping(param, var_type):
        if var_type == -1:
            return '{}_prev'.format(param)
        if var_type == 0:
            return param
        if var_type == 1:
            return '{}_fwd'.format(param)

    @staticmethod
    def rename_variables(equations, state_variables, control_variables, mixed_variables, static_variables):
        renamed_equations = []

        def name_mapping(param, var_type):
            return EquationParser.generate_mapping(param, var_type)

        for equation in equations:
            result = equation

            for parameter in static_variables:
                result = result.replace(parameter, name_mapping(parameter, 0))

            for parameter in mixed_variables:
                result = result.replace('{}(-1)'.format(parameter), name_mapping(parameter, -1))
                result = result.replace('{}(+1)'.format(parameter), name_mapping(parameter, 1))
                result = result.replace(parameter, name_mapping(parameter, 0))

            for parameter in state_variables:
                result = result.replace('{}(-1)'.format(parameter), name_mapping(parameter, -1))
                result = result.replace(parameter, name_mapping(parameter, 0))

            for parameter in control_variables:
                result = result.replace('{}(+1)'.format(parameter), name_mapping(parameter, 1))
                result = result.replace(parameter, name_mapping(parameter, 0))

            renamed_equations.append(result)

        return renamed_equations

    @staticmethod
    def remove_mixed_variables(in_equations, mixed_variables):
        equations = in_equations.copy()
        new_fw_variables = []
        new_equations = []
        for i in range(len(mixed_variables)):
            new_var = mix_prefix.format(i)
            mix_var = mixed_variables[i]

            mix_symbol = EquationParser.generate_mapping(mix_var, 0)
            mix_pre_symbol = EquationParser.generate_mapping(mix_var, -1)
            mix_fwd_symbol = EquationParser.generate_mapping(mix_var, +1)
            new_symbol = EquationParser.generate_mapping(new_var, 0)
            new_fwd_symbol = EquationParser.generate_mapping(new_var, +1)

            new_eq = "{new} = {mix} - {mix_pre}".format(new=new_symbol, mix=mix_symbol, mix_pre=mix_pre_symbol)
            new_equations.append(new_eq)
            new_fw_variables.append(new_var)

            for j in range(len(equations)):
                equations[j] = equations[j]\
                    .replace(mix_fwd_symbol, "({new_fwd} + {mix_curr})".format(new_fwd=new_fwd_symbol, mix_curr=mix_symbol))

        return equations + new_equations, new_fw_variables

    @staticmethod
    def parse_equations_to_functional(equations, variables, shocks):
        state_variables, control_variables, mixed_variables, static_variables = EquationParser.split_variables(
            equations, variables)

        renamed_equations = EquationParser.rename_variables(equations, state_variables, control_variables, mixed_variables, static_variables)

        renamed_equations, new_fwd_var = EquationParser.remove_mixed_variables(renamed_equations, mixed_variables)

        logger.debug(renamed_equations)
        logger.debug(new_fwd_var)

        control_variables += new_fwd_var
        state_variables += mixed_variables
        mixed_variables = []

        parsed_rhs = []

        for equation in renamed_equations:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            parsed_rhs.append(p_rhs - p_lhs)

        def name_map(param, var_type):
            return EquationParser.generate_mapping(param, var_type)

        ordered_variables = [name_map(x, 0) for x in static_variables] + \
            [name_map(x, 0) for x in state_variables] + \
            [name_map(x, 0) for x in mixed_variables] + \
            [name_map(x, 0) for x in control_variables] + \
            [name_map(x, -1) for x in state_variables] + \
            [name_map(x, -1) for x in mixed_variables] + \
            [name_map(x, 1) for x in mixed_variables] + \
            [name_map(x, 1) for x in control_variables] + \
            shocks

        equation_matrix, _ = EquationParser.equations_to_matrices(parsed_rhs, ordered_variables)

        no_static = len(static_variables)
        no_state = len(state_variables)
        no_mixed = len(mixed_variables)
        no_control = len(control_variables)
        no_variables = no_static + no_state + no_mixed + no_control

        logger.debug("Variable separation: (static, state, mixed, control)")
        logger.debug(ordered_variables)
        logger.debug(static_variables)
        logger.debug(state_variables)
        logger.debug(mixed_variables)
        logger.debug(control_variables)

        f_y_plus = equation_matrix[:, (no_variables + no_state + no_mixed):(no_variables + no_state + 2 * no_mixed + no_control)]
        f_y_zero = equation_matrix[:, :no_variables]
        f_y_minus = equation_matrix[:, no_variables:(no_variables + no_state + no_mixed)]
        f_u = equation_matrix[:, (no_variables + no_state + 2 * no_mixed + no_control):]

        return f_y_plus, f_y_zero, f_y_minus, f_u, static_variables, state_variables, mixed_variables, control_variables

    @staticmethod
    def parse_equations_to_matrices(equations, variables, shocks):
        state_variables, control_variables, mixed_variables, static_variables = EquationParser.split_variables(equations, variables)

        assert len(mixed_variables) == 0, 'old model doesn\'t support mixed variables'

        renamed_equations = EquationParser.rename_variables(equations, state_variables, control_variables, mixed_variables, static_variables)

        parsed_rhs = []

        for equation in renamed_equations:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            parsed_rhs.append(p_rhs - p_lhs)

        # todo static vars get dummy param
        ordered_variables = static_variables + state_variables + \
            ['{param}_{param}'.format(param=x) for x in control_variables] + \
            ['{param}_{param}'.format(param=x) for x in static_variables] + \
            ['{param}_{param}'.format(param=x) for x in state_variables] + \
            control_variables + shocks

        equation_matrix, _ = EquationParser.equations_to_matrices(parsed_rhs, ordered_variables)

        left_state_matrix = equation_matrix[:, :len(variables)] * (-1)
        right_state_matrix = equation_matrix[:, len(variables):len(variables) * 2]
        shock_matrix = equation_matrix[:, len(variables) * 2:]

        logger.debug("PARSE EQUATIONS TO MATRICES:")
        logger.debug(pretty(static_variables))
        logger.debug(pretty(state_variables))
        logger.debug(pretty(control_variables))
        logger.debug(pretty(ordered_variables))
        logger.debug("Renamed equations:")
        logger.debug(pretty(renamed_equations))
        logger.debug("Matrices:")
        logger.debug(pretty(left_state_matrix))
        logger.debug(pretty(right_state_matrix))
        logger.debug(pretty(shock_matrix))

        return left_state_matrix, right_state_matrix, shock_matrix, state_variables, control_variables, static_variables

        # prepare equations
