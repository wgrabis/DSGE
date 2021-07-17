from sympy import sympify, symbols, linear_eq_to_matrix, pprint


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

        print(result)

        return result

    @staticmethod
    def equations_to_matrices(equations, variables):
        print("Eq to matrice")
        print(equations)
        print(variables)
        left, right = linear_eq_to_matrix(equations, symbols(" ".join(variables)))
        pprint(left, wrap_line=False)
        pprint(right, wrap_line=False)

        return left, right

    @staticmethod
    def split_variables(equations, variables):
        state_variables = set()
        control_variables = set()

        for equation in equations:
            for variable in variables:
                if equation.find('{}(+1)'.format(variable)) != -1:
                    control_variables.add(variable)
                if equation.find('{}(-1)'.format(variable)) != -1:
                    state_variables.add(variable)

        print("split_variables")
        print(state_variables)
        print(control_variables)

        assert len(state_variables.intersection(control_variables)) == 0

        state_variables = state_variables.union(set(variables).difference(control_variables))

        return list(state_variables), list(control_variables)

    @staticmethod
    def rename_variables(equations, state_variables, control_variables):
        renamed_equations = []

        def name_mapping(param, is_prev):
            if is_prev:
                return '{param}_{param}'.format(param=param)
            else:
                return param

        for equation in equations:
            result = equation

            for parameter in state_variables:
                result = result.replace('{}(-1)'.format(parameter), name_mapping(parameter, 1))
                result = result.replace(parameter, name_mapping(parameter, 0))

            for parameter in control_variables:
                result = result.replace('{}(+1)'.format(parameter), name_mapping(parameter, 1))
                result = result.replace(parameter, name_mapping(parameter, 0))

            renamed_equations.append(result)

        return renamed_equations

    @staticmethod
    def parse_equations_to_matrices(equations, variables, shocks):
        state_variables, control_variables = EquationParser.split_variables(equations, variables)

        renamed_equations = EquationParser.rename_variables(equations, state_variables, control_variables)

        parsed_rhs = []

        for equation in renamed_equations:
            p_lhs, p_rhs = EquationParser.build_equation(equation)
            parsed_rhs.append(p_rhs - p_lhs)

        ordered_variables = state_variables + ['{param}_{param}'.format(param=x) for x in control_variables] + \
            ['{param}_{param}'.format(param=x) for x in state_variables] + control_variables + shocks

        equation_matrix, _ = EquationParser.equations_to_matrices(parsed_rhs, ordered_variables)

        left_state_matrix = equation_matrix[:, :len(variables)] * (-1)
        right_state_matrix = equation_matrix[:, len(variables):len(variables) * 2]
        shock_matrix = equation_matrix[:, len(variables) * 2:]

        print("PARSE EQUATIONS TO MATRICES:")
        pprint(state_variables)
        pprint(control_variables)
        pprint(ordered_variables)
        print("Renamed equations:")
        pprint(renamed_equations)
        print("Matrices:")
        pprint(left_state_matrix)
        pprint(right_state_matrix)
        pprint(shock_matrix)

        return left_state_matrix, right_state_matrix, shock_matrix, state_variables, control_variables

        # prepare equations
