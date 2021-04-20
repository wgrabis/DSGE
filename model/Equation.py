from sympy import sympify, symbols, linear_eq_to_matrix


class EquationParser:
    @staticmethod
    def build_equation(equation):
        lhs, rhs = equation.split("=")

        print("parsing")
        print(lhs)
        print(rhs)

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
        print(left)
        print(right)

        return left, right
