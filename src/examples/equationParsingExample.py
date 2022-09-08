from sympy import symbols, linear_eq_to_matrix, Function, Symbol, sympify
from sympy.printing.dot import dotprint
import numpy as np

from sympy.parsing.sympy_parser import parse_expr

from model.DsgeModelBuilder import DsgeModelBuilder
from model.Equation import EquationParser


def test_equations():
    equations = [
        "w(t) = 400*z*b + 100*t + 100",
        "y(t) = 300*x*w + 10*t + 50*v",
        "p(t) = 1*z*d + 40*v*d",
    ]

    structural = ["x", "z"]

    equation_builder = EquationParser()

    x, z, v = symbols('x z v')
    t = symbols('t')
    lhs, rhs = [], []
    for equation in equations:
        p_lhs, p_rhs = EquationParser.build_equation(equation)
        lhs.append(p_lhs)
        rhs.append(p_rhs)

    print(lhs)
    print(rhs)

    left, right = EquationParser.equations_to_matrices(rhs, ["x", "z", "v"])
    # # print(rhs.subs({t: 10}))
    #
    # t_matrix = left.col(-1)
    # structural_matrix = left.col_del(-1)
    # w, d, b = symbols('w d b')
    # print(t_matrix)
    # print(left.subs({w: 10, d: 5, b: 1}))
    # print(left)
    print(left)
    print(right)
    print(-1*right)

    m1, m2, m3 = DsgeModelBuilder.prepare_measurement_matrices(
        equations, ["x", "z", "v"], ["b", "w", "d"]
    )

    print("---")
    print(m1)
    print(m2)
    print(m3)

    print(m1(np.array([1, 1, 1])))
    print(m2(np.array([1, 1, 1])))
    print(m3(np.array([1, 1, 1])))


def test_equations2():
    equations = [
        "x = a*x(-1) + e",
        "y = b*y(-1) + c*x(-1) + d*p(-1)",
        "p = p(-1) + f",
    ]

    structural = ["a", "b", "c", "d"]
    shocks = ["e", "f"]

    print(sympify("f(1)"))
    print(dotprint(sympify("f(1)")))

    lhs, rhs = [], []
    for equation in equations:
        s_lhs, s_rhs = equation.split("=")

        p_lhs = sympify(s_lhs)
        p_rhs = sympify(s_rhs)

        print(dotprint(p_rhs))

        lhs.append(p_lhs)
        rhs.append(p_rhs)

    print(lhs)
    print(rhs)

    x, y, p = symbols('x y z')

    left, right = linear_eq_to_matrix(rhs, [x, y, p])

    print(left)
    print(right)

    transition, shock = DsgeModelBuilder.prepare_state_matrices(equations, ['e', 'f'], ['x', 'y', 'p'])

    print("State matrices")

    print(transition)
    print(shock)
