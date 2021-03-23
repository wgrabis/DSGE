import numpy as np
from sympy import symbols, Symbol


def calculate(scipy_item, parameters, values):
    subs = {}
    expr = scipy_item
    for i in range(len(parameters)):
        expr = expr.subs(Symbol(parameters[i]), values[i])
        # subs[symbols(parameters[i])] = values[i]

    return expr


class VariableMatrix:
    def __init__(self, matrix, parameters):
        self.matrix = matrix
        self.parameters = parameters

    def __call__(self, values):
        return np.matrix(calculate(self.matrix, self.parameters, values), dtype='float')


class VariableVector:
    def __init__(self, vector, parameters):
        self.vector = vector
        self.parameters = parameters

    def __call__(self, values):
        return np.array(calculate(self.vector, self.parameters, values), dtype='float')


class CompVariableMatrix:
    def __init__(self, variable_matrix, computation):
        self.variable_matrix = variable_matrix
        self.computation = computation

    def __call__(self, values):
        matrix = self.variable_matrix(values)

        return self.computation(matrix)
