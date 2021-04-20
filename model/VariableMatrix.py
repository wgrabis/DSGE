import numpy as np
from sympy import symbols, Symbol


def calculate(scipy_item, parameters, values):
    subs = {}
    expr = scipy_item

    print("calculate")
    print(scipy_item)
    print(parameters)
    print(values)

    for i in range(len(parameters)):
        expr = expr.subs(Symbol(parameters[i]), values[i])
        # subs[symbols(parameters[i])] = values[i]

    return expr


class VariableMatrix:
    def __init__(self, matrix, parameters):
        self.matrix = matrix
        self.parameters = parameters

    def __call__(self, values):
        valued_matrix = np.array(calculate(self.matrix, self.parameters, values), dtype='float')

        print("Variable-matrix")
        print(valued_matrix)

        return valued_matrix


class VariableVector:
    def __init__(self, vector, parameters):
        self.vector = vector
        self.parameters = parameters

    def __call__(self, values):
        valued_vector = np.array(calculate(self.vector, self.parameters, values), dtype='float')

        # if len(valued_vector.shape) > 1 and valued_vector.shape[1] == 1:
        #     valued_vector = valued_vector.T

        # assert valued_vector.shape == (len(self.vector), )

        return valued_vector.reshape((valued_vector.shape[0], ))


class CompVariableMatrix:
    def __init__(self, variable_matrix, computation):
        self.variable_matrix = variable_matrix
        self.computation = computation

    def __call__(self, values):
        matrix = self.variable_matrix(values)
        print("comp-matrix")
        print(matrix)
        print(matrix.shape)

        return self.computation(matrix)
