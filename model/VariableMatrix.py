import numpy as np
from sympy import symbols, Symbol, pprint


def calculate(scipy_item, parameters, values, definitions):
    subs = {}
    expr = scipy_item

    print("calculate")
    pprint(scipy_item, wrap_line=False)
    pprint(parameters, wrap_line=False)
    pprint(values, wrap_line=False)

    for i in range(len(parameters)):
        expr = expr.subs(Symbol(parameters[i]), values[i])
        # subs[symbols(parameters[i])] = values[i]

    print("calculate-definitions")
    pprint(expr, wrap_line=False)
    pprint(definitions, wrap_line=False)

    for (name, value) in definitions:
        expr = expr.subs(name, value)

    pprint(expr, wrap_line=False)

    return expr


class VariableMatrix:
    def __init__(self, matrix, parameters, definition_set):
        self.matrix = matrix
        self.parameters = parameters
        self.definition_set = definition_set

    def __call__(self, values):
        print("Variable-matrix")
        print(self.matrix)
        print(values)
        valued_matrix = np.array(calculate(self.matrix, self.parameters, values, self.definition_set(values)), dtype='float')

        print(valued_matrix)

        return valued_matrix


class VariableVector:
    def __init__(self, vector, parameters, definition_set):
        self.vector = vector
        self.parameters = parameters
        self.definition_set = definition_set

    def __call__(self, values):
        valued_vector = np.array(calculate(self.vector, self.parameters, values, self.definition_set(values)), dtype='float')

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
