import numpy as np
from sympy import symbols, Symbol, pprint, pretty

import logging

log = logging.getLogger(__name__)


def calculate(scipy_item, parameters, values, definitions):
    subs = {}
    expr = scipy_item

    log.debug("calculate")
    # log.debug(pretty(scipy_item))
    log.debug(pretty(parameters))
    log.debug(pretty(values))

    for i in range(len(parameters)):
        expr = expr.subs(Symbol(parameters[i]), values[i])
        # subs[symbols(parameters[i])] = values[i]

    log.debug("calculate-definitions")
    # log.debug(pretty(expr))
    log.debug(pretty(definitions))

    for (name, value) in definitions:
        expr = expr.subs(name, value)

    # log.debug(pretty(expr))

    return expr


class VariableMatrix:
    def __init__(self, matrix, parameters, definition_set):
        self.matrix = matrix
        self.parameters = parameters
        self.definition_set = definition_set

    def __call__(self, values):
        log.debug("Variable-matrix")
        log.debug(self.matrix)
        log.debug(values)

        valued_matrix = np.array(calculate(self.matrix, self.parameters, values, self.definition_set(values)), dtype='float')

        log.debug(valued_matrix)

        return valued_matrix

    def print(self):
        pprint(self.matrix, wrap_line=False)


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

    def print(self):
        pprint(self.vector, wrap_line=False)


class CompMultiVariableMatrix:
    def __init__(self, variable_matrices, computation):
        self.variable_matrices = variable_matrices
        self.computation = computation

    def __call__(self, values):
        matrices = [var_matrix(values) for var_matrix in self.variable_matrices]

        return self.computation(matrices)

    def print(self):
        for var_matrix in self.variable_matrices:
            var_matrix.print()


class CompVariableMatrix:
    def __init__(self, variable_matrix, computation):
        self.variable_matrix = variable_matrix
        self.computation = computation

    def __call__(self, values):
        matrix = self.variable_matrix(values)

        return self.computation(matrix)

    def print(self, description):
        self.variable_matrix.print()
