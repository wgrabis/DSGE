from sympy import Symbol
import numpy as np


class DefinitionSet:
    def __init__(self, parameters, definitions):
        print("definition-set")
        print(definitions)

        self.definitions = definitions
        self.parameters = parameters

        self.last_values = None
        self.last_calculated = None

    def __call__(self, values):
        if np.array_equiv(values, self.last_values):
            return self.last_calculated

        partial_expr = []

        for i in range(len(self.definitions)):
            name, expr = self.definitions[i]
            for j in range(len(self.parameters)):
                par_name = self.parameters[j]
                expr = expr.subs(Symbol(par_name), values[j])
            partial_expr.append((name, expr))

        print("calculated-expression-partial")
        print(partial_expr)

        calculated_expr = []

        for i in range(len(self.definitions)):
            name, value = partial_expr[i]

            symbol_name = Symbol(str(name))

            calculated_expr.append((symbol_name, value))
            for j in range(i, len(self.definitions)):
                partial_name, expr = partial_expr[j]

                expr = expr.subs(symbol_name, value)

                partial_expr[j] = (partial_name, expr)

        print("calculated-expression")
        print(calculated_expr)

        return calculated_expr

