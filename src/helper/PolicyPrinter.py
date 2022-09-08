from sympy import pprint


class PolicyPrinter:
    @staticmethod
    def print(model, g_y_plus, g_y_minus, g_u):
        variables = model.variables

        print("Policy Function:")
        print(variables)
        pprint(g_y_minus, wrap_line=False)
        pprint(g_y_plus, wrap_line=False)
        print("Shock")
        pprint(g_u, wrap_line=False)