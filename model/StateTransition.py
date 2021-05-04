import numpy as np


class StateTransition:
    def __init__(self, model, posterior):
        transition, shock = model.build_matrices(posterior)

        self.transition_matrix = transition
        self.shock_matrix = shock

    def __call__(self, x_k, shocks):
        x_next = np.dot(self.transition_matrix, x_k) + np.dot(self.shock_matrix, shocks)

        return x_next
