import logging

import numpy as np

from util.NpUtils import to_np

logger = logging.getLogger(__name__)


def cast_to_vector(vector):
    return np.array(vector).flatten()


class PolicyFunction:
    def __init__(self, model, g_y_minus, g_y_plus, g_y_static, g_u):
        self.model = model
        self.gy_minus = np.array(g_y_minus, dtype="complex").astype(np.float32)
        self.gy_plus = np.array(g_y_plus, dtype="complex").astype(np.float32)
        self.gy_static = np.array(g_y_static, dtype="complex").astype(np.float32)
        self.gu = np.array(g_u, dtype="complex").astype(np.float32)

        self.no_static = len(model.static_vars)
        self.no_state = len(model.state_vars)
        self.no_mixed = len(model.mixed_vars)
        self.no_control = len(model.control_vars)

    def print(self):
        static_vars = self.model.static_vars
        state_vars = self.model.state_vars
        control_vars = self.model.control_vars

        policy_dict = {}

        print("Policy depends on {}".format(" ".join(["{}(-1)".format(x) for x in self.model.state_vars] + self.model.shocks)))
        for i in range(len(static_vars)):
            policy_dict[static_vars[i]] = np.concatenate((cast_to_vector(self.gy_static[i, :]), cast_to_vector(self.gu[i, :])))
        for i in range(len(state_vars)):
            policy_dict[state_vars[i]] = np.concatenate((cast_to_vector(self.gy_minus[i, :]), cast_to_vector(self.gu[i + len(static_vars), :])))
        for i in range(len(control_vars)):
            policy_dict[control_vars[i]] = np.concatenate((cast_to_vector(self.gy_plus[i, :]), cast_to_vector(self.gu[i + len(static_vars + state_vars),:])))
        for variable in self.model.ordered_variables:
            print("{param} {transition}".format(
                param=variable,
                transition=policy_dict[variable]))

    def map_to_transition(self):
        var_count = len(self.model.ordered_variables)

        transition_matrix = np.zeros((var_count, var_count))

        non_control_count = self.no_static + self.no_state

        transition_matrix[:self.no_static, self.no_static:non_control_count] = self.gy_static
        transition_matrix[self.no_static:non_control_count, self.no_static:non_control_count] = self.gy_minus
        transition_matrix[non_control_count:, self.no_static:non_control_count] = self.gy_plus

        return transition_matrix, to_np(self.gu)

    def get_start_x(self):
        return np.zeros(self.no_state + self.no_mixed)

    def get_x(self, full_vector):
        return full_vector[:self.no_state + self.no_mixed]

    def predict(self, x_curr, curr_shock):
        control_start = self.no_static + self.no_state + self.no_mixed
        all_count = self.no_static + self.no_state + self.no_mixed + self.no_control

        calc_shock = self.gu @ curr_shock

        x_next = self.gy_minus @ x_curr + calc_shock[self.no_static:control_start]
        y_next = self.gy_plus @ x_curr + calc_shock[control_start:]
        x_static = self.gy_static @ x_curr + calc_shock[:self.no_static]

        var_vector = np.zeros(all_count)

        var_vector[:self.no_static] = x_static[:]
        var_vector[self.no_static: control_start] = x_next[:]
        var_vector[control_start:] = y_next[:]

        return var_vector, x_next

