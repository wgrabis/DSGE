import numpy as np


def cast_to_vector(vector):
    return np.array(vector).flatten()


class PolicyFunction:
    def __init__(self, model, g_y_minus, g_y_plus, g_y_static, g_u):
        self.model = model
        self.gy_minus = np.array(g_y_minus, dtype="complex").astype(np.float32)
        self.gy_plus = np.array(g_y_plus, dtype="complex").astype(np.float32)
        self.gy_static = np.array(g_y_static, dtype="complex").astype(np.float32)
        self.gu = np.array(g_u, dtype="complex").astype(np.float32)

        print("policy function")
        print("GU")
        print(self.gu)

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

    def gen_shock(self, time):
        if time == 0:
            return self.model.shock_prior.get_mean()
        return np.zeros(len(self.model.shocks))

    def predict(self, time):
        no_static = len(self.model.static_vars)
        no_state = len(self.model.state_vars)
        no_mixed = len(self.model.mixed_vars)
        no_control = len(self.model.control_vars)

        x_curr = np.zeros(no_state + no_mixed)

        calc_vectors = []

        for i in range(time):
            curr_shock = self.gen_shock(i)

            calc_shock = self.gu @ curr_shock

            x_next = self.gy_minus @ x_curr + calc_shock[no_static:no_static + no_state + no_mixed]
            y_next = self.gy_plus @ x_curr + calc_shock[no_static + no_state + no_mixed:]
            x_static = self.gy_static @ x_curr + calc_shock[:no_static]

            var_vector = np.zeros(no_static + no_state + no_mixed + no_control)

            var_vector[:no_static] = x_static[:]
            var_vector[no_static: no_static + no_state + no_mixed] = x_next[:]
            var_vector[no_static + no_state + no_mixed:] = y_next[:]

            x_curr = x_next

            # print(var_vector)

            calc_vectors.append(var_vector)

        return calc_vectors
