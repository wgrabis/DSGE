

class DsgeModel:
    def __init__(self, filter, state_transition, state_shock, measure_base, measure_time, measure_state, measure_error):
        self.filter = filter
        self.state_transition = state_transition
        self.state_shock = state_shock
        self.measure_base = measure_base
        self.measure_time = measure_time
        self.measure_state = measure_state
        self.measure_error = measure_error

    def likelihood_function(self):
        # apply kalman filter
        pass