import numpy as np


class RandomParameter:
    def __init__(self, name, lower_bound=None, upper_bound=None):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def scale_variable(self, value, for_covariance=False):
        if for_covariance and (value == self.lower_bound or value == self.upper_bound):
            return value

        if self.lower_bound is None and self.upper_bound is not None:
            return np.log(-value + self.upper_bound)

        if self.lower_bound is not None and self.upper_bound is None:
            return np.log(value - self.lower_bound)

        if self.lower_bound is not None and self.upper_bound is not None:
            return np.log((self.upper_bound - value)/(value - self.lower_bound))

        return value

    def rescale_variable(self, seed):
        if self.lower_bound is None and self.upper_bound is not None:
            return self.upper_bound - np.exp(seed)

        if self.lower_bound is not None and self.upper_bound is None:
            return self.lower_bound + np.exp(seed)

        if self.lower_bound is not None and self.upper_bound is not None:
            return (self.upper_bound + np.exp(seed) * self.lower_bound)/(1 + np.exp(seed))

        return seed