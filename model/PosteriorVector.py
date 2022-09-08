import numpy as np


class PosteriorVector:
    def __init__(self, seed, structural_set, pre_seed=None):
        self.seed = seed
        self.pre_seed = pre_seed
        self.structural_set = structural_set
        self.full_vector = None

    def get_full_vector(self):
        if self.pre_seed is None:
            self.pre_seed = self.structural_set.rescale_seed(self.seed)
        if self.full_vector is None:
            self.full_vector = self.structural_set.get_bound_posterior_vector(self.pre_seed)
        return self.full_vector

    def get_seed(self):
        return self.seed
