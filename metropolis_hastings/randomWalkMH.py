from random import random
from scipy import stats
import numpy as np

from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.metropolisHastings import MetropolisHastings


class RandomWalkMH(MetropolisHastings):
    def __init__(self, rounds, model, data, with_start=None):
        self.data = data
        self.likelihood_algorithm = LikelihoodAlgorithm()

        self.c = 0.234

        self.walk_covariance = with_start

        if with_start is None:
            self.walk_covariance = np.identity(len(model.structural))

        super().__init__(rounds, model)

    def draw_posterior(self, current_draw):

        probability_covariance = self.walk_covariance * (self.c * self.c)
        # next_distribution = stats.norm(current_draw, self.c * self.c * self.var_c)
        print("draw")
        print(probability_covariance)
        print(current_draw)
        return np.random.multivariate_normal(current_draw, probability_covariance)

    def accept(self, current_draw, draw):
        print("accept")
        print(current_draw)
        print(draw)
        draw_likelihood = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, draw)
        current_likelihood = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, current_draw)

        roll = random()

        return roll <= draw_likelihood/current_likelihood

    def get_starting_posterior(self):
        return self.model.get_prior_posterior()
