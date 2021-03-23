from random import random

from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.metropolisHastings import MetropolisHastings


class RandomWalkMH(MetropolisHastings):
    def __init__(self, rounds, model, data):
        self.data = data
        self.likelihood_algorithm = LikelihoodAlgorithm()
        super().__init__(rounds, model)

    def draw_posterior(self, current_draw):
        #todo
        pass

    def accept(self, current_draw, draw):
        draw_likelihood = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, draw)
        current_likelihood = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, current_draw)

        roll = random()

        return roll <= min(draw_likelihood/current_likelihood, 1)

    def get_starting_posterior(self):
        return self.model.get_prior_posterior()
