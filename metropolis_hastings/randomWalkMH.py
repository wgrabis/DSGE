from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.metropolisHastings import MetropolisHastings


class RandomWalkMH(MetropolisHastings):
    def __init__(self, rounds, model):
        self.likelihood_algorithm = LikelihoodAlgorithm()
        super().__init__(rounds, model)

    def draw_posterior(self):
        pass

    def accept(self, draw):
        pass

    def get_starting_posterior(self):
        pass