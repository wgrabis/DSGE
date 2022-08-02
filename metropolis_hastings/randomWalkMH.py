import logging
from random import random
from scipy import stats
import numpy as np

from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.metropolisHastings import MetropolisHastings

logger = logging.getLogger(__name__)


class RandomWalkMH(MetropolisHastings):
    def __init__(self, rounds, model, data, with_covariance=None):
        self.likelihood_algorithm = LikelihoodAlgorithm()

        self.c = 0.234

        self.walk_covariance = with_covariance

        if with_covariance is None:
            self.walk_covariance = np.identity(len(model.structural))

        super().__init__(rounds, model, data)

    def draw_posterior(self, current_draw):

        probability_covariance = self.walk_covariance * (self.c * self.c)
        # next_distribution = stats.norm(current_draw, self.c * self.c * self.var_c)
        logger.debug("draw")
        logger.debug(probability_covariance)
        logger.debug(current_draw)
        return np.random.multivariate_normal(current_draw, probability_covariance)

    def accept(self, current_draw, draw):
        logger.debug("accept")
        logger.debug(current_draw)
        logger.debug(draw)
        draw_likelihood, distribution = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, draw)
        current_likelihood, _ = self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, current_draw)

        roll = random()

        logger.info("Roll accept")
        logger.info(current_draw)
        logger.info(draw)
        logger.info(draw_likelihood - current_likelihood)
        logger.info("Roll value:")
        logger.info(current_likelihood)
        logger.info(draw_likelihood)
        logger.info(draw_likelihood - current_likelihood)

        if draw_likelihood > current_likelihood:
            return True, distribution, draw_likelihood

        draw_probability = np.exp2(draw_likelihood - current_likelihood)

        logger.info("Roll probability")
        logger.info(draw_probability)
        logger.info(roll)
        logger.info("Success:")
        logger.info(roll <= draw_probability)

        # draw_likelihood/current_likelihood
        return roll <= draw_probability, distribution, draw_likelihood

    def get_starting_posterior(self):
        return self.model.get_prior_posterior()
