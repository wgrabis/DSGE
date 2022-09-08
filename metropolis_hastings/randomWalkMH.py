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
            self.walk_covariance = np.identity(model.structural_prior.random_len)

        super().__init__(rounds, model, data)

    def draw_posterior(self, iteration, current_draw):
        if iteration % 100 == 0:
            self.c = self.c * 0.75

        probability_covariance = self.walk_covariance * (self.c * self.c)

        logger.debug("draw")
        logger.debug(probability_covariance)

        random_seed = current_draw.get_seed()

        logger.debug(random_seed)
        next_seed = np.random.multivariate_normal(random_seed, probability_covariance)

        # random_part = self.model.structural_prior.get_random_part(current_draw)
        # next_value = np.random.multivariate_normal(random_part, probability_covariance)
        #
        # while not self.model.structural_prior.check_bounds(next_value):
        #     next_value = np.random.multivariate_normal(random_part, probability_covariance)

        logger.debug(current_draw)
        logger.debug(next_seed)

        return self.model.structural_prior.get_move_vector(next_seed)

    def accept(self, current_draw, current_value, draw):
        logger.info("accept")
        logger.info(current_draw.get_full_vector())
        logger.info(draw.get_full_vector())

        try:
            draw_likelihood, distribution = \
                self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, draw.get_full_vector())
        except:
            return False, None, 0

        roll = random()

        logger.info("Roll accept")
        logger.info(draw_likelihood - current_value)
        logger.info("Roll value:")
        logger.info(current_value)
        logger.info(draw_likelihood)

        if draw_likelihood > current_value:
            return True, distribution, draw_likelihood

        draw_probability = np.exp2(draw_likelihood - current_value)

        logger.info("Roll probability")
        logger.info(draw_probability)
        logger.info(roll)
        logger.info("Success:")
        logger.info(roll <= draw_probability)

        # draw_likelihood/current_likelihood
        return roll <= draw_probability, distribution, draw_likelihood

    def get_starting_posterior(self):
        prior_vector = self.model.structural_prior.get_prior_vector()

        likelihood, distribution = \
            self.likelihood_algorithm.get_likelihood_probability(self.model, self.data, prior_vector.get_full_vector())

        return prior_vector, distribution, likelihood
