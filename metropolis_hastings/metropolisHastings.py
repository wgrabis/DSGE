import logging
from abc import ABC, abstractmethod

from forecast.ForecastAlgorithm import ForecastAlgorithm
from helper.DataHistory import DataHistory
from model.PosteriorStory import PosteriorStory

logger = logging.getLogger(__name__)


class MetropolisHastings(ABC):
    def __init__(self, rounds, model, data):
        self.rounds = rounds
        self.model = model
        self.data = data

    @abstractmethod
    def draw_posterior(self, iteration, current_posterior):
        pass

    @abstractmethod
    def accept(self, current_draw, current_value, draw):
        pass

    @abstractmethod
    def get_starting_posterior(self):
        pass

    # todo refactor for forecasting
    def calculate_posterior(self, story=None, history=None):
        data_history = DataHistory(self.model.structural_prior.ordered_params, self.model.structural_prior.name_map) if history is None else history
        start_posterior, start_distribution, start_value = self.get_starting_posterior()
        posteriors = PosteriorStory() if story is None else story

        start_posterior_full = start_posterior.get_full_vector()

        posteriors.add(start_posterior_full, start_distribution, start_value)
        data_history.add_record(start_posterior_full)

        current_posterior = start_posterior
        current_value = start_value
        current_distribution = start_distribution

        logger.debug("mh-start")
        logger.debug(start_posterior_full)

        for i in range(1, self.rounds):
            draw = self.draw_posterior(i, current_posterior)

            logger.info("Iteration " + str(i))
            logger.info(draw)

            logger.debug("mh-draw")
            logger.debug(draw)

            accepted, distribution, value = self.accept(current_posterior, current_value, draw)

            if accepted:
                draw_vector = draw.get_full_vector()

                logger.debug("mh-accepted")
                logger.debug(draw_vector)
                logger.debug(current_posterior.get_full_vector())

                current_posterior = draw
                current_value = value
                current_distribution = distribution

                data_history.add_record(draw_vector)
                posteriors.add(draw_vector, distribution, value)

        return posteriors, data_history, (current_distribution, current_value)
