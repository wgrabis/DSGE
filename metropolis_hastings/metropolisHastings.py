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
    def draw_posterior(self, current_posterior):
        pass

    @abstractmethod
    def accept(self, current_draw, draw):
        pass

    @abstractmethod
    def get_starting_posterior(self):
        pass

    # todo refactor for forecasting
    def calculate_posterior(self):
        data_history = DataHistory(self.model.structural_prior.structural)
        current_posterior = self.get_starting_posterior()
        logger.debug("mh-start")
        logger.debug(current_posterior)
        data_history.add_record(current_posterior)
        any_accepted = False

        posteriors = PosteriorStory()

        for i in range(1, self.rounds):
            draw = self.draw_posterior(current_posterior)
            logger.debug("mh-draw")
            logger.debug(draw)

            accepted, distribution, value = self.accept(current_posterior, draw)

            if accepted:
                logger.debug("mh-accepted")
                logger.debug(draw)
                logger.debug(current_posterior)
                data_history.add_record(draw)
                current_posterior = draw

                posteriors.add(draw, distribution, value)

            if not any_accepted:
                if not accepted:
                    data_history.add_record(draw)
                    current_posterior = draw

                    posteriors.add(draw, distribution, 0)
                any_accepted = True

        return posteriors, data_history
