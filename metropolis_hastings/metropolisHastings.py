from abc import ABC, abstractmethod

from forecast.ForecastAlgorithm import ForecastAlgorithm
from helper.DataHistory import DataHistory


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
        data_history = DataHistory()
        current_posterior = self.get_starting_posterior()
        print("mh-start")
        print(current_posterior)
        data_history.add_record(current_posterior)
        final_distribution = None
        for i in range(1, self.rounds):
            draw = self.draw_posterior(current_posterior)
            print("mh-draw")
            print(draw)

            accepted, distribution = self.accept(current_posterior, draw)

            if accepted:
                final_distribution = distribution
                print("mh-accepted")
                print(draw)
                print(current_posterior)
                data_history.add_record(draw)
                current_posterior = draw

        forecast_algorithm = ForecastAlgorithm(self.model)

        forecasted = forecast_algorithm.calculate(current_posterior, 50, 10, self.data.estimation_time, self.data, final_distribution)

        return current_posterior, data_history, forecasted
