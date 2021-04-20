from abc import ABC, abstractmethod

from helper.DataHistory import DataHistory


class MetropolisHastings(ABC):
    def __init__(self, rounds, model):
        self.rounds = rounds
        self.model = model

    @abstractmethod
    def draw_posterior(self, current_posterior):
        pass

    @abstractmethod
    def accept(self, current_draw, draw):
        pass

    @abstractmethod
    def get_starting_posterior(self):
        pass

    def calculate_posterior(self):
        data_history = DataHistory()
        current_posterior = self.get_starting_posterior()
        print("mh-start")
        print(current_posterior)
        data_history.add_record(current_posterior)
        for i in range(1, self.rounds):
            draw = self.draw_posterior(current_posterior)
            print("mh-draw")
            print(draw)
            if self.accept(current_posterior, draw):
                print("mh-accepted")
                print(draw)
                print(current_posterior)
                data_history.add_record(draw)
                current_posterior = draw

        return current_posterior, data_history
