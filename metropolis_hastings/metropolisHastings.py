from abc import ABC, abstractmethod


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
        current_posterior = self.get_starting_posterior()
        for i in range(1, self.rounds):
            draw = self.draw_posterior(current_posterior)
            if self.accept(current_posterior, draw):
                current_posterior = draw

        return current_posterior
