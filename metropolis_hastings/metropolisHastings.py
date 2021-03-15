from abc import ABC, abstractmethod


class MetropolisHastings(ABC):
    def __init__(self, rounds, model):
        self.rounds = rounds
        self.current_posterior = None
        self.model = model

    @abstractmethod
    def draw_posterior(self):
        pass

    @abstractmethod
    def accept(self, draw):
        pass

    @abstractmethod
    def get_starting_posterior(self):
        pass


    def calculate_posterior(self):
        self.current_posterior = self.get_starting_posterior()
        for i in range(1, self.rounds):
            draw = self.draw_posterior()
            if self.accept(draw):
                self.current_posterior = draw

        return self.current_posterior
