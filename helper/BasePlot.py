from abc import ABC, abstractmethod


class BasePlot(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def draw_plot(self, axs):
        pass

