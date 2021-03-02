from abc import ABC, abstractmethod


class BasePlot(ABC):
    @abstractmethod
    def draw_plot(self, axs):
        pass
