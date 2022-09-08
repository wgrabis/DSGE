from abc import ABC, abstractmethod


class BasePlot(ABC):
    def __init__(self, name, file_name=None):
        self.name = name
        self.file_name = name if file_name is None else file_name

    @abstractmethod
    def draw_plot(self, axs):
        pass

