from abc import ABC, abstractmethod

import numpy as np

from helper.AreaLinePlot import AreaLinePlot
import logging

log = logging.getLogger(__name__)


class ForecastData(ABC):
    def __init__(self, observable_names, main_data):
        self.observable_names = observable_names
        self.observable_len = len(observable_names)

        self.main_forecast = main_data
        self.time_len = len(main_data)

    @abstractmethod
    def prepare_plots(self):
        pass
