from abc import ABC, abstractmethod

import numpy as np

from helper.AreaLinePlot import AreaLinePlot
import logging

log = logging.getLogger(__name__)


class ForecastData(ABC):
    def __init__(self, observable_names):
        self.observable_names = observable_names
        self.observable_len = len(observable_names)

    @abstractmethod
    def prepare_plots(self):
        pass
