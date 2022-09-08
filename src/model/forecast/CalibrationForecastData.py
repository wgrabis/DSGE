import logging

from helper.AreaLinePlot import AreaLinePlot
from helper.LinePlot import LinePlot
from model.forecast.ForecastData import ForecastData

log = logging.getLogger(__name__)


class CalibrationForecastData(ForecastData):
    def __init__(self, forecast_data, observable_names):
        super().__init__(observable_names, forecast_data)

    def prepare_plots(self):
        forecast_time = self.time_len

        data_x = []
        plot_data = [[] for _ in range(self.observable_len)]

        for i in range(forecast_time):
            data_x.append(i)
            for j in range(self.observable_len):
                plot_data[j].append(self.main_forecast[i][j])

        plots = []

        log.debug("Prepare plots")
        log.debug(plot_data)
        log.debug(self.main_forecast)

        for i in range(self.observable_len):
            plots.append(
                LinePlot(self.observable_names[i], data_x, plot_data[i], "time", ""))

        return plots

