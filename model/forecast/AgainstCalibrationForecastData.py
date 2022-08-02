import logging

from helper.LineWithTargetPlot import LineWithTargetPlot
from model.forecast.ForecastData import ForecastData

log = logging.getLogger(__name__)


class AgainstCalibrationForecastData(ForecastData):
    def __init__(self, forecast_data, other_forecast_data):
        super().__init__(forecast_data.observable_names, forecast_data.main_forecast)
        self.against_data = other_forecast_data

    def prepare_plots(self):
        forecast_time = self.time_len

        other_observable_names = self.against_data.observable_names
        other_forecast = self.against_data.main_forecast

        observable_id_mapping = {}

        for j in range(self.observable_len):
            p_obs = self.observable_names[j]
            ids = [i for i, x in enumerate(other_observable_names) if x == p_obs]
            assert len(ids) == 1
            observable_id_mapping[j] = ids[0]

        data_x = []
        plot_data = [[] for _ in range(self.observable_len)]
        against_data = [[] for _ in range(self.observable_len)]

        for i in range(forecast_time):
            data_x.append(i)
            for j in range(self.observable_len):
                plot_data[j].append(self.main_forecast[i][j])
                against_data[j].append(other_forecast[i][observable_id_mapping[j]])

        plots = []

        log.debug("Prepare plots")
        log.debug(plot_data)
        log.debug(self.main_forecast)

        for i in range(self.observable_len):
            plots.append(
                LineWithTargetPlot(self.observable_names[i], data_x, plot_data[i], against_data[i], "time", ""))

        return plots
