import numpy as np

from helper.AreaLinePlot import AreaLinePlot
import logging

log = logging.getLogger(__name__)


class ForecastData:
    def __init__(self, estimations):
        self.estimations = estimations

        self.forecasts = []
        self.bottom_observables = []
        self.top_observables = []
        self.main_forecast = None
        self.setup = False

        self.time_len = 0
        self.observable_len = 0

    def setup_data(self, observables):
        if self.setup:
            return

        self.time_len = len(observables)
        self.observable_len = len(observables[0])

        self.bottom_observables = [[np.Inf for _ in range(self.time_len)] for _ in range(self.observable_len)]
        self.top_observables = [[np.NINF for _ in range(self.time_len)] for _ in range(self.observable_len)]

        self.setup = True

    def add_main_forecast(self, observables):
        self.setup_data(observables)

        self.main_forecast = observables

    def add_posterior_forecast(self, observables):
        self.setup_data(observables)

        self.forecasts.append(observables)

        for i in range(self.time_len):
            for j in range(self.observable_len):
                self.update_observable(i, j, observables[i][j])

    def update_observable(self, time, observable, value):
        log.debug("Update observables")
        log.debug(time)
        log.debug(observable)
        log.debug(value)
        log.debug(self.bottom_observables[observable][time])
        log.debug(self.top_observables[observable][time])

        self.bottom_observables[observable][time] = min(self.bottom_observables[observable][time], value)
        self.top_observables[observable][time] = max(self.top_observables[observable][time], value)

    def prepare_plots(self):
        estimation_time = self.estimations.estimation_time

        data_x =[]
        top_plot = [[] for _ in range(self.observable_len)]
        bottom_plot = [[] for _ in range(self.observable_len)]
        average_plot = [[] for _ in range(self.observable_len)]

        for i in range(estimation_time):
            # todo add description of x
            data_x.append(i)
            estimations = self.estimations[i]
            for j in range(self.observable_len):
                top_plot[j].append(estimations[j])
                bottom_plot[j].append(estimations[j])
                average_plot[j].append(estimations[j])

        forecast_time = self.time_len

        for i in range(forecast_time):
            data_x.append(estimation_time + i)
            for j in range(self.observable_len):
                if self.top_observables[j][i] == np.NINF:
                    top_plot[j].append(self.main_forecast[i][j])
                else:
                    top_plot[j].append(self.top_observables[j][i])

                if self.top_observables[j][i] == np.inf:
                    bottom_plot[j].append(self.main_forecast[i][j])
                else:
                    bottom_plot[j].append(self.bottom_observables[j][i])
                average_plot[j].append(self.main_forecast[i][j])

        plots = []

        log.debug("Prepare plots")
        log.debug(average_plot)
        log.debug(self.main_forecast)

        for i in range(self.observable_len):
            plots.append(
                AreaLinePlot("", data_x, average_plot[i], average_plot[i], average_plot[i], "", ""))
            # plots.append(AreaLinePlot("observ %i" % i, data_x, top_plot[i], bottom_plot[i], average_plot[i], "time", "value"))

        return plots
