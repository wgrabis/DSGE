import logging

from helper.LinePlot import LinePlot

logger = logging.getLogger(__name__)


# todo add common ancestor class hierarchy
class DataHistory:
    def __init__(self, structural_params):
        self.iterations = 0
        self.data = []
        self.structural_params = structural_params

    def prepare_plots(self, iter_start=0, iter_end=-1):
        if iter_end == -1:
            iter_end = self.iterations

        structural_len = len(self.structural_params)

        x_plot_data = []
        y_plot_data = [[] for _ in range(structural_len)]

        for i in range(iter_start, iter_end):
            x_plot_data.append(i)
            for j in range(structural_len):
                y_plot_data[j].append(self.data[i][j])

        plots = []

        logger.debug("Prepare plots")
        logger.debug(y_plot_data)
        logger.debug(self.data)

        for i in range(structural_len):
            plots.append(
                LinePlot("$" + self.structural_params[i] + "$", x_plot_data, y_plot_data[i], "time", ""))

        return plots

    def add_record(self, record):
        self.data.append(record)
        self.iterations += 1
