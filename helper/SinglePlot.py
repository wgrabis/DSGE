from helper.BasePlot import BasePlot


class SinglePlot(BasePlot):
    def __init__(self, name, data, x_name, y_name):
        self.name = name
        self.data = data
        self.x_name = x_name
        self.y_name = y_name

    def draw_plot(self, axs):
        axs.plot()
