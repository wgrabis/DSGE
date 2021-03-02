from helper.BasePlot import BasePlot


class MultiPlot(BasePlot):
    def __init__(self, name, data_x, data_y, x_name, y_name):
        self.name = name
        self.data_x = data_x
        self.data_y = data_y
        self.x_name = x_name
        self.y_name = y_name

    def draw_plot(self, axs):
        count = len(self.data_x)
        for i in range(0, count):
            axs.plot(self.data_x[i], self.data_y[i])

        axs.set_title(self.name)
        axs.set(xlabel=self.x_name, ylabel=self.y_name)