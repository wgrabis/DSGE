from helper.BasePlot import BasePlot


class AreaLinePlot(BasePlot):
    def __init__(self, name, data_x, top_y, bottom_y, average_y, x_name, y_name):
        self.name = name
        self.data_x = data_x
        self.top = top_y
        self.bottom = bottom_y
        self.average = average_y
        self.x_name = x_name
        self.y_name = y_name

    def draw_plot(self, axs):
        # todo bugged
        # axs.fill_between(self.data_x, self.bottom, self.top)
        axs.plot(self.data_x, self.average, color='r')

        axs.set_title(self.name)
        axs.set(xlabel=self.x_name, ylabel=self.y_name)