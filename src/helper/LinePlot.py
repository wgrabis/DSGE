from helper.BasePlot import BasePlot


class LinePlot(BasePlot):
    def __init__(self, name, data_x, plot_data, x_name, y_name, display_name=None):
        super().__init__(name, display_name)
        self.data_x = data_x
        self.plot_data = plot_data
        self.x_name = x_name
        self.y_name = y_name

    def draw_plot(self, axs):
        axs.plot(self.data_x, self.plot_data, color='r')
        axs.plot(self.data_x, [0 for _ in self.data_x], color='b')

        max_y = max(self.plot_data)
        min_y = min(self.plot_data)

        max_y = max(0, max_y)
        min_y = min(0, min_y)

        if min_y < 0 < max_y:
            max_y = max(abs(min_y), max_y)
            min_y = -1 * max(abs(min_y), max_y)

        axs.set_ylim([min_y * 1.1, max_y * 1.1])
        axs.set_xlim([0, max(self.data_x)])

        axs.set_title(self.name)
        axs.set(xlabel=self.x_name, ylabel=self.y_name)