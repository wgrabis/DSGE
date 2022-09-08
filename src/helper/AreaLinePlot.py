from helper.BasePlot import BasePlot


class AreaLinePlot(BasePlot):
    def __init__(self, name, data_x, top_y, bottom_y, average_y, x_name, y_name):
        super().__init__(name)
        self.data_x = data_x
        self.top = top_y
        self.bottom = bottom_y
        self.average = average_y
        self.x_name = x_name
        self.y_name = y_name

    def draw_plot(self, axs):
        axs.fill_between(self.data_x, self.bottom, self.top)
        axs.plot(self.data_x, self.average, color='r')
        axs.plot(self.data_x, [0 for _ in self.data_x], color='b')
        # axs.figure(figsize=(3, 3))
        # axs..set_aspect('equal')

        max_y = max(self.average)
        min_y = min(self.average)

        max_y = max(0, max_y)
        min_y = min(0, min_y)

        if min_y < 0 < max_y:
            max_y = max(abs(min_y), max_y)
            min_y = -1 * max(abs(min_y), max_y)

        axs.set_ylim([min_y * 1.1, max_y * 1.1])
        axs.set_xlim([0, max(self.data_x)])

        axs.set_title(self.name)
        axs.set(xlabel=self.x_name, ylabel=self.y_name)
