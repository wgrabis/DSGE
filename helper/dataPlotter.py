import matplotlib.pyplot as pyplot
from collections import namedtuple

PlotSet = namedtuple('PlotSet', 'data name x y')


class DataPlotter:
    def __init__(self):
        self.setCount = 0
        self.sets = []

    def add_multiplot(self, name, data_x, data_y, x_name, y_name):
        count = len(data_x)
        for i in range(0, count):
            pyplot.plot(data_x[i], data_y[i])

        pyplot.title(name)
        pyplot.ylabel(x_name)
        pyplot.xlabel(y_name)
        pyplot.show()

    def add_plot(self, data, name, x_name, y_name):
        data_set = PlotSet(data, name, x_name, y_name)

        self.sets[self.setCount] = data_set
        self.setCount += 1

    def draw_plots(self):
        fig, axs = pyplot.subplots(self.setCount)

        for i in range(0, self.setCount):
            data_set = self.sets[i]

            axs[i].set_title(data_set.name)
            axs[i].plot('x', 'y', data_set.data)
            axs[i].set(xlabel=data_set.x, ylabel=data_set.y)
