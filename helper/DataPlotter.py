import matplotlib.pyplot as pyplot
import math


class DataPlotter:
    def __init__(self):
        self.setCount = 0
        self.sets = []

    def add_plot(self, plot):
        self.sets.append(plot)
        self.setCount += 1

    def add_plots(self, plots):
        for plot in plots:
            self.sets.append(plot)
        self.setCount += len(plots)

    def draw_plots(self):
        box_width = math.ceil(math.sqrt(self.setCount))

        fig, axs = pyplot.subplots(box_width, box_width)

        for i in range(0, self.setCount):
            sub_plot = self.sets[i]

            ind_i = math.floor(i / box_width)
            ind_j = i - ind_i * box_width

            if self.setCount == 1:
                sub_plot.draw_plot(axs)
            else:
                sub_plot.draw_plot(axs[ind_i][ind_j])

        # pyplot.gca().set_aspect('equal')
        pyplot.show()
