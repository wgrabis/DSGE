import matplotlib.pyplot as pyplot


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
        fig, axs = pyplot.subplots(self.setCount)

        for i in range(0, self.setCount):
            sub_plot = self.sets[i]

            if self.setCount == 1:
                sub_plot.draw_plot(axs)
            else:
                sub_plot.draw_plot(axs[i])

        pyplot.show()
