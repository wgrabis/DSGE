import os

import numpy

import matplotlib.pyplot as pyplot
import matplotlib
import math

TEXT_WIDTH_PT = 506.295 # Get this from LaTeX using \the\textwidth


def figure_size(figure_size_scale):
    inches_per_pt = 1.0 / 72.27 # Convert pt to inch
    golden_mean = (numpy.sqrt(5.0) - 1.0) / 2.0 # Aesthetic ratio (you could change this)
    figure_width = TEXT_WIDTH_PT * inches_per_pt * figure_size_scale
    figure_height = figure_width * golden_mean
    return [figure_width, figure_height]


class DataPlotter:
    def __init__(self, plot_config):
        self.setCount = 0
        self.sets = []
        self.plot_config = plot_config

    def initialize_figure(self, figure_size_scale = 1.0):
        publication_with_latex = {
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "font.family": "serif",
            # "font.serif": ["Palatino"],
            "axes.labelsize": 8,  # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
            "savefig.dpi": 125,
            "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts}",
            # "figure.figsize": figure_size(figure_size_scale)
        }
        # pyplot.style.use(plot_style)
        matplotlib.rcParams.update(publication_with_latex)

    def add_plot(self, plot):
        self.sets.append(plot)
        self.setCount += 1

    def add_plots(self, plots):
        for plot in plots:
            self.sets.append(plot)
        self.setCount += len(plots)

    def draw_plots(self):
        self.initialize_figure()
        box_width = math.ceil(math.sqrt(self.setCount))

        for i in range(0, self.setCount):
            sub_plot = self.sets[i]

            ind_i = math.floor(i / box_width)
            ind_j = i - ind_i * box_width

            if self.plot_config.as_single_plot:
                fig = pyplot.figure()
                ax = fig.add_subplot(111)
                sub_plot.draw_plot(ax)
                if self.plot_config.auto_save_plot:
                    fig.savefig(os.path.join(self.plot_config.plot_dir, sub_plot.name + ".png"))
            else:
                if self.setCount == 1:
                    fig, axs = pyplot.subplots(box_width, box_width)
                    sub_plot.draw_plot(axs)
                else:
                    col_span = 1

                    if ind_i == box_width - 1 and i == self.setCount:
                        col_span = box_width - ind_j
                    ax = pyplot.subplot2grid((box_width, box_width), (ind_i, ind_j), colspan=col_span)
                    sub_plot.draw_plot(ax)


        # pyplot.gca().set_aspect('equal')
        pyplot.show()
