class PlotConfig:
    def __init__(self, time, as_single_plot, auto_save_plot, plot_dir):
        self.time = time
        self.as_single_plot = as_single_plot
        self.auto_save_plot = auto_save_plot
        self.plot_dir = plot_dir

    @staticmethod
    def parse(time, as_single_plot, plot_dir):
        return PlotConfig(time, as_single_plot, plot_dir != '', plot_dir)

