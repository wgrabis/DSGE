class PlotConfig:
    def __init__(self, time, as_single_plot, auto_save_plot, plot_dir, disable_show_plots):
        self.time = time
        self.as_single_plot = as_single_plot
        self.auto_save_plot = auto_save_plot
        self.plot_dir = plot_dir
        self.disable_show_plots = disable_show_plots

    @staticmethod
    def parse(time, as_single_plot, plot_dir, disable_show_plots):
        return PlotConfig(time, as_single_plot, plot_dir != '', plot_dir, disable_show_plots)

