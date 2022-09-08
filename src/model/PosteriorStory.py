import logging

from helper.LinePlot import LinePlot

logger = logging.getLogger(__name__)


class PosteriorStory:
    def __init__(self):
        self.posteriors = []
        self.posterior_values = []
        self.burnout = 0.4
        self.burnout_max = 50
        self.len = 0

    def add(self, posterior, distribution, posterior_value):
        self.len += 1
        self.posteriors.append((posterior, distribution))
        self.posterior_values.append(posterior_value)

    def get_post_burnout(self):
        print(self.posteriors)

        if self.len == 1:
            return self.posteriors

        if int(self.len * self.burnout) > self.burnout_max:
            return self.posteriors[self.burnout_max:]

        return self.posteriors[int(self.len * self.burnout):]

    def last(self):
        return self.posteriors[self.len - 1]

    def get_posterior_plot(self):
        x_plot_data = []
        y_plot_data = []

        for i in range(self.len):
            x_plot_data.append(i)
            y_plot_data.append(self.posterior_values[i])

        logger.debug("Prepare plots")
        logger.debug(y_plot_data)
        logger.debug(self.posterior_values)

        return [LinePlot("Likelihood", x_plot_data, y_plot_data, "round", "")]
