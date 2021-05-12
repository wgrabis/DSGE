from examples.equationParsingExample import test_equations, test_equations2
from examples.kalmanExample import test_kalman
from forecast.ForecastAlgorithm import ForecastAlgorithm
from format.JsonFormat import JsonFormat
from format.ParseFile import parse_model_file
from helper.DataPlotter import DataPlotter
from helper.StackedPlot import StackedPlot
from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.randomWalkMH import RandomWalkMH
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
from math import sin
import numpy as np

from model.EstimationData import EstimationData

model_builder = DsgeModelBuilder()


def test():
    formula = "sin(x)*x**2"
    code = ast.parse(formula)
    print(code)


def test2():
    a = np.array([[5, 1, 3],
                  [1, 1, 1],
                  [1, 2, 1]])

    b = np.array([1, 2, 3])

    print(a.shape)
    print(b.shape)
    print(np.dot(a, b))


def run_dsge(file_name):
    data_plotter = DataPlotter()

    name, equations, parameters, variables, estimations = parse_model_file(file_name)

    model = model_builder.build(name, equations, parameters, variables)

    likelihood_algorithm = LikelihoodAlgorithm()

    print("Posterior")
    print(model.get_prior_posterior())

    # retries = 0
    rounds = 100

    # probability = likelihood_algorithm.get_likelihood_probability(model, estimations, model.get_prior_posterior())

    mh_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=model.posterior_covariance())
    posteriors, history = mh_algorithm.calculate_posterior()
    histories = [history]

    # for i in range(retries):
    #     mh_algorithm = RandomWalkMH(rounds, model, estimations, posterior)
    #     posterior, data_history, observables = mh_algorithm.calculate_posterior()
    #     histories.append(data_history)

    forecast_alg = ForecastAlgorithm(model)

    observables = forecast_alg.calculate(posteriors.get_post_burnout(), 50, 10, estimations.estimation_time, estimations)

    print("calculated observables")
    print(observables)

    posterior, _ = posteriors.last()

    print("calculated posterior")
    print(posterior)

    # for i in range(len(histories)):
    #     name = "calculated posterior"
    #     for j in range(posterior.shape[0]):
    #         data_x, data_y = histories[i].prepare_plot(j)
    #         print("data")
    #         print(data_x)
    #         print(data_y)
    #         data_plotter.add_plot(StackedPlot(name, [data_x], [data_y], 'iter', 'post'))
    #         name = ''

    data_plotter.add_plots(observables.prepare_plots())

    data_plotter.draw_plots()

    # print(probability)
    # algorithm = RandomWalkMH(rounds, model, estimations)

    # posterior = algorithm.calculate_posterior()


if __name__ == '__main__':
    # data_plotter = DataPlotter()
    # test_kalman(data_plotter)
    # data_plotter.draw_plots()
    # test_equations2()
    test2()
    run_dsge("samples/baseModel.json")
    # test()

