from examples.equationParsingExample import test_equations, test_equations2
from examples.kalmanExample import test_kalman
from forecast.BlanchardKahnForecast import BlanchardKahnForecast
from forecast.BlanchardRaw import BlanchardRaw
from forecast.ForecastAlgorithm import ForecastAlgorithm
from format.JsonFormat import JsonFormat
from format.ParseFile import parse_model_file
from helper.DataPlotter import DataPlotter
from helper.StackedPlot import StackedPlot
from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.randomWalkMH import RandomWalkMH
from model.Distribution import NormalVectorDistribution
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
from math import sin
import numpy as np
import pandas as pd
from sympy import Matrix

from model.Equation import EquationParser

desired_width = 320
pd.set_option('display.width', desired_width)

from model.EstimationData import EstimationData

model_builder = DsgeModelBuilder()


def blanchard_raw_test():
    A = Matrix([[1.1 , 1], [0.3,  1]])
    B = Matrix([[0.95, 0.75], [1.8, 0.7]])
    C = Matrix([[0.3,  0.1], [-0.5, -1.2]])

    BlanchardRaw().calculate(A, B, C, [], [], 1, 0)

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


def forecast_blanchard_dsge_debug(file_name):
    data_plotter = DataPlotter()

    raw_model, estimations = parse_model_file(file_name)

    variables, structural, shocks = raw_model.entities()

    EquationParser.parse_equations_to_matrices(raw_model.equations, variables, shocks)


def forecast_blanchard_dsge(file_name, state_count):
    data_plotter = DataPlotter()

    raw_model, estimations = parse_model_file(file_name)

    model = model_builder.build(raw_model)

    blanchard_forecast_alg = BlanchardKahnForecast(model, state_count)

    observables = blanchard_forecast_alg.calculate(20)

    data_plotter.add_plots(observables.prepare_plots())

    data_plotter.draw_plots()


def forecast_dsge(file_name):
    data_plotter = DataPlotter()

    raw_model, estimations = parse_model_file(file_name)

    model = model_builder.build(raw_model)

    forecast_alg = ForecastAlgorithm(model)

    # likelihood_algorithm = LikelihoodAlgorithm()

    preposterior = model.get_prior_posterior()

    transition_matrix, shock_matrix = model.build_matrices(preposterior)
    noise_covariance = model.noise_covariance(preposterior)

    structural_mean = len(model.variables)

    posterior = NormalVectorDistribution(
        np.linalg.solve(transition_matrix - np.eye(transition_matrix.shape[0]), np.zeros(transition_matrix.shape[0], dtype='float')),
        np.zeros((structural_mean, structural_mean)))#model.get_prior_posterior()

    # _, distribution = likelihood_algorithm.get_likelihood_probability(model, estimations, posterior)

    posteriors = [(preposterior, posterior)]

    observables = forecast_alg.calculate(posteriors, 100, 100, estimations.estimation_time,
                                         estimations)
    data_plotter.add_plots(observables.prepare_plots())

    data_plotter.draw_plots()


def run_dsge(file_name):
    data_plotter = DataPlotter()

    raw_model, estimations = parse_model_file(file_name)

    model = model_builder.build(raw_model)

    likelihood_algorithm = LikelihoodAlgorithm()

    print("Posterior")
    print(model.get_prior_posterior())

    # retries = 0
    rounds = 3

    # probability = likelihood_algorithm.get_likelihood_probability(model, estimations, model.get_prior_posterior())

    mh_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=model.posterior_covariance())
    posteriors, history = mh_algorithm.calculate_posterior()
    histories = [history]

    # print(posteriors.get_post_burnout()[0])

    # for i in range(retries):
    #     mh_algorithm = RandomWalkMH(rounds, model, estimations, posterior)
    #     posterior, data_history, observables = mh_algorithm.calculate_posterior()
    #     histories.append(data_history)

    forecast_alg = ForecastAlgorithm(model)

    observables = forecast_alg.calculate([posteriors.get_post_burnout()[0]], 50, 10, estimations.estimation_time, estimations)

    print("calculated observables")
    # print(observables)

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
    blanchard_raw_test()
    # forecast_blanchard_dsge_debug("samples/rbcModelRe.json")
    # forecast_dsge(".json")
    # test()

