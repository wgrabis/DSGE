import logging
import sys

from examples.equationParsingExample import test_equations, test_equations2
from examples.kalmanExample import test_kalman
from forecast.BKForecast import BlanchardKahnForecastOld
from forecast.BlanchardKahnForecast import BlanchardKahnForecast
from forecast.BlanchardRaw import BlanchardRaw
from forecast.ForecastAlgorithm import ForecastAlgorithm
from format.JsonFormat import JsonFormat
from format.ParseFile import parse_model_file
from helper.DataPlotter import DataPlotter
from helper.StackedPlot import StackedPlot
from helper.test.TestTransition import test_transition
from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.randomWalkMH import RandomWalkMH
from model.Distribution import NormalVectorDistribution
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
from math import sin
import numpy as np
import pandas as pd
from sympy import Matrix, pprint
import argparse

from model.Equation import EquationParser
from model.config.PlotConfig import PlotConfig
from model.forecast.AgainstCalibrationForecastData import AgainstCalibrationForecastData
from util.RunUtils import RunMode

desired_width = 320
pd.set_option('display.width', desired_width)

from model.EstimationData import EstimationData

model_builder = DsgeModelBuilder()


def setup_logging(is_debug):
    np.set_printoptions(linewidth=np.nan)
    logging_level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def blanchard_raw_test(file_name):
    # A = Matrix([[1.1 , 1], [0.3,  1]])
    # B = Matrix([[0.95, 0.75], [1.8, 0.7]])
    # C = Matrix([[0.3,  0.1], [-0.5, -1.2]])
    raw_model, estimations = parse_model_file(file_name)

    model = model_builder.build(raw_model)

    A, B, C = model.blanchard_raw_representation([])

    print("Matrix triple:")
    pprint(A)
    pprint(B)
    pprint(C)

    BlanchardRaw().non_singular_calculate(A, B, C, np.zeros(model.state_var_count), model.shock_prior.get_mean(), model.state_var_count, len(model.variables) - model.state_var_count, 20)


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


# def forecast_blanchard_dsge_debug(file_name):
#     data_plotter = DataPlotter()
#
#     raw_model, estimations = parse_model_file(file_name)
#
#     variables, structural, shocks = raw_model.entities()
#
#     EquationParser.parse_equations_to_matrices(raw_model.equations, variables, shocks)


def forecast_blanchard_dsge(file_name, is_debug):
    raw_model, _ = parse_model_file(file_name)

    model = model_builder.build(raw_model)

    if is_debug:
        model.print_debug()

    blanchard_forecast_alg = BlanchardKahnForecast()

    policy = blanchard_forecast_alg.calculate_policy(model)

    if is_debug:
        policy.print()

    observables = blanchard_forecast_alg.predict_observables(model, policy, plot_config.time)

    return observables


def run_estimation(file_name, is_debug):
    raw_model, estimations = parse_model_file(file_name)

    assert estimations is not None

    model = model_builder.build(raw_model)

    if is_debug:
        model.print_debug()

    rounds = plot_config.time

    mh_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=model.posterior_covariance())

    posteriors, history = mh_algorithm.calculate_posterior()

    (posterior, distribution) = posteriors.last()

    print("Final posterior:")
    print(model.structural)
    print(posterior)

    return history, posteriors


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

    # data_plotter = DataPlotter()
    # test_kalman(data_plotter)
    # data_plotter.draw_plots()
    # test_equations2()
    # test2()
    # blanchard_raw_test()
    # forecast_blanchard_dsge("samples/toyModel2.json")
    # forecast_blanchard_dsge("samples/philipCurveRe.json", False)
    # forecast_blanchard_dsge("samples/simpleModel.json", True)
    # forecast_blanchard_dsge("samples/rbcModelRe.json")
    # forecast_blanchard_dsge("samples/nkModel.json")
    # forecast_blanchard_dsge("samples/pbar1.json")
    # forecast_dsge(".json")
    # test()


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog="dsgeSolve", description='DSGE solver')

    my_parser.add_argument('modelFile', help="file containing model", type=str)
    my_parser.add_argument('-m', '--mode', type=str, default='fblanchard')
    my_parser.add_argument('-d', '--debug', action='store_true')
    my_parser.add_argument('-t', '--time', type=int, default=40)
    my_parser.add_argument('-sp', '--singlePlot', action='store_true')
    my_parser.add_argument('-pdir', '--plotDir', type=str, default=None)
    my_parser.add_argument('-ds', '--disableShow', action='store_true')
    my_parser.add_argument('-ra', '--runAgainst', type=str, default=None)

    args = my_parser.parse_args()

    run_mode = RunMode(args.mode)
    is_debug = args.debug
    model_file_name = args.modelFile

    run_against = args.runAgainst

    plot_config = PlotConfig.parse(args.time, args.singlePlot, args.plotDir, args.disableShow)

    data_plotter = DataPlotter(plot_config)

    setup_logging(is_debug)

    plots = []

    if run_mode == RunMode.forecastBlanchard:
        main_observables = forecast_blanchard_dsge(model_file_name, is_debug)

        assert main_observables is not None

        if run_against is not None:
            against_observable = forecast_blanchard_dsge(run_against, is_debug)
            main_observables = AgainstCalibrationForecastData(main_observables, against_observable)

        plots = main_observables.prepare_plots()

    if run_mode == RunMode.estimation:
        main_observables, posterior_story = run_estimation(model_file_name, is_debug)

        plots = main_observables.prepare_plots() + posterior_story.get_posterior_plot()

    data_plotter.add_plots(plots)

    data_plotter.draw_plots()

