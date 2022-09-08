import logging
import sys

from forecast.ImpulseResponseForecast import ImpulseResponseForecast
from forecast.RandomPathForecast import RandomPathForecast
from solver.BlanchardRaw import BlanchardRaw
from format.ParseFile import parse_model_file
from helper.DataPlotter import DataPlotter
from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.randomWalkMH import RandomWalkMH
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
import numpy as np
import pandas as pd
from sympy import pprint
import argparse

from model.config.PlotConfig import PlotConfig
from model.forecast.AgainstCalibrationForecastData import AgainstCalibrationForecastData
from util.RunUtils import RunMode

desired_width = 320
pd.set_option('display.width', desired_width)

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

    calibration = model.structural_prior.get_prior_vector().get_full_vector()
    blanchard_forecast_alg = ImpulseResponseForecast(model, calibration)

    observables, policy = blanchard_forecast_alg.predict_observables(plot_config.time)

    if is_debug:
        policy.print()

    return observables


def run_forecast_with_estimation(file_name, is_debug):
    raw_model, estimations = parse_model_file(file_name)

    assert estimations is not None

    model = model_builder.build(raw_model)

    if is_debug:
        model.print_debug()

    rounds = plot_config.rounds

    covariance = model.structural_prior.get_param_covariance()

    mh_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=covariance)

    posteriors, history, stats = mh_algorithm.calculate_posterior()

    forecast_alg = RandomPathForecast(model, posteriors, estimations)

    # todo add rounds and time separate
    forecast_data = forecast_alg.calculate(20, plot_config.time, 0)

    return history, posteriors, forecast_data


def run_estimation(file_name, is_debug, chain_run):
    raw_model, estimations = parse_model_file(file_name)

    assert estimations is not None

    model = model_builder.build(raw_model)

    if is_debug:
        model.print_debug()

    rounds = plot_config.rounds

    covariance = model.structural_prior.get_param_covariance()

    mh_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=covariance)

    posteriors, history, stats = mh_algorithm.calculate_posterior()

    if chain_run:
        distribution, _ = stats

        print("First run:")
        print(distribution.get_covariance())

        mh2_algorithm = RandomWalkMH(rounds, model, estimations, with_covariance=distribution.get_covariance())
        posteriors, history = mh2_algorithm.calculate_posterior(posteriors, history)

    (posterior, distribution) = posteriors.last()

    print("Final posterior:")
    print(model.structural_prior.ordered_params)
    print(posterior)

    return history, posteriors


def test_run_estimate(model_file, is_debug):
    raw_model, estimations = parse_model_file(model_file)

    assert estimations is not None

    model = model_builder.build(raw_model)

    if is_debug:
        model.print_debug()

    likelihood_alg = LikelihoodAlgorithm()

    test_vectors = [
        ("good", [0.99, 0.99, 0.0836, 0.0001, 0.9470, 0.9625, 0.0617, 0.3597, 0.2536, 0.0347]),
        ("wrong", [0.99425141,  0.03169621,  0.00172063, 0.96534852, -0.44178758,  0.016841,    0.9888124, 0.82655394, 0.56722404,  0.58486679]),
        ("wrong", [0.99,        0.99,        0.72814111, -0.00398059,  0.76622819,  0.39811385, -0.09975318,  0.68298919, 0.41500156,  0.14496968]),
        ("prior", model.structural_prior.get_prior_vector().get_full_vector()),
        ("wrong", [0.99, 0.99, 0.35377216, 0.0188116, 0.78261931, 0.20572711, -0.17007625,  0.23627613,  0.02664442, 0.03837244]),
        ("wrong", [ 9.90000000e-01,  9.90000000e-01,  6.42226200e-01,  7.34113439e-03,  6.45596798e-01,  4.14374081e-02, -1.36375698e-01,  2.21047584e-01,  1.96998441e-02,  6.71683795e-04]),
        ("wrong", [0.99,       0.99,       0.00878366, 0.03821123, 0.98792298, 0.10398958, 0.02864189, 0.64379357, 0.73062375, 0.06499799]),
        ("wrong best", [9.90000000e-01, 9.90000000e-01, 8.51225072e-02, 7.87604623e-04, 9.75630602e-01, 9.82302572e-01, 3.57974845e-02, 9.90387028e-01, 8.96625294e-01, 1.88979308e-06]),
        ("wrong", [ 9.90000000e-01,  9.90000000e-01,  6.17804018e-02,  2.13153636e-08,  9.67864532e-01,  4.73437991e-01, -6.47478143e-02, 3.59700000e-01,  3.71395453e-01,  5.19534675e-06]),
        ("wrong", [ 9.90000000e-01,  9.90000000e-01,  5.25342664e-03,  6.07739311e-08,  9.79622512e-01,  8.69952395e-01, -1.01649482e-01, 3.59700000e-01,  8.15358328e-01,  4.00100341e-07])
    ]

    results = [(flag, likelihood_alg.get_likelihood_probability(model, estimations, np.array(vector))) for (flag, vector) in test_vectors]

    print("TestModeResult:")
    for (flag, (probability, covariance)) in results:
        print(flag + ":")
        print(probability)
        print(covariance)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog="dsgeSolve", description='DSGE solver')

    my_parser.add_argument('modelFile', help="file containing model", type=str)
    my_parser.add_argument('-m', '--mode', type=str, default='fblanchard')
    my_parser.add_argument('-d', '--debug', action='store_true')
    my_parser.add_argument('-t', '--time', type=int, default=40)
    my_parser.add_argument('-r', '--rounds', type=int, default=100)
    my_parser.add_argument('-sp', '--singlePlot', action='store_true')
    my_parser.add_argument('-pdir', '--plotDir', type=str, default=None)
    my_parser.add_argument('-ds', '--disableShow', action='store_true')
    my_parser.add_argument('-ra', '--runAgainst', type=str, default=None)
    my_parser.add_argument('-c', '--chain', action='store_true')

    args = my_parser.parse_args()

    run_mode = RunMode(args.mode)
    is_debug = args.debug
    model_file_name = args.modelFile
    chain_run = args.chain

    run_against = args.runAgainst

    plot_config = PlotConfig.parse(args.time, args.singlePlot, args.plotDir, args.disableShow, args.rounds)

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
        main_observables, posterior_story = run_estimation(model_file_name, is_debug, chain_run)

        plots = main_observables.prepare_plots() + posterior_story.get_posterior_plot()

    if run_mode == RunMode.forecastEstimation:
        main_observables, posterior_story, forecast_observables = \
            run_forecast_with_estimation(model_file_name, is_debug)

        plots = main_observables.prepare_plots() + posterior_story.get_posterior_plot() + \
                forecast_observables.prepare_plots()

    if run_mode == RunMode.testEstimation:
        test_run_estimate(model_file_name, is_debug)

    if len(plots) > 0:
        data_plotter.add_plots(plots)
        data_plotter.draw_plots()

