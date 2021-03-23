from examples.equationParsingExample import test_equations, test_equations2
from examples.kalmanExample import test_kalman
from format.JsonFormat import JsonFormat
from helper.DataPlotter import DataPlotter
from likelihood.LikelihoodAlgorithm import LikelihoodAlgorithm
from metropolis_hastings.randomWalkMH import RandomWalkMH
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
from math import sin

from model.EstimationData import EstimationData

model_builder = DsgeModelBuilder()


def test():
    formula = "sin(x)*x**2"
    code = ast.parse(formula)
    print(code)


def run_dsge(file_name, file_format):
    formatter = None

    if file_format == 'json':
        formatter = JsonFormat()

    rounds = 1000

    name, equations, parameters, estimations = None, None, None, None

    with open(file_name, "r") as data:
        print(data)
        name, equations, parameters, variables, estimations = formatter.parse_format(data)

    model = model_builder.build(name, equations, parameters, variables)

    likelihood_algorithm = LikelihoodAlgorithm()

    print("Posterior")
    print(model.get_prior_posterior())

    probability = likelihood_algorithm.get_likelihood_probability(model, estimations, model.get_prior_posterior())

    print(probability)
    # algorithm = RandomWalkMH(rounds, model, estimations)

    # posterior = algorithm.calculate_posterior()


if __name__ == '__main__':
    # data_plotter = DataPlotter()
    # test_kalman(data_plotter)
    # data_plotter.draw_plots()
    # test_equations2()
    run_dsge("samples/baseModel.json", "json")
    test()

