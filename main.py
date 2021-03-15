from examples.kalmanExample import test_kalman
from format.JsonFormat import JsonFormat
from helper.DataPlotter import DataPlotter
from model.DsgeModelBuilder import DsgeModelBuilder
import ast
from math import sin

model_builder = DsgeModelBuilder()


def test():
    formula = "sin(x)*x**2"
    code = ast.parse(formula)
    print(code)


def run_dsge(file_name, file_format):
    data = open(file_name, "r")
    formatter = None

    if file_format == 'json':
        formatter = JsonFormat()

    name, equations, structural, shocks, priors, estimations = formatter.parse_format(data)

    model = model_builder.build(name, equations, structural, shocks, priors)


if __name__ == '__main__':
    # data_plotter = DataPlotter()
    # test_kalman(data_plotter)
    # data_plotter.draw_plots()
    test();

