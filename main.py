from examples.kalmanExample import test_kalman
from format.JsonFormat import JsonFormat
from model.DsgeModelBuilder import DsgeModelBuilder

model_builder = DsgeModelBuilder()


def run_dsge(file_name, file_format):
    data = open(file_name, "r")
    formatter = None

    if file_format == 'json':
        formatter = JsonFormat()

    name, parameters, equations = formatter.parse_format(data)

    model = model_builder.build(name, parameters, equations)


if __name__ == '__main__':
    test_kalman()

