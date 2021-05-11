import os
import csv

from format.JsonFormat import JsonFormat
from model.EstimationData import EstimationData


def parse_estimation_data(data_description):
    if 'data' in data_description:
        return EstimationData(data_description['data'])
    if 'name' in data_description:
        data_filename = data_description['name']
        _, file_extension = os.path.splitext(data_filename)
        if file_extension == 'csv':
            csv_reader = csv.reader(data_filename, delimiter=',')
            data = []
            for row in csv_reader:
                data.append(row)
            return EstimationData(data)


def parse_model_file(file_name):
    _, file_extension = os.path.splitext(file_name)

    formatter = None

    if file_extension == '.json':
        formatter = JsonFormat()

    assert formatter is not None

    with open(file_name, "r") as data:
        print(data)
        name, equations, parameters, variables, estimations_info = formatter.parse_format(data)

    estimations = parse_estimation_data(estimations_info)

    return name, equations, parameters, variables, estimations
