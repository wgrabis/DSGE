import os
import csv

from format.JsonFormat import JsonFormat
from model.EstimationData import EstimationData


def parse_estimation_data(data_description, observable_len):
    if 'data' in data_description:
        return EstimationData(data_description['data'], observable_len)
    if 'name' in data_description:
        data_filename = data_description['name']
        _, file_extension = os.path.splitext(data_filename)
        if file_extension == 'csv':
            csv_reader = csv.reader(data_filename, delimiter=',')
            data = []
            for row in csv_reader:
                data.append(row)
            return EstimationData(data, observable_len)


def parse_model_file(file_name):
    _, file_extension = os.path.splitext(file_name)

    formatter = None

    if file_extension == '.json':
        formatter = JsonFormat()

    assert formatter is not None

    with open(file_name, "r") as data:
        print(data)
        raw_model, estimations_info = formatter.parse_format(data)

    observable_len = len(raw_model.observables)

    estimations = parse_estimation_data(estimations_info, observable_len)

    return raw_model, estimations
