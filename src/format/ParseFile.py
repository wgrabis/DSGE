import os
import csv
import logging

from format.JsonFormat import JsonFormat
from model.EstimationData import EstimationData

log = logging.getLogger(__name__)


def parse_estimation_data(data_description, observable_len):
    if 'data' in data_description:
        return EstimationData(data_description['data'], observable_len)
    if 'name' in data_description:
        data_filename = data_description['name']
        _, file_extension = os.path.splitext(data_filename)
        if file_extension == '.csv':
            with open(data_filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                data = []
                observable_names = None
                for row in csv_reader:
                    if observable_names is None:
                        observable_names = row
                    else:
                        data.append(row)

                return EstimationData(data, observable_len, observable_names)


def parse_model_file(file_name):
    _, file_extension = os.path.splitext(file_name)

    formatter = None

    if file_extension == '.json':
        formatter = JsonFormat()

    assert formatter is not None

    with open(file_name, "r") as data:
        log.debug(data)
        raw_model, estimations_info = formatter.parse_format(data)

    observable_len = len(raw_model.observables)

    if estimations_info is not None:
        estimations = parse_estimation_data(estimations_info, observable_len)
    else:
        estimations = None

    return raw_model, estimations
