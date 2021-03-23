from format.Format import Format
import json

from model.EstimationData import EstimationData
from model.Wrappers import TDistribution


class JsonFormat(Format):
    def parse_format(self, file):
        model = json.load(file)

        name = model['name']
        parameters = model['parameters']
        equations = model['equations']
        estimations = model['data']
        variables = model['variables']

        # shocks, structural = [], []
        #
        # priors = {}

        # for key in parameters:
        #     parameter = parameters[key]
        #     priors[key] = TDistribution(parameter.mean, parameter.covariance)
        #     if parameter.type == 'shock':
        #         shocks.append(key)
        #
        #     if parameter.type == 'structural':
        #         structural.append(key)

        return name, equations, parameters, variables, EstimationData(estimations)
