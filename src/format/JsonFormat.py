from format.Format import Format
import json

from model.EstimationData import EstimationData
from model.RawModel import RawModel
from model.Wrappers import TDistribution


class JsonFormat(Format):
    def parse_format(self, file):
        model = json.load(file)

        name = model['name']
        variables = model['variables']
        structural = model['structural']
        shocks = model['shocks']

        model_equations = model['model']

        definitions = model_equations['definitions']
        equations = model_equations['equations']
        observables = model_equations['observables']

        priors = model['priors']

        estimations = None

        if 'estimations' in model:
            estimations = model['estimations']

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

        return RawModel(name, variables, structural, shocks, definitions, equations, observables, priors), estimations
