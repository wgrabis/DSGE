from format.Format import Format
import json

from model.Wrappers import Distribution


class JsonFormat(Format):
    def parse_format(self, file):
        model = json.loads(file)

        name = model['name']
        parameters = model['parameters']
        equations = model['equations']
        estimations = model['data']

        shocks, structural = [], []

        priors = {}

        for key in parameters:
            parameter = parameters[key]
            priors[key] = Distribution(parameter.mean, parameter.covariance)
            if parameter.type == 'shock':
                shocks.append(key)

            if parameter.type == 'structural':
                structural.append(key)

        return name, equations, structural, shocks, priors, estimations
