from format.Format import Format
import json


class JsonFormat(Format):
    def parse_format(self, file):
        model = json.loads(file)

        name = model['name']
        parameters = model['parameters']
        equations = model['equations']

        return name, parameters, equations
