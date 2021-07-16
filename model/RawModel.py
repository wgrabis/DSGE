class RawModel:
    def __init__(self, name, variables, structural, shocks, definitions, equations, observables, priors):
        self.name = name
        self.variables = variables
        self.structural = structural
        self.shocks = shocks
        self.definitions = definitions
        self.equations = equations
        self.observables = observables
        self.priors = priors

    def entities(self):
        return self.variables, self.structural, self.shocks
