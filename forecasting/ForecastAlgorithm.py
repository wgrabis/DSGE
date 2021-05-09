

class ForecastAlgorithm:
    def forecast(self, model, posterior):
        measurement_function, measurement_matrix = model.measurement_matrices(posterior)
        transition_matrix, shock_matrix = model.build_matrices(posterior)
        noise_covariance = model.noise_covariance(posterior)

        pass