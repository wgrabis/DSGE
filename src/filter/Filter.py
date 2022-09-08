from abc import ABC, abstractmethod


class Filter(ABC):
    def __init__(self, transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function, measurement_noise_covariance):
        self.transition = transition
        self.shock_matrix = shock_matrix
        self.shock_covariance = shock_covariance
        self.measurement_matrix = measurement_matrix
        self.measurement_function = measurement_function
        self.measurement_noise_covariance = measurement_noise_covariance
    # @abstractmethod
    # def predict(self, prev_distribution, transition, shock_matrix, shock_covariance, noise_covariance):
    #     pass
    #
    # @abstractmethod
    # def update(self, transition,
    #            time, prev_distribution,
    #            measurement_matrix, measurement_function, measurement_noise_covariance,
    #            measurement_vector):
    #     pass

    @abstractmethod
    def filter(self, time, prev_distribution, y_t):
        pass
