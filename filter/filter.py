from abc import ABC, abstractmethod


class Filter(ABC):
    @abstractmethod
    def predict(self, x_k, p_k, transition, control_input_matrix, control_input, noise_covariance):
        pass

    @abstractmethod
    def update(self, x_k, p_k, measurement_matrix, measurement_noise_covariance, measurement_vector):
        pass