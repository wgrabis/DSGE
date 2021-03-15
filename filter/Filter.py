from abc import ABC, abstractmethod


class Filter(ABC):
    @abstractmethod
    def predict(self, prev_distribution, transition, control_input_matrix, control_input, noise_covariance):
        pass

    @abstractmethod
    def update(self,
               prev_distribution,
               measurement_matrix, measurement_function, measurement_noise_covariance,
               measurement_vector):
        pass
