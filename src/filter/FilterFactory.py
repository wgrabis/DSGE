from abc import ABC, abstractmethod

from filter.KalmanThirdFilter import KalmanThirdFilter


class FilterFactory(ABC):
    @abstractmethod
    def build_filter(self, transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function, measurement_noise_covariance):
        pass


class KalmanFilterFactory(FilterFactory):
    def build_filter(self, transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function,
                     measurement_noise_covariance):
        return KalmanThirdFilter(transition, shock_matrix, shock_covariance, measurement_matrix, measurement_function, measurement_noise_covariance)