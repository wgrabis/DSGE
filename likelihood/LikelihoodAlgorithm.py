from numpy import dot
import numpy as np
import math
import logging

from model.Distribution import NormalVectorDistribution
from model.Wrappers import TDistribution
import scipy.linalg as scipy

logger = logging.getLogger(__name__)


class LikelihoodAlgorithm:
    def get_likelihood_probability(self, model, data, posterior):
        likelihood_filter = model.likelihood_filter

        # setup
        transition_matrix, shock_matrix = model.build_mh_form(posterior)
        noise_covariance = model.noise_covariance(posterior)

        measurement_function, measurement_matrix = model.measurement_matrices(posterior)

        distribution = self.get_invariant_distribution(transition_matrix, shock_matrix, model.shock_prior)

        logger.debug("Likelihood-start")
        logger.debug(posterior)
        logger.debug(distribution.get_vectors())

        # run algorithm
        control_input_matrix = np.eye(transition_matrix.shape[0])
        control_input = np.zeros(transition_matrix.shape[0])

        distributions_history = []

        likelihood = 0.0

        for t in range(data.estimation_time):
            next_distribution = likelihood_filter.predict(
                distribution.get_vectors(),
                transition_matrix,
                shock_matrix,
                model.shock_prior.get_covariance(),
                control_input_matrix,
                control_input,
                noise_covariance
            )

            logger.debug("Likelihood-update")
            logger.debug(next_distribution)

            distributions_history.append(next_distribution)
            measurement = data[t]

            updated_distribution, point_likelihood = likelihood_filter.update(
                transition_matrix, t,
                next_distribution.get_vectors(),
                measurement_matrix,
                measurement_function,
                model.measurement_noise_covariance,
                measurement
            )

            likelihood += point_likelihood

            logger.debug("likelihood")
            logger.debug(likelihood)

            distribution = updated_distribution

        # distribution(posterior) * distribution(data)
        # (distribution(data)  = distribution1(y1)*distribution2(y2)*... )

        posterior_probability = model.prior_probability(posterior)

        logger.info("posterior")
        logger.info(posterior)
        logger.info(model.structural_prior.structural)
        logger.info(posterior_probability)
        # print(math.exp(-likelihood))
        posterior_probability += likelihood

        # for t in range(1, data.estimation_time):
        #     posterior_probability *= distributions_history[t - 1].probability_of(data.measurements[t - 1])

        logger.info("posterior probability")
        logger.info(posterior_probability)

        return posterior_probability, distribution

    @staticmethod
    def get_invariant_distribution(transition_matrix, shock_matrix, shock_distribution):
        A = transition_matrix - np.eye(transition_matrix.shape[0])
        B = shock_matrix @ shock_distribution.get_mean()#np.zeros(transition_matrix.shape[0], dtype='float')

        logger.debug("get invariant distribution")
        logger.debug(transition_matrix)
        logger.debug(shock_distribution.get_covariance())

        logger.debug(A)
        logger.debug(B)

        s_0 = np.linalg.solve(A, B)
        p_0 = scipy.solve_discrete_lyapunov(transition_matrix,
                                            shock_matrix @ shock_distribution.get_covariance() @ shock_matrix.transpose())

        logger.debug("Invariant distribution")
        logger.debug(s_0)
        logger.debug(p_0)

        return NormalVectorDistribution(s_0, np.matrix(p_0, dtype=float))
