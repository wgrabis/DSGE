import logging
import numpy as np

logger = logging.getLogger(__name__)


def test_transition(model):
    posterior = model.get_prior_posterior()

    transition_matrix, shock_matrix = model.build_mh_form(posterior)

    A = transition_matrix - np.eye(transition_matrix.shape[0])
    B = shock_matrix @ model.shock_prior.get_mean()

    x_k = np.linalg.solve(A, B)

    for i in range(50):
        x_k = transition_matrix @ x_k

        logger.debug("Iteration: " + str(i))
        logger.debug(x_k)
