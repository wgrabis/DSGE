from filter.KalmanFilter import KalmanFilter
from helper.DataHistory import DataHistory
from helper.DataPlotter import DataPlotter
from helper.StackedPlot import StackedPlot
from model.DsgeModelBuilder import DsgeModelBuilder
import numpy as np

# model_builder = DsgeModelBuilder()
# data_plotter = DataPlotter()


def test_kalman(data_plotter):

    dt = 0.1

    X = np.array([[0.0], [0.0], [0.1], [0.1]])
    P = np.diag((0.01, 0.01, 0.01, 0.01))
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    Q = np.eye(X.shape[0])
    B = np.eye(X.shape[0])
    U = np.zeros((X.shape[0], 1))

    Y = np.array([[X[0, 0] + abs(np.random.randn(1)[0])], [X[1, 0] + abs(np.random.randn(1)[0])]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = np.eye(Y.shape[0])
    N_iter = 50

    data_filter = KalmanFilter()
    data_history = DataHistory()
    data_X= [[0] * N_iter, [0] * N_iter]
    data_Y = [[0] * N_iter, [0] * N_iter]

    for i in range(0, N_iter):
        (X, P) = data_filter.predict(X, P, F, B, U, Q)
        (X, P) = data_filter.update(X, P, H, R, Y)
        Y = np.array([[X[0, 0] + abs(0.1 * np.random.randn(1)[0])], [X[1, 0] + abs(0.1 * np.random.randn(1)[0])]])
        data_X[0][i] = X[0][0]
        data_X[1][i] = Y[0][0]
        data_Y[0][i] = X[1][0]
        data_Y[1][i] = Y[1][0]

    data_plotter.add_plot(StackedPlot('Kalman plot', data_X, data_Y, 'X', 'Y'))
