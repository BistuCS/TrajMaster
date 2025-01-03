from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import numpy as np


class Traj_Predict():
    # Note: `traj` is passed by reference; slices are also passed as views.
    def __init__(self, traj):
        self.traj = traj
        pass

    def predict(self):
        last_trajs_points = []
        last_point_x, last_point_y, last_t = self.traj[-1][:3]
        # `num` specifies the number of historical data points used for prediction
        num = 20

        if len(self.traj) >= num:
            kf = KalmanFilter(dim_x=4, dim_z=2)

            # System dynamic matrix and measurement noise matrix for Kalman filter
            dt = 1.0
            kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            kf.R = np.diag([0.1, 0.1])  # Measurement noise covariance matrix
            kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.01 ** 2)  # System dynamic noise covariance matrix

            # Initialize the state vector and covariance matrix
            # kf.P = np.diag([100, 100, 10, 10])
            sigma = 0.05
            R = np.diag(2 * [sigma ** 2])
            kf.P = np.diag([R[0, 0], R[1, 1], 1.0, 1.0])

            # Historical trajectory points
            a = np.array(self.traj)
            history = a[-num:, 0:2]
            history = history.astype(float)

            kf.x = np.array([history[0][0], history[0][1], 0, 0])

            # Incorporate historical trajectory points to update the state vector and covariance matrix
            for i in range(history.shape[0]):
                z = history[i].reshape((2, 1))
                kf.predict()
                kf.update(z)
            last_point_x, last_point_y = kf.x[:2]
            last_t = self.traj[-1][2] - self.traj[-2][2] + self.traj[-1][2]

        last_trajs_points.append([last_point_x, last_point_y, last_t])
        return last_trajs_points
