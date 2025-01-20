from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import numpy as np


class Traj_Predict():
    # Note: `traj` is passed by reference; slices are also passed as views.
    def __init__(self, traj, pos_dim):
        self.traj = traj
        self.pos_dim = pos_dim
        pass

    def predict(self):
        last_trajs_points = []
        if self.pos_dim ==2:
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

        elif self.pos_dim == 3:
            last_point_x, last_point_y, last_point_z, last_t = self.traj[-1][:4]
            # `num` specifies the number of historical data points used for prediction
            num = 20

            if len(self.traj) >= num:
                kf = KalmanFilter(dim_x=6, dim_z=3)

                # System dynamic matrix and measurement noise matrix for Kalman filter
                dt = 1.0
                kf.F = np.array([[1, 0, 0, dt, 0, 0],
                                 [0, 1, 0, 0, dt, 0],
                                 [0, 0, 1, 0, 0, dt],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])
                kf.H = np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0]])
                kf.R = np.diag([0.1, 0.1, 0.1])  # Measurement noise covariance matrix
                # kf.Q = Q_discrete_white_noise(dim=6, dt=dt, var=0.01 ** 2)  # System dynamic noise covariance matrix
                # Manually construct the Q matrix
                q = 0.01 ** 2
                kf.Q = np.array([[q * dt ** 4 / 4, 0, 0, q * dt ** 3 / 2, 0, 0],
                                 [0, q * dt ** 4 / 4, 0, 0, q * dt ** 3 / 2, 0],
                                 [0, 0, q * dt ** 4 / 4, 0, 0, q * dt ** 3 / 2],
                                 [q * dt ** 3 / 2, 0, 0, q * dt ** 2, 0, 0],
                                 [0, q * dt ** 3 / 2, 0, 0, q * dt ** 2, 0],
                                 [0, 0, q * dt ** 3 / 2, 0, 0, q * dt ** 2]])

                # Initialize the state vector and covariance matrix
                # kf.P = np.diag([100, 100, 100, 10, 10, 10])
                sigma = 0.05
                R = np.diag(3 * [sigma ** 2])
                kf.P = np.diag([R[0, 0], R[1, 1], R[2, 2], 1.0, 1.0, 1.0])

                # Historical trajectory points
                a = np.array(self.traj)
                history = a[-num:, 0:3]  # 包含 z 坐标
                history = history.astype(float)

                kf.x = np.array([history[0][0], history[0][1], history[0][2], 0, 0, 0])

                # Incorporate historical trajectory points to update the state vector and covariance matrix
                for i in range(history.shape[0]):
                    z = history[i].reshape((3, 1))
                    kf.predict()
                    kf.update(z)
                last_point_x, last_point_y, last_point_z = kf.x[:3]
                last_t = self.traj[-1][3] - self.traj[-2][3] + self.traj[-1][3]

            last_trajs_points.append([last_point_x, last_point_y, last_point_z, last_t])
            return last_trajs_points

