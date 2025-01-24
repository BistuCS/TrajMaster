from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import numpy as np


class Traj_Predict():
    # Note: `traj` is passed by reference; slices are also passed as views.
    def __init__(self, traj, pos_dim):
        self.traj = traj
        self.pos_dim = pos_dim

    def predict(self):
        last_trajs_points = []
        # `num` specifies the number of historical data points used for prediction
        num = 20

        if len(self.traj) >= num:
            # 动态设置卡尔曼滤波器的维度
            dim_x = 2 * self.pos_dim
            dim_z = self.pos_dim
            kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

            # 系统动态矩阵 F
            dt = 1.0
            F = np.eye(dim_x)
            for i in range(self.pos_dim):
                F[i, i + self.pos_dim] = dt
            kf.F = F

            # 测量矩阵 H
            H = np.zeros((dim_z, dim_x))
            for i in range(dim_z):
                H[i, i] = 1
            kf.H = H

            # 测量噪声协方差矩阵 R
            kf.R = np.diag([0.1] * dim_z)

            # 系统动态噪声协方差矩阵 Q
            q = 0.01 ** 2
            Q = np.zeros((dim_x, dim_x))
            for i in range(self.pos_dim):
                Q[i, i] = q * dt ** 4 / 4
                Q[i, i + self.pos_dim] = q * dt ** 3 / 2
                Q[i + self.pos_dim, i] = q * dt ** 3 / 2
                Q[i + self.pos_dim, i + self.pos_dim] = q * dt ** 2
            kf.Q = Q

            # 初始化状态向量和协方差矩阵
            sigma = 0.05
            R = np.diag([sigma ** 2] * dim_z)
            P_diag = list(R.diagonal()) + [1.0] * self.pos_dim
            kf.P = np.diag(P_diag)

            # 历史轨迹点
            a = np.array(self.traj)
            history = a[-num:, :self.pos_dim]
            history = history.astype(float)

            kf.x = np.zeros(dim_x)
            kf.x[:self.pos_dim] = history[0]

            # 结合历史轨迹点更新状态向量和协方差矩阵
            for i in range(history.shape[0]):
                z = history[i].reshape((dim_z, 1))
                kf.predict()
                kf.update(z)

            last_point = kf.x[:self.pos_dim]
            last_t = self.traj[-1][self.pos_dim] - self.traj[-2][self.pos_dim] + self.traj[-1][self.pos_dim]
        else:
            last_point = self.traj[-1][:self.pos_dim]
            last_t = self.traj[-1][self.pos_dim]

        last_point = list(last_point) + [last_t]
        last_trajs_points.append(last_point)
        return last_trajs_points
