import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import dist


class Points_Association():
    def __init__(self, new_points, pred_traj_points, last_trajs_points, threshold_d):
        self.new_points = new_points
        self.pred_traj_points = pred_traj_points
        self.last_trajs_points = last_trajs_points
        self.threshold_d = threshold_d

    def association(self):
        """Trajectory association"""
        """First establish edges: controlled by a threshold. 
        If the distance between two points exceeds the threshold, they are considered unreachable."""
        len_new = len(self.new_points)
        len_pred = len(self.pred_traj_points)
        cost = np.array([[0 for j in range(len_new)] for i in range(len_pred)])
        
        for i in range(len_pred):
            mn = float('inf')
            for j in range(len_new):
                x1, y1, _ = self.pred_traj_points[i]
                x2, y2 = self.new_points[j][:2]
                cost[i][j] = dis = dist(x1, y1, x2, y2)
                mn = min(mn, dis)
            if mn > self.threshold_d:
                for j in range(len_new):
                    cost[i][j] = mn

        if len_pred == 0:
            return []
        """After association, return the successfully matched pairs."""
        row_ind, col_ind = linear_sum_assignment(cost)
        match_pair = []
        for i in range(len(row_ind)):
            idx_pre, idx_new = row_ind[i], col_ind[i]
            if cost[idx_pre][idx_new] > self.threshold_d:
                continue
            match_pair.append([idx_pre, idx_new])

        return match_pair
