import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import dist

class Points_Association():
    def __init__(self, new_points, pred_traj_points, last_trajs_points, threshold_d, pos_dim):
        self.new_points = new_points
        self.pred_traj_points = pred_traj_points
        self.last_trajs_points = last_trajs_points
        self.threshold_d = threshold_d
        self.pos_dim = pos_dim

    def association(self):
        """Trajectory association"""
        """First establish edges: controlled by a threshold. 
        If the distance between two points exceeds the threshold, they are considered unreachable."""
        len_new = len(self.new_points)
        len_pred = len(self.pred_traj_points)
        # pred line  & new column
        cost = np.array([[0 for j in range(len_new)] for i in range(len_pred)])

        def get_distance_function():
            if self.pos_dim == 2:
                return lambda p1, p2: dist(p1[0], p1[1], p2[0], p2[1])
            else:
                return lambda p1, p2: self.dist_general(p1[:self.pos_dim], p2[:self.pos_dim])

        distance_func = get_distance_function()

        for i in range(len_pred):
            mn = float('inf')
            for j in range(len_new):
                cost[i][j] = dis = distance_func(self.pred_traj_points[i], self.new_points[j])
                mn = min(mn, dis)
            if mn > self.threshold_d:
                for j in range(len_new):
                    cost[i][j] = mn

        if len_pred == 0:
            return []
        """After association, return the successfully matched pairs."""
        # Find the elements in each row and column of the cost matrix so that their sum is minimized,
        # and return their row and column subscript lists
        row_ind, col_ind = linear_sum_assignment(cost)
        match_pair = []
        for i in range(len(row_ind)):
            idx_pre, idx_new = row_ind[i], col_ind[i]
            if cost[idx_pre][idx_new] > self.threshold_d:
                continue
            match_pair.append([idx_pre, idx_new])

        return match_pair

    def dist_general(self, point1, point2):
        """General distance calculation for dimensions > 2"""
        dist_squared = 0
        for dim in range(self.pos_dim):
            dist_squared += (point1[dim] - point2[dim]) ** 2
        return np.sqrt(dist_squared) * 100000