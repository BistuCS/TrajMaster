from tqdm import tqdm
from collections import defaultdict
from points_association import *
from predict import *

# Association parameters
THRESHOLD = 1900
THRESHOLD_TIMES_OUT = 9
CYCLE = {1: 6, 2: 6, 3: 8, 4: 8, 5: 6, 6: 6}

class Traj_Association:
    def __init__(self, data, ds_id, pos_dim):
        """
        Initialization
        :param data: Trajectory data source dictionary for a single data s {time step: [(ds_id, lon, lat, time), ...]}
        """
        self.ob1Ps = []  # List to store trajectories
        self.data = data  # Input data
        self.ds_id = ds_id  # Automatically extract data source ID
        self.pos_dim = pos_dim
        self.timestep = sorted(list(self.data.keys()))  # Get and sort the time step list
        self.cycle = CYCLE[self.ds_id]  # Determine the cycle based on data source ID
        self.THRESHOLD = THRESHOLD  # Distance threshold for point association
        self.THRESHOLD_TIMES_OUT = THRESHOLD_TIMES_OUT  # Timeout threshold in time steps

    def _extract_ds_id(self):
        """
        Extract data source ID from data
        :return: data source ID
        """
        for time_step, points in self.data.items():
            if points:  # Ensure there is data in the current time step
                return points[0][0]  # Extract ds_id from the first point
        raise ValueError("data source data is empty, unable to extract ds_id")

    def association(self):
        """
        Perform association on the data source trajectory data
        :return: Associated trajectory dictionary {trajectory ID: [[time, lon, lat], ...]}
        """
        last_trajs_points = []  # Trajectory points from the previous time step
        pred_trajs_points = []  # Predicted trajectory points for the current step
        dict_idx_points = {}  # Mapping for trajectory points and indices

        # Iterate over time steps
        tqdm_list_timestep = tqdm(self.timestep, ncols=80)

        for cur_t in tqdm_list_timestep:

            tqdm_list_timestep.set_description('Data correlation')  # Set progress bar description
            new_points = []  # Store new points for the current time step

            # Extract points for the current time step

            for point in self.data[cur_t]:
                new_points.append(point[:self.pos_dim] + [cur_t])

            # Point association, get matching pairs
            match_pairs = Points_Association(new_points, pred_trajs_points,
                                             last_trajs_points, self.THRESHOLD, self.pos_dim).association()

            matched = []  # Matched points
            for i, j in match_pairs:
                key = tuple(pred_trajs_points[i][:self.pos_dim])
                if key in dict_idx_points:
                    self.ob1Ps[dict_idx_points[key]].append(new_points[j])
                else:
                    self.ob1Ps.append([new_points[j]])
                    dict_idx_points[key] = len(self.ob1Ps) - 1
                matched.append(j)

            # Unmatched points form new trajectories
            unmatched = set(range(len(new_points))) - set(matched)
            for j in unmatched:
                self.ob1Ps.append([new_points[j]])

            # Filter out non-timed-out trajectories
            cur_trajs_idx = []
            for idx, trajs in enumerate(self.ob1Ps):
                # Get the last point of the trajectory using slicing
                last_point = trajs[-1][:self.pos_dim + 1]
                t = last_point[-1]
                if t >= cur_t - self.THRESHOLD_TIMES_OUT * self.cycle:  # Check if it has not timed out
                    cur_trajs_idx.append(idx)

            # Update predicted trajectories and last trajectory points
            last_trajs_points = []
            pred_trajs_points = []
            dict_idx_points = {}

            for idx in cur_trajs_idx:
                traj = self.ob1Ps[idx]
                last_trajs_points.append(traj[-1][:self.pos_dim])  # Last point of the trajectory
                # Predict trajectory point
                predict_result = Traj_Predict(traj, self.pos_dim).predict()[0]
                pred_trajs_points.append(predict_result[:self.pos_dim + 1])
                dict_idx_points[tuple(predict_result[:self.pos_dim])] = idx

        # Construct the associated trajectory dictionary
        trajs_ds = defaultdict(list)
        for traj_id, trajs in enumerate(self.ob1Ps):
            for point in trajs:
                # Extract time information, which is always the last element
                t = point[-1]
                # Extract the position information and determine
                # the number of elements of the position information based on self.pos_dim
                position_info = point[:self.pos_dim]
                # Combining location information and time information
                trajectory_point = [t] + list(position_info)
                trajs_ds[traj_id].append(trajectory_point)

        # Return the associated trajectory dictionary for a single data source {traj_id: list[t, lon, lat]}
        return trajs_ds

