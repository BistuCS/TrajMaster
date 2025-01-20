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

        # Handling 2D position
        if self.pos_dim == 2:

            for cur_t in tqdm_list_timestep:

                tqdm_list_timestep.set_description('Data correlation')  # Set progress bar description
                new_points = []  # Store new points for the current time step

                # for sensor_id, lon, lat, time in self.data[cur_t]:  #old code
                #     new_points.append([lon, lat, time])

                # Extract points for the current time step

                for position_x, position_y in self.data[cur_t]:
                    new_points.append([position_x, position_y, cur_t])

                # Point association, get matching pairs
                match_pairs = Points_Association(new_points, pred_trajs_points,
                                                 last_trajs_points, self.THRESHOLD, self.pos_dim).association()

                matched = []  # Matched points
                for i, j in match_pairs:
                    self.ob1Ps[dict_idx_points[tuple(pred_trajs_points[i][:2])]].append(new_points[j])
                    matched.append(j)

                # Unmatched points form new trajectories
                unmatched = set(range(len(new_points))) - set(matched)
                for j in unmatched:
                    self.ob1Ps.append([new_points[j]])

                # Filter out non-timed-out trajectories
                cur_trajs_idx = []
                for idx, trajs in enumerate(self.ob1Ps):
                    # x, y, t = trajs[-1]  # Get the last point of the trajectory
                    x, y, t = trajs[-1] # Get the last point of the trajectory
                    if t >= cur_t - self.THRESHOLD_TIMES_OUT * self.cycle:  # Check if it has not timed out
                        cur_trajs_idx.append(idx)

                # Update predicted trajectories and last trajectory points
                last_trajs_points = []
                pred_trajs_points = []
                dict_idx_points = {}

                for idx in cur_trajs_idx:
                    traj = self.ob1Ps[idx]
                    last_trajs_points.append(traj[-1][:2])  # Last point of the trajectory
                    # Predict trajectory point
                    predict_x, predict_y, predict_t = Traj_Predict(traj, self.pos_dim).predict()[0]
                    pred_trajs_points.append([predict_x, predict_y, predict_t])
                    dict_idx_points[(predict_x, predict_y)] = idx

            # Construct the associated trajectory dictionary
            trajs_ds = defaultdict(list)
            for traj_id, trajs in enumerate(self.ob1Ps):
                for position_x, position_y, t in trajs:
                    trajs_ds[traj_id].append([t, position_x, position_y])

            # Return the associated trajectory dictionary for a single data source {traj_id: list[t, lon, lat]}
            return trajs_ds
        # Handling 3D position
        elif self.pos_dim == 3:
            for cur_t in tqdm_list_timestep:
                tqdm_list_timestep.set_description('Data correlation')  # Set progress bar description
                new_points = []  # Store new points for the current time step

                # for sensor_id, lon, lat, time in self.data[cur_t]:  #old code
                #     new_points.append([lon, lat, time])

                # Extract points for the current time step

                for position_x, position_y, position_z in self.data[cur_t]:
                    new_points.append([position_x, position_y, position_z, cur_t])

                # Point association, get matching pairs

                match_pairs = Points_Association(new_points, pred_trajs_points,
                                                 last_trajs_points, self.THRESHOLD, self.pos_dim).association()

                matched = []  # Matched points
                for i, j in match_pairs:
                    self.ob1Ps[dict_idx_points[tuple(pred_trajs_points[i][:3])]].append(new_points[j])
                    matched.append(j)

                # Unmatched points form new trajectories
                unmatched = set(range(len(new_points))) - set(matched)
                for j in unmatched:
                    self.ob1Ps.append([new_points[j]])

                # Filter out non-timed-out trajectories
                cur_trajs_idx = []
                for idx, trajs in enumerate(self.ob1Ps):
                    # x, y, t = trajs[-1]  # Get the last point of the trajectory
                    x, y, z, t = trajs[-1]  # Get the last point of the trajectory
                    if t >= cur_t - self.THRESHOLD_TIMES_OUT * self.cycle:  # Check if it has not timed out
                        cur_trajs_idx.append(idx)

                # Update predicted trajectories and last trajectory points
                last_trajs_points = []
                pred_trajs_points = []
                dict_idx_points = {}

                for idx in cur_trajs_idx:
                    traj = self.ob1Ps[idx]
                    last_trajs_points.append(traj[-1][:3])  # Last point of the trajectory
                    # Predict trajectory point
                    predict_x, predict_y, predict_z, predict_t = Traj_Predict(traj, self.pos_dim).predict()[0]
                    pred_trajs_points.append([predict_x, predict_y, predict_z, predict_t])
                    dict_idx_points[(predict_x, predict_y, predict_z)] = idx

            # Construct the associated trajectory dictionary
            trajs_ds = defaultdict(list)
            for traj_id, trajs in enumerate(self.ob1Ps):
                for position_x, position_y, position_z,  t in trajs:
                    trajs_ds[traj_id].append([t, position_x, position_y, position_z])

            # Return the associated trajectory dictionary for a single data source {traj_id: list[t, lon, lat]}
            return trajs_ds
