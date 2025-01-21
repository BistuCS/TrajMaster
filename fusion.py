import json
import csv
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

# ---- Filter parameters ----
PAR_kf_P = 1000
PAR_kf_R = 30000
PAR_kf_Q = 1000

def load_trajectory_data(file_path):
    """Load trajectory file"""
    traj_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            traj_id = int(row["id"])
            trajectory = json.loads(row["trajectory"].replace("'", '"'))
            traj_dict[traj_id] = trajectory
    return traj_dict


def fuse_trajectories(dic_match, traj_dict1, traj_dict2, output_file, pos_dim):
    """Fuse paired trajectories based on the matching dictionary and save to a file"""
    fused_trajectories = {}

    fused_id = 1

    for id1, id2_list in dic_match.items():
        traj1 = traj_dict1.get(id1, [])

        # Iterate over each corresponding trajectory ID
        for id2 in id2_list:
            traj2 = traj_dict2.get(id2, [])

            # Merge trajectory points and sort by time
            combined_traj = sorted(traj1 + traj2, key=lambda x: x[0])

            # Initialize Extended Kalman Filter
            if combined_traj:
                if pos_dim == 2:
                    t_start, lon_start, lat_start = combined_traj[0]
                    kf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
                    kf.x = np.array([lon_start, lat_start])
                    kf.P *= PAR_kf_P
                    kf.R *= PAR_kf_R
                    kf.Q *= PAR_kf_Q

                    def HJacobian(x):
                        return np.array([[1, 0], [0, 1]])

                    def Hx(x):
                        return np.array([x[0], x[1]])

                    # Fuse the trajectory
                    fused_trajectory = []
                    prev_time = None
                    prev_lon = None
                    prev_lat = None

                    for t, lon, lat in combined_traj:
                        # Only fuse when time points are different
                        if prev_time is None or prev_time!= t:
                            # Fuse trajectory points at the same time
                            z = np.array([lon, lat])
                            kf.predict()

                            # Fuse matched trajectory points using the Kalman filter update step
                            kf.update(z, HJacobian, Hx)
                            lon_new, lat_new = kf.x

                            # Add the fused point to the trajectory
                            fused_trajectory.append([t, lon_new, lat_new])

                            # Update previous time point values
                            prev_time = t
                            prev_lon = lon_new
                            prev_lat = lat_new

                elif pos_dim == 3:
                    t_start, lon_start, lat_start, z_start = combined_traj[0]
                    kf = ExtendedKalmanFilter(dim_x=3, dim_z=3)
                    kf.x = np.array([lon_start, lat_start, z_start])
                    kf.P *= PAR_kf_P
                    kf.R *= PAR_kf_R
                    kf.Q *= PAR_kf_Q

                    def HJacobian(x):
                        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                    def Hx(x):
                        return np.array([x[0], x[1], x[2]])

                    # Fuse the trajectory
                    fused_trajectory = []
                    prev_time = None
                    prev_lon = None
                    prev_lat = None
                    prev_z = None

                    for t, lon, lat, z in combined_traj:
                        # Only fuse when time points are different
                        if prev_time is None or prev_time!= t:
                            # Fuse trajectory points at the same time
                            z_vec = np.array([lon, lat, z])
                            kf.predict()

                            # Fuse matched trajectory points using the Kalman filter update step
                            kf.update(z_vec, HJacobian, Hx)
                            lon_new, lat_new, z_new = kf.x

                            # Add the fused point to the trajectory
                            fused_trajectory.append([t, lon_new, lat_new, z_new])

                            # Update previous time point values
                            prev_time = t
                            prev_lon = lon_new
                            prev_lat = lat_new
                            prev_z = z_new
                # Save fused results to a CSV file with columns: traj_id and trajectory

                fused_trajectories[fused_id] = fused_trajectory
                fused_id += 1
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "trajectory"])
        for fused_id, trajectory in fused_trajectories.items():
            trajectory_str = json.dumps(trajectory)  # Convert trajectory to JSON string
            writer.writerow([fused_id, trajectory_str])

    print(f"Fused trajectories saved to {output_file}")
