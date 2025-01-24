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
                kf = ExtendedKalmanFilter(dim_x=pos_dim, dim_z=pos_dim)
                kf.x = np.array(combined_traj[0][1:1 + pos_dim])
                kf.P *= PAR_kf_P
                kf.R *= PAR_kf_R
                kf.Q *= PAR_kf_Q

                def HJacobian(x):
                    return np.eye(pos_dim)

                def Hx(x):
                    return x

                # Fuse the trajectory
                fused_trajectory = []
                prev_time = None
                prev_values = [None] * pos_dim

                for point in combined_traj:
                    t = point[0]
                    z = np.array(point[1:1 + pos_dim])
                    # Only fuse when time points are different
                    if prev_time is None or prev_time != t:
                        # Fuse trajectory points at the same time
                        kf.predict()

                        # Fuse matched trajectory points using the Kalman filter update step
                        kf.update(z, HJacobian, Hx)
                        new_values = kf.x

                        # Add the fused point to the trajectory
                        fused_point = [t] + list(new_values)
                        fused_trajectory.append(fused_point)

                        # Update previous time point values
                        prev_time = t
                        prev_values = new_values

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
