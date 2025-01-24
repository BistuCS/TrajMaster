import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm


class Load_Data():
    def __init__(self):
        self.trajectory_data = defaultdict(lambda: defaultdict(list))  # Nested dictionary structure
        self.position_dim = 0
        self.ds_trajs = {}

    def get_data(self, addr_trajectory_data):
        """Load data for a single ds"""
        # Extract ds_id from the filename (assuming the filename is 'X.csv' where X is the ds_id)
        ds_id = int(os.path.basename(addr_trajectory_data).split('.')[0])

        # Read trajectory data file
        traj_df = pd.read_csv(addr_trajectory_data, sep=',', encoding='utf-8')

        # Use tqdm to set up a progress bar
        tqdm_df = tqdm(traj_df.iterrows(), total=len(traj_df), desc="Load Trajectory Data", ncols=80)

        # Iterate through the trajectory data and store it in the nested dictionary
        for _, row in tqdm_df:
            time = int(row['time'])
            position_str = row['position']
            position_str = position_str.strip("() ").replace(" ", "")
            parts = position_str.split(",")
            # Convert parts to floats and store in a list using map function
            position = list(map(float, parts))
            # Get the dimension of the position
            dim = len(position)
            # Check whether the entire trajectory data is in the same dimension
            if self.position_dim == 0:
                self.position_dim = dim  # Set the dimension if not set yet
            elif self.position_dim != dim:
                raise ValueError(f"Data dimension mismatch: expected {self.position_dim}, got {dim}")

            # Add to the nested dictionary
            self.trajectory_data[ds_id][time].append(position)

            # Update progress description
            tqdm_df.set_description('Load Data')

        print(f"Data loading completed. A total of {len(traj_df)} records were loaded.")




