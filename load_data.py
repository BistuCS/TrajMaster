import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class Load_Data():
    def __init__(self):
        self.trajectory_data = defaultdict(lambda: defaultdict(list))  # Nested dictionary structure
        self.ds_trajs = {}

    def get_data(self, addr_trajectory_data):
        """Load data for a single ds"""
        # Read trajectory data file
        traj_df = pd.read_csv(addr_trajectory_data, sep=',', encoding='utf-8')

        # Use tqdm to set up a progress bar
        tqdm_df = tqdm(traj_df.iterrows(), total=len(traj_df), desc="Load Trajectory Data", ncols=80)

        # Iterate through the trajectory data and store it in the nested dictionary
        for _, row in tqdm_df:
            time = int(row['time'])
            ds_id = int(row['ds_id'])
            lon = round(float(row['lon']), 6)  # Keep 6 decimal places
            lat = round(float(row['lat']), 6)

            # Add to the nested dictionary
            self.trajectory_data[ds_id][time].append((ds_id, lon, lat, time))

            # Update progress description
            tqdm_df.set_description('Load Data')

        print(f"Data loading completed. A total of {len(traj_df)} records were loaded.")
