import os
import pandas as pd
from load_data import Load_Data
from association import Traj_Association
from matching import match
from utils import *
from fusion import *
from pathlib import Path  

def main():

    # Get the absolute path of the current script

    current_folder = Path(__file__).parent  # Get the directory of the current script

    # Directories for data files: relative to the current script
    input_folder = current_folder / "demo/input"  # Input folder
    output_folder = current_folder / "demo/output"  # Output folder

    # Get the list of data source IDs
    ds_ids = get_ds_ids(input_folder)

    # Create data loader class
    loader = Load_Data()  # No file path needed, dynamic loading used

    # Load data from all data sources
    for ds_id in ds_ids:
        # Construct file path
        trajectory_file = input_folder / f"{ds_id}.csv"
        if trajectory_file.exists():
            loader.get_data(trajectory_file)  # Dynamically load the data file
        else:
            print(f"The data file {trajectory_file} for ds {ds_id} does not exist, skipping.")

    # Print loading results
    print("Data loaded.")

    # Perform association on trajectories for each data source
    for ds_id in ds_ids:
        # Get trajectory data for a single data source
        ds_data = loader.trajectory_data[ds_id]

        # Data Dimensions
        pos_dim = loader.position_dim
        # Create Traj_Association object
        ds_trajs = Traj_Association(ds_data, ds_id, pos_dim)

        # Get associated trajectories
        trajs = ds_trajs.association()

        loader.ds_trajs = trajs

        # Save associated trajectories to CSV file
        output_file = output_folder / f"ds{ds_id}_trajs.csv"
        save_trajectory_to_csv(trajs, output_file)

    print("Association completed.")

    # Match and fuse trajectories from two data sources
    ds_pair = [(1, 2), (3, 4), (5, 6)]
    for ds1, ds2 in ds_pair:
        ds1_file = output_folder / f"ds{ds1}_trajs.csv"
        ds2_file = output_folder / f"ds{ds2}_trajs.csv"
        
        if ds1_file.exists() and ds2_file.exists():
            ds_1_trajs = read_trajectory_from_csv(ds1_file)
            ds_2_trajs = read_trajectory_from_csv(ds2_file)
            
            # Match trajectories from two data sources
            pos_dim = loader.position_dim
            matched_results = {}
            matched_results[ds1] = match(ds_1_trajs, ds_2_trajs, pos_dim)

            # Fuse paired trajectories from two data sources
            traj_dict1 = load_trajectory_data(ds1_file)
            traj_dict2 = load_trajectory_data(ds2_file)
            fused_file = output_folder / f"fused{ds1}_{ds2}_trajs.csv"
            fuse_trajectories(matched_results, traj_dict1, traj_dict2, fused_file, pos_dim)

    print("Fusion completed.")


if __name__ == "__main__":
    main()
