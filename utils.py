import os
import csv
import json
from geopy.distance import distance
from collections import defaultdict

def dist(alng, alat, blng, blat):
    """
    :param alng: Longitude of target a
    :param alat: Latitude of target a
    :param blng: Longitude of target b
    :param blat: Latitude of target b
    :return: Distance between a and b in meters
    """
    return distance((alat, alng), (blat, blng)).m

def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):  # If it's a defaultdict, convert it to a regular dictionary
        return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):  # If it's a regular dictionary, recursively process its values
        return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):  # If it's a list, recursively process each element
        return [convert_defaultdict_to_dict(v) for v in d]
    else:  # If it's any other type, return it as is
        return d

def save_trajectory_to_csv(ds_trajectory_dict, filename):
    """
    Save a single ds's trajectory data dictionary as a CSV file.
    
    :param ds_trajectory_dict: A dictionary of ds trajectories {id: [[time, lon, lat], ...]}
    :param filename: The name of the CSV file to save
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["id", "trajectory"])

        # Iterate through each trajectory
        for traj_id, trajectory in ds_trajectory_dict.items():
            # Convert the trajectory points (time, lon, lat) to a JSON string
            trajectory_str = json.dumps(trajectory)
            writer.writerow([traj_id, trajectory_str])

    print(f"Data saved to {filename}")

def read_trajectory_from_csv(filename):
    """
    Read trajectory data from a CSV file and return it as a dictionary.
    
    :param filename: The name of the CSV file
    :return: A dictionary of trajectories {traj_id: [[time, lon, lat], ...]}
    """
    ds_trajectory_dict = {}
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        
        for row in reader:
            traj_id = int(row[0])  # traj_id is an integer
            trajectory = json.loads(row[1])  # Deserialize the trajectory points
            # If the trajectory data format is a string like '[time, lon, lat]', process it accordingly
            ds_trajectory_dict[traj_id] = trajectory

    return ds_trajectory_dict

def load_trajectory_data(file_path):
    """Load trajectory data from a file"""
    traj_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            traj_id = int(row["id"])
            trajectory = json.loads(row["trajectory"].replace("'", '"'))
            traj_dict[traj_id] = trajectory
    return traj_dict

def get_ds_ids(input_folder):
    # Get the list of files in the folder
    files = os.listdir(input_folder)
    
    # Filter out all CSV files and extract ds IDs from the filenames
    ds_ids = []
    for file in files:
        # Check if the file ends with '.csv'
        if file.endswith('.csv'):
            # Extract the ds ID from the filename (assuming the format is '1.csv', '2.csv', etc.)
            ds_id = file.split('.')[0]  # Get the first part of the filename as the ds ID
            ds_ids.append(int(ds_id))
    
    return ds_ids
