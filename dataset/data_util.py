import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py

class TrajectoryDataset(Dataset):
    """Dataset for loading trajectory data from HDF5 files."""
    def __init__(self, file_paths, traj_length):
        self.samples = []
        self.load_samples(file_paths, traj_length)
    
    def load_samples(self, file_paths, traj_length):
        for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
            with h5py.File(file_path, 'r') as h5_file:
                for user_id in h5_file.keys(): # Iterate over users in the HDF5 file
                    user_group = h5_file[user_id]
                    latitudes = user_group['latitudes'][:]
                    longitudes = user_group['longitudes'][:]
                    hours = user_group['hours'][:]
                    
                    # Create samples by sliding a window of traj_length over the user's trajectory
                    if len(latitudes) > traj_length:
                        for j in range(len(latitudes) - traj_length + 1):
                            self.samples.append((hours[j:j+traj_length], latitudes[j:j+traj_length], longitudes[j:j+traj_length]))
                    elif len(latitudes) == traj_length:
                        self.samples.append((hours[:], latitudes[:], longitudes[:]))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hours, latitudes, longitudes = self.samples[idx]
        return torch.tensor(hours, dtype=torch.float32), torch.tensor(latitudes, dtype=torch.float32), torch.tensor(longitudes, dtype=torch.float32)


class PatternDataset:
    """Dataset for loading trajectory patterns, possibly for prototype learning."""
    def __init__(self, file_paths):
        self.trajectories = []
        self.load_samples(file_paths)

    def load_samples(self, file_paths):
        for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
            with pd.HDFStore(file_path, 'r') as store: # Using pandas HDFStore
                data = store['data']
                for i in range(len(data)):
                    abs_time_list = np.array(data['ABS_TIME'][i])
                    lat_list = np.array(data['LAT'][i])
                    lng_list = np.array(data['LNG'][i])
                    trajectory = list(zip(abs_time_list, lat_list, lng_list))
                    self.trajectories.append(trajectory)

    def get_all_trajectories(self):
        return self.trajectories

    def pad_trajectories(self):
        max_length = max(len(traj) for traj in self.trajectories)
        padded_samples = []

        for traj in self.trajectories:
            if len(traj) < max_length:
                # Pad shorter trajectories with their last point
                last_point = traj[-1]
                padding = [last_point] * (max_length - len(traj))
                padded_traj = traj + padding
            else:
                padded_traj = traj
            padded_samples.append(padded_traj)

        return padded_samples


class MinMaxScaler:
    """Min-Max Scaler for trajectory data."""
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        self.min_val = data.amin(dim=(0, 1), keepdim=True)
        self.max_val = data.amax(dim=(0, 1), keepdim=True)

    def transform(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data):
        return data * (self.max_val - self.min_val) + self.min_val

