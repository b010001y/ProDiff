import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, file_paths, traj_length):
        self.samples = []
        self.load_samples(file_paths, traj_length)

    def load_samples(self, file_paths, traj_length):
        for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
            with pd.HDFStore(file_path, 'r') as store:
                data = store['data']
                for i in range(len(data)):
                    lat_list = np.array(data['LAT'][i])
                    lng_list = np.array(data['LNG'][i])
                    abs_time_list = np.array(data['ABS_TIME'][i])

                    lat_list = lat_list.reshape(-1, 1)
                    lng_list = lng_list.reshape(-1, 1)

                    if len(lat_list) > traj_length:
                        for j in range(len(lat_list) - traj_length + 1):
                            self.samples.append((abs_time_list[j:j+traj_length], lat_list[j:j+traj_length].flatten(), lng_list[j:j+traj_length].flatten()))
                    elif len(lat_list) == traj_length:
                        self.samples.append((abs_time_list[:], lat_list[:].flatten(), lng_list[:].flatten()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        abs_time, lat, lng = self.samples[idx]
        return torch.tensor(abs_time, dtype=torch.float32), torch.tensor(lat, dtype=torch.float32), torch.tensor(lng, dtype=torch.float32)


class PatternDataset:
    def __init__(self, file_paths):
        self.trajectories = []
        self.load_samples(file_paths)

    def load_samples(self, file_paths):
        for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
            with pd.HDFStore(file_path, 'r') as store:
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
                last_point = traj[-1]
                padding = [last_point] * (max_length - len(traj))
                padded_traj = traj + padding
            else:
                padded_traj = traj
            padded_samples.append(padded_traj)

        return padded_samples


class MinMaxScaler:
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

