from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
import torch
import os
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
import random
def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


def time_warping(x, length=200):
    """
    Resamples a trajectory to a new length.
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    warped_trajectory = np.zeros((2, length))
    for i in range(2):
        warped_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return warped_trajectory.T


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    # get mean and variance of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also return noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.cuda(), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def divide_grids(boundary, grids_num):
    lati_min, lati_max = boundary['lati_min'], boundary['lati_max']
    long_min, long_max = boundary['long_min'], boundary['long_max']
    # Divide the latitude and longitude into grids_num intervals.
    lati_interval = (lati_max - lati_min) / grids_num
    long_interval = (long_max - long_min) / grids_num
    # Create arrays of latitude and longitude values.
    latgrids = np.arange(lati_min, lati_max, lati_interval)
    longrids = np.arange(long_min, long_max, long_interval)
    return latgrids, longrids


# calculate the distance between two points
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def destroy_process_group():
    destroy_process_group()


import torch

class IterativeKMeans:
    def __init__(self, num_clusters, device, num_iters=100, tol=1e-4):
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.tol = tol
        self.cluster_centers = None
        self.labels = None
        self.device = device

    def fit(self, X):
        # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.clone().detach().to(self.device)
        num_samples, num_features = X.shape
        indices = torch.randperm(num_samples)[:self.num_clusters]
        self.cluster_centers = X[indices].clone().detach()
        self.labels = torch.argmin(torch.cdist(X, self.cluster_centers), dim=1).cpu().numpy()

        for _ in range(self.num_iters):
            distances = torch.cdist(X, self.cluster_centers)
            labels = torch.argmin(distances, dim=1)
            # new_cluster_centers = torch.stack([X[labels == i].mean(dim=0) for i in range(self.num_clusters)])
            new_cluster_centers = torch.stack([X[labels == i].mean(dim=0) if (labels == i).sum() > 0 else self.cluster_centers[i] for i in range(self.num_clusters)])
            center_shift = torch.norm(new_cluster_centers - self.cluster_centers, dim=1).sum().item()
            if center_shift < self.tol:
                break
            self.cluster_centers = new_cluster_centers

        self.labels = labels.cpu().numpy()
        return self.cluster_centers, self.labels

    def update(self, new_X, original_X):
        combined_X = torch.cat([original_X, new_X], dim=0)
        combined_X = combined_X.clone().detach().to(self.device)

        for _ in range(self.num_iters):
            distances = torch.cdist(combined_X, self.cluster_centers)
            labels = torch.argmin(distances, dim=1)
            new_cluster_centers = torch.stack([combined_X[labels == i].mean(dim=0) if (labels == i).sum() > 0 else self.cluster_centers[i] for i in range(self.num_clusters)])
            center_shift = torch.norm(new_cluster_centers - self.cluster_centers, dim=1).sum().item()
            if center_shift < self.tol:
                break
            self.cluster_centers = new_cluster_centers

        self.labels = labels.cpu().numpy()
        return self.cluster_centers, self.labels

    def predict(self, X):
        # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.clone().detach().to(self.device)
        distances = torch.cdist(X, self.cluster_centers)
        labels = torch.argmin(distances, dim=1)
        return labels

    def to(self, device):
        self.device = device
        if self.cluster_centers is not None:
            self.cluster_centers = self.cluster_centers.to(device)
        return self


def assign_labels(prototypes, features):
    # Calculate pairwise distances between all features and prototypes
    distances = F.pairwise_distance(features.unsqueeze(1), prototypes.unsqueeze(0))
    # Find the index of the prototype with the minimum distance (on the second dimension)
    labels = torch.argmin(distances, dim=-1)

    return labels


def get_positive_negative_pairs(prototypes, samples):
    positive_pairs = []
    negative_pairs = []
    for sample in samples:
        distances = F.pairwise_distance(sample.unsqueeze(0), prototypes)
        pos_idx = torch.argmin(distances).item()
        neg_idx = torch.argmax(distances).item()
        positive_pairs.append(prototypes[pos_idx])
        negative_pairs.append(prototypes[neg_idx])
    return torch.stack(positive_pairs), torch.stack(negative_pairs)



def mask_data_general(x: torch.Tensor):
    """Mask the input data""" 
    mask = torch.ones_like(x)
    mask[:, :, 1:-1] = 0
    return x * mask.float()

def update_npy(file_path, data):
    if os.path.exists(file_path):
        existing_data = np.load(file_path, allow_pickle=True).item()
        existing_data.update(data)
    else:
        existing_data = data
    np.save(file_path, existing_data)


def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r * 1000  # Return distance in meters

def get_data_paths(data_config, for_train=True):
    """Get the file paths for training or testing data for TKY-like structure.
    Assumes data_config.traj_path1 points to a directory containing train.h5 and test.h5.
    """
    base_path = data_config.traj_path1
    if not isinstance(base_path, str):
        base_path = str(base_path)

    if for_train:

        file_path = os.path.join(base_path, "train.h5")
    else:
        file_path = os.path.join(base_path, "test.h5")
    

    return [file_path]