import numpy as np
from .utils import haversine

def mean_absolute_error_per_point(pred, true):
    """
    Calculates the Mean Absolute Error Per Point (MAEPP) for a batch.
    :param pred: Predicted time, shape (batch_size, traj_length)
    :param true: Ground truth time, shape (batch_size, traj_length)
    :return: Mean Absolute Error Per Point (MAEPP) for the batch.
    """
    maepp = np.abs(pred - true).mean()
    return maepp

def mean_absolute_error_per_sample(pred, true):
    """
    Calculates the Mean Absolute Error Per Sample (MAEPS) for a batch.
    :param pred: Predicted time, shape (batch_size, traj_length)
    :param true: Ground truth time, shape (batch_size, traj_length)
    :return: Mean Absolute Error Per Sample (MAEPS) for the batch.
    """
    mae_per_sample = np.abs(pred - true).mean(axis=1)
    maeps = mae_per_sample.mean()
    return maeps

def mean_trajectory_deviation(pred, true):
    """
    Calculates the Mean Trajectory Deviation (MTD) for a batch.
    :param pred: Predicted trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :param true: Ground truth trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :return: Mean Trajectory Deviation (MTD) for the batch.
    """
    batch_size, traj_length, _ = pred.shape # Assuming pred shape is (batch_size, traj_length, num_features)
    deviations = []
    for i in range(batch_size):
        # Assuming lat is at index 1 and lon is at index 2 if num_features is 3,
        # or lat is index 0 and lon is index 1 if num_features is 2 (after potential permute)
        # The original code used pred[i, :, 1] and pred[i, :, 2] which might imply features are [time, lat, lon]
        # and slicing was done after permuting to (batch_size, num_coords, traj_length).
        # For (batch_size, traj_length, num_features), access directly.
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2] # Adapt if lat/lon indices are different
        true_lat, true_lon = true[i, :, 1], true[i, :, 2] # Adapt if lat/lon indices are different
        deviation = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        deviations.append(np.mean(deviation))
    mtd = np.mean(deviations)
    return mtd

def mean_point_to_point_error(pred, true):
    """
    Calculates the Mean Point-to-Point Error (MPPE) for a batch.
    :param pred: Predicted trajectories, shape (batch_size, traj_length, 2) or (batch_size, traj_length, 3)
    :param true: Ground truth trajectories, shape (batch_size, traj_length, 2) or (batch_size, traj_length, 3)
    :return: Mean Point-to-Point Error (MPPE) for the batch.
    """
    batch_size, traj_length, _ = pred.shape
    total_error = 0
    for i in range(batch_size):
        for j in range(traj_length):
            pred_lat, pred_lon = pred[i, j, 1], pred[i, j, 2] # Adapt if lat/lon indices are different
            true_lat, true_lon = true[i, j, 1], true[i, j, 2] # Adapt if lat/lon indices are different
            point_error = haversine(pred_lat, pred_lon, true_lat, true_lon)
            total_error += point_error
    mppe = total_error / (batch_size * traj_length)
    return mppe

def trajectory_coverage(pred, true, thresholds):
    """
    Calculates Trajectory Coverage (TC) for each sample at multiple thresholds.
    :param pred: Predicted trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :param true: Ground truth trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :param thresholds: List of deviation thresholds.
    :return: A dictionary of trajectory coverage for each sample at various thresholds,
             and the average trajectory coverage (APTC).
    """
    batch_size, traj_length, _ = pred.shape
    tc_dict = {f'TC@{threshold}': [] for threshold in thresholds}
    for i in range(batch_size):
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2] # Adapt if lat/lon indices are different
        true_lat, true_lon = true[i, :, 1], true[i, :, 2] # Adapt if lat/lon indices are different
        deviations = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        for threshold in thresholds:
            tc = (deviations <= threshold).mean() # Original comment: tc = deviations.mean() <= threshold, this seems more standard.
            tc_dict[f'TC@{threshold}'].append(tc)
    aptc = {k: np.mean(v) for k, v in tc_dict.items()}
    avg_aptc = np.mean(list(aptc.values()))
    return aptc, avg_aptc

def max_trajectory_deviation(pred, true):
    """
    Calculates the Maximum Trajectory Deviation (MaxTD) for each sample in a batch.
    :param pred: Predicted trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :param true: Ground truth trajectories, shape (batch_size, 2, traj_length) or (batch_size, traj_length, 2/3)
    :return: Maximum Trajectory Deviation (MaxTD) for the batch.
    """
    batch_size, traj_length, _ = pred.shape
    max_deviations = []
    for i in range(batch_size):
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2] # Adapt if lat/lon indices are different
        true_lat, true_lon = true[i, :, 1], true[i, :, 2] # Adapt if lat/lon indices are different
        deviation = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        max_deviations.append(np.max(deviation))
    max_td = np.max(max_deviations)
    return max_td