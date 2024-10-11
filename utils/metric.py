import numpy as np
from .utils import haversine

def mean_absolute_error_per_point(pred, true):
    """
    计算批次的平均每点时间绝对误差 (Mean Absolute Error Per Point, MAEPP)
    :param pred: 预测的时间，形状为 (batch_size, traj_length)
    :param true: 真实的时间，形状为 (batch_size, traj_length)
    :return: 批次的平均每点时间绝对误差 (MAEPP)
    """
    maepp = np.abs(pred - true).mean()
    return maepp

def mean_absolute_error_per_sample(pred, true):
    """
    计算批次的平均每样本时间绝对误差 (Mean Absolute Error Per Sample, MAEPS)
    :param pred: 预测的时间，形状为 (batch_size, traj_length)
    :param true: 真实的时间，形状为 (batch_size, traj_length)
    :return: 批次的平均每样本时间绝对误差 (MAEPS)
    """
    mae_per_sample = np.abs(pred - true).mean(axis=1)
    maeps = mae_per_sample.mean()
    return maeps

def mean_trajectory_deviation(pred, true):
    """
    计算批次的平均轨迹偏差 (Mean Trajectory Deviation, MTD)
    :param pred: 预测的轨迹，形状为 (batch_size, 2, traj_length)
    :param true: 真实的轨迹，形状为 (batch_size, 2, traj_length)
    :return: 批次的平均轨迹偏差 (MTD)
    """
    batch_size, traj_length, _ = pred.shape
    deviations = []
    for i in range(batch_size):
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2]
        true_lat, true_lon = true[i, :, 1], true[i, :, 2]
        deviation = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        deviations.append(np.mean(deviation))
    mtd = np.mean(deviations)
    return mtd

def mean_point_to_point_error(pred, true):
    """
    计算批次的平均点对点误差 (Mean Point-to-Point Error, MPPE)
    :param pred: 预测的轨迹，形状为 (batch_size, traj_length, 2)
    :param true: 真实的轨迹，形状为 (batch_size, traj_length, 2)
    :return: 批次的平均点对点误差 (MPPE)
    """
    batch_size, traj_length, _ = pred.shape
    total_error = 0
    for i in range(batch_size):
        for j in range(traj_length):
            pred_lat, pred_lon = pred[i, j, 1], pred[i, j, 2]
            true_lat, true_lon = true[i, j, 1], true[i, j, 2]
            point_error = haversine(pred_lat, pred_lon, true_lat, true_lon)
            total_error += point_error
    mppe = total_error / (batch_size * traj_length)
    return mppe

def trajectory_coverage(pred, true, thresholds):
    """
    计算每个样本在多个阈值下的轨迹覆盖率 (Trajectory Coverage, TC)
    :param pred: 预测的轨迹，形状为 (batch_size, 2, traj_length)
    :param true: 真实的轨迹，形状为 (batch_size, 2, traj_length)
    :param thresholds: 偏差阈值列表
    :return: 每个样本在各个阈值下的轨迹覆盖率字典，以及平均轨迹覆盖率 (APTC)
    """
    batch_size, traj_length, _ = pred.shape
    tc_dict = {f'TC@{threshold}': [] for threshold in thresholds}
    for i in range(batch_size):
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2]
        true_lat, true_lon = true[i, :, 1], true[i, :, 2]
        deviations = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        for threshold in thresholds:
            tc = (deviations <= threshold).mean() #tc = deviations.mean() <= threshold 这里先不改，之后整体再改
            tc_dict[f'TC@{threshold}'].append(tc)
    aptc = {k: np.mean(v) for k, v in tc_dict.items()}
    avg_aptc = np.mean(list(aptc.values()))
    return aptc, avg_aptc

def max_trajectory_deviation(pred, true):
    """
    计算每个样本的最大轨迹偏差 (Maximum Trajectory Deviation, MaxTD)
    :param pred: 预测的轨迹，形状为 (batch_size, 2, traj_length)
    :param true: 真实的轨迹，形状为 (batch_size, 2, traj_length)
    :return: 批次的最大轨迹偏差 (MaxTD)
    """
    batch_size, traj_length, _ = pred.shape
    max_deviations = []
    for i in range(batch_size):
        pred_lat, pred_lon = pred[i, :, 1], pred[i, :, 2]
        true_lat, true_lon = true[i, :, 1], true[i, :, 2]
        deviation = np.array([haversine(pred_lat[j], pred_lon[j], true_lat[j], true_lon[j]) for j in range(traj_length)])
        max_deviations.append(np.max(deviation))
    max_td = np.max(max_deviations)
    return max_td