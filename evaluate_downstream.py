"""
evaluate_downstream.py
Evaluation script for trajectory prediction and completion tasks.
Adapted for Yasoz/UniTraj, supporting batch-wise haversine metric calculation
(MAE, RMSE in meters) compatible with the repository's data pipeline.

Usage:
    python evaluate_downstream.py --mode prediction --mask_last_n 5 --data_dir ./data
    python evaluate_downstream.py --mode completion --mask_ratio 0.3 --model_path ./checkpoints/model.pt
"""
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader

from conf import config as config_module
from utils.logger import Logger
from utils.utils import haversine, get_data_paths
from dataset.data_util import TrajectoryDataset, MinMaxScaler

def dict_to_ns(d):
    """Recursively convert nested dicts to SimpleNamespace."""
    return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})

def calculate_haversine_metrics(pred_lat, pred_lon, true_lat, true_lon):
    """
    Calculate MAE and RMSE in meters using haversine distance.
    Args:
        pred_lat, pred_lon, true_lat, true_lon: numpy arrays of shape (N,) or (N, T)
    Returns:
        mae, rmse in meters
    """
    p_lat, p_lon = pred_lat.flatten(), pred_lon.flatten()
    t_lat, t_lon = true_lat.flatten(), true_lon.flatten()

    distances = haversine(p_lat, p_lon, t_lat, t_lon)
    mae = np.mean(distances)
    rmse = np.sqrt(np.mean(distances**2))
    return mae, rmse

def mask_prediction(data, mask_last_n=5):
    """Mask the last N points for lat/lon (indices 1 and 2 in time, lat, lon)."""
    masked = data.clone()
    if mask_last_n > 0:
        masked[:, -mask_last_n:, 1:] = 0.0
    return masked

def mask_completion(data, mask_ratio=0.5, seed=None):
    """Mask arbitrary indices based on ratio."""
    if seed is not None:
        np.random.seed(seed)
    B, T, _ = data.shape
    mask = torch.rand(B, T, device=data.device) > mask_ratio
    mask = mask.unsqueeze(-1).expand_as(data)
    return data * mask.float()

def main():
    parser = argparse.ArgumentParser(description="Downstream Evaluation: Trajectory Prediction & Completion")
    parser.add_argument("--mode", type=str, default="prediction", choices=["prediction", "completion"],
                        help="Evaluation mode: 'prediction' (mask last N) or 'completion' (mask arbitrary)")
    parser.add_argument("--mask_last_n", type=int, default=5, help="Number of trailing points to mask for prediction")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="Fraction of points to mask for completion")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--traj_length", type=int, default=20, help="Length of trajectory sequences")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--out_file", type=str, default="eval_results.json", help="Output file for metrics")
    args = parser.parse_args()

    # Load and patch config
    cfg_dict = config_module.load_config()
    config = dict_to_ns(cfg_dict)
    config.data.traj_path1 = args.data_dir
    config.data.traj_length = args.traj_length

    logger = Logger(name="downstream_eval", level="info")
    logger.info(f"Initializing {args.mode} evaluation pipeline...")

    # 1. Load Dataset
    file_paths = get_data_paths(config.data, for_train=False)
    dataset = TrajectoryDataset(file_paths, config.data.traj_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # 2. Fit Normalization Scaler
    logger.info("Fitting MinMaxScaler on dataset...")
    scaler = MinMaxScaler()
    all_batches = []
    for h, lat, lon in dataset:
        all_batches.append(torch.stack([h, lat, lon], dim=-1))
    all_data = torch.stack(all_batches, dim=0)
    scaler.fit(all_data)
    logger.info(f"Scaler min: {scaler.min_val.mean():.4f}, max: {scaler.max_val.mean():.4f}")

    # 3. Model Loading (Optional)
    model = None
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}...")
        try:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                logger.info("Checkpoint loaded. Model instantiation should be uncommented below for inference.")
            else:
                logger.warning("Checkpoint format unrecognized. Skipping model inference.")
                model = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning("No model path provided. Using identity mapping (masked input) for metric pipeline verification.")

    # 4. Evaluation Loop
    total_mae = 0.0
    total_rmse = 0.0
    total_eval_points = 0
    num_batches = 0

    logger.info(f"Starting evaluation on {args.device}...")
    for hours, lats, lons in dataloader:
        B, T = lats.shape
        data = torch.stack([hours, lats, lons], dim=-1).to(args.device)

        # Apply masking
        if args.mode == "prediction":
            masked_data = mask_prediction(data, args.mask_last_n)
            eval_indices = slice(-args.mask_last_n, None)
        else:
            masked_data = mask_completion(data, args.mask_ratio)
            eval_indices = slice(None)

        # Inference
        with torch.no_grad():
            if model is not None:
                # pred_data = model(masked_data)  # Uncomment when model is defined
                pred_data = masked_data.clone()  # Fallback
            else:
                pred_data = masked_data.clone()

        # Denormalize
        true_denorm = scaler.inverse_transform(data)
        pred_denorm = scaler.inverse_transform(pred_data)

        # Extract lat/lon (indices 1 and 2)
        true_lat, true_lon = true_denorm[:, eval_indices, 1].cpu().numpy(), true_denorm[:, eval_indices, 2].cpu().numpy()
        pred_lat, pred_lon = pred_denorm[:, eval_indices, 1].cpu().numpy(), pred_denorm[:, eval_indices, 2].cpu().numpy()

        # Calculate Metrics
        mae, rmse = calculate_haversine_metrics(pred_lat, pred_lon, true_lat, true_lon)

        batch_points = true_lat.size
        total_mae += mae * batch_points
        total_rmse += rmse * batch_points
        total_eval_points += batch_points

        num_batches += 1
        logger.info(f"Batch {num_batches:03d} | MAE: {mae:.4f} m | RMSE: {rmse:.4f} m")

    # 5. Aggregate & Save Results
    avg_mae = total_mae / total_eval_points
    avg_rmse = total_rmse / total_eval_points

    logger.info("Evaluation Complete.")
    logger.info(f"Overall {args.mode.capitalize()} Metrics -> MAE: {avg_mae:.4f} m, RMSE: {avg_rmse:.4f} m")

    results = {
        "mode": args.mode,
        "mask_config": {"last_n": args.mask_last_n} if args.mode == "prediction" else {"ratio": args.mask_ratio},
        "MAE_meters": float(avg_mae),
        "RMSE_meters": float(avg_rmse),
        "dataset_size": len(dataset),
        "eval_points": int(total_eval_points)
    }

    out_path = Path(args.out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
