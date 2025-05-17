import os
import torch
import datetime
import shutil
from pathlib import Path
import argparse
from types import SimpleNamespace
import sys
import numpy as np


from conf import config as config_module # Corrected alias
from utils.logger import Logger, log_info
from utils.utils import set_seed, ddp_setup, destroy_process_group, get_data_paths
from dataset.data_util import TrajectoryDataset
from torch.utils.data import DataLoader


from train import train_main 
from test import test_model

def setup_experiment_environment(base_exp_dir, exp_name_with_timestamp, config_to_save, files_to_copy=None):
    """Sets up the experiment directory structure and saves essential files."""
    exp_dir = base_exp_dir / exp_name_with_timestamp
    results_dir = exp_dir / 'results'
    models_dir = exp_dir / 'models' # Unified models dir, not timestamped sub-dir by default here
    logs_dir = exp_dir / 'logs'
    code_save_dir = exp_dir / 'code_snapshot'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(code_save_dir, exist_ok=True)

    # Save configuration
    # (Convert SimpleNamespace to dict for easier saving if needed, or save as text)
    with open(exp_dir / 'config_used.txt', 'w') as f:
        import json
        # Convert SimpleNamespace to dict for JSON serialization
        def ns_to_dict(ns):
            if isinstance(ns, SimpleNamespace):
                return {k: ns_to_dict(v) for k, v in ns.__dict__.items()}
            elif isinstance(ns, dict):
                return {k: ns_to_dict(v) for k, v in ns.items()}
            elif isinstance(ns, list):
                return [ns_to_dict(i) for i in ns]
            return ns
        config_dict = ns_to_dict(config_to_save)
        json.dump(config_dict, f, indent=4)

    # Copy essential code files
    if files_to_copy:
        for file_path_str in files_to_copy:
            try:
                file_path = Path(file_path_str)
                if file_path.exists():
                    shutil.copy(file_path, code_save_dir)
                else:
                    print(f"Warning: File to copy not found: {file_path_str}") # Use logger if available
            except Exception as e:
                print(f"Warning: Could not copy file {file_path_str}: {e}")
    
    return exp_dir, models_dir, logs_dir, results_dir

def main():
    parser = argparse.ArgumentParser(description='Unified Trajectory Interpolation - Training with Periodic Validation')
    parser.add_argument('--sampling_type', type=str, default='ddpm', choices=['ddpm', 'ddim'], 
                        help='Diffusion sampling type (ddpm or ddim) - influences periodic validation if DDIM is chosen, and experiment naming.')
    parser.add_argument('--config_module_path', type=str, default='conf.config', 
                        help='Python module path for base configuration (e.g., conf.config)')
    parser.add_argument('--exp_name', type=str, default='traj_interp_exp', 
                        help='Base name for the experiment directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID to use')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training (DDP)')
    
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps for periodic validation')
    parser.add_argument('--ddim_eta', type=float, default=0.0, 
                        help='DDIM stochasticity parameter for periodic validation (0=deterministic, 1=DDPM-like)')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for more detailed logs')

    args = parser.parse_args()

    # --- Basic Setup ---
    if args.distributed:
        ddp_setup(args.distributed) # Sets LOCAL_RANK env var if not already set by launcher
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        local_rank = 0
    
    if not args.distributed or local_rank == 0: # Setup master process first or if not distributed
        print(f"Running on device: cuda:{args.device_id}" if torch.cuda.is_available() else "Running on CPU")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id if not args.distributed else local_rank)
    
    set_seed(args.seed + local_rank) # Ensure different seeds for different processes in DDP for some operations

    # --- Load Configuration ---
    try:

        base_config_dict = config_module.load_config() # from conf.config import load_config
    except Exception as e:
        print(f"Error loading base configuration from {args.config_module_path}: {e}")
        sys.exit(1)

    cfg_ns = {k: SimpleNamespace(**v) for k, v in base_config_dict.items()}
    config = SimpleNamespace(**cfg_ns)

    # Update config with command-line arguments
    config.debug = args.debug
    config.training.dis_gpu = args.distributed
    config.sampling.type = args.sampling_type
    config.sampling.ddim_steps = args.ddim_steps
    config.sampling.ddim_eta = args.ddim_eta
    config.device_id = args.device_id # Pass device_id for train_main
    # Ensure other necessary fields exist in config (add defaults if not in config.py)
    if not hasattr(config, 'model'): config.model = SimpleNamespace()
    if not hasattr(config.model, 'loss_type'): config.model.loss_type = 'l1' # Default
    if not hasattr(config.training, 'learning_rate'): config.training.learning_rate = 2e-4
    if not hasattr(config.training, 'warmup_epochs'): config.training.warmup_epochs = 10
    if not hasattr(config.training, 'contrastive_margin'): config.training.contrastive_margin = 1.0
    if not hasattr(config.training, 'use_amp'): config.training.use_amp = True 
    if not hasattr(config.training, 'kmeans_memory_size'): config.training.kmeans_memory_size = 10 # Batches
    if not hasattr(config.training, 'ce_loss_weight'): config.training.ce_loss_weight = 0.1
    if not hasattr(config.training, 'diffusion_loss_weight'): config.training.diffusion_loss_weight = 1.0
    if not hasattr(config.training, 'contrastive_loss_weight'): config.training.contrastive_loss_weight = 1.0

    # --- Setup Experiment Environment (only on rank 0 if DDP) ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Include sampling type in experiment name for clarity
    exp_name_ts = f"{args.exp_name}_{config.data.dataset}_len{config.data.traj_length}_{args.sampling_type}_{timestamp_str}"
    
    exp_dir, models_save_dir, logs_dir, results_dir = Path("."), Path("."), Path("."), Path(".") # Defaults for non-rank0
    if local_rank == 0:
        root_dir = Path(__file__).resolve().parent # Project root
        base_experiment_path = root_dir / "Experiments" # Changed from "Backups"
        
        files_to_copy_snapshot = [
            'main.py', 'train.py', 'test.py', 'conf/config.py',
            'diffProModel/Diffusion.py', 'diffProModel/protoTrans.py', 'diffProModel/loss.py',
            'utils/utils.py', 'utils/logger.py', 'utils/metric.py', 'dataset/data_util.py'
        ]
        exp_dir, models_save_dir, logs_dir, results_dir = setup_experiment_environment(
            base_experiment_path, exp_name_ts, config, files_to_copy_snapshot
        )
    
    # Logger setup (after exp_dir is known by all processes if DDP, or just for rank 0)
    logger = None
    if local_rank == 0:
        log_file_path = logs_dir / f"log_{timestamp_str}.txt"
        logger = Logger(
            name=exp_name_ts,
            log_path=log_file_path,
            colorize=True,
            level="debug" if args.debug else "info"
        )
        logger.info(f"Experiment directory: {exp_dir}")
        log_info(config, logger) # Log the configuration details
        logger.info(f"Using sampling type for periodic validation: {args.sampling_type}")
        if args.sampling_type == 'ddim':
            logger.info(f"DDIM Steps for validation: {args.ddim_steps}, Eta for validation: {args.ddim_eta}")

    # Barrier to ensure exp_dir is created by rank 0 before other ranks proceed if DDP
    if args.distributed:
        torch.distributed.barrier()

    # --- Main Execution: Call Training (which includes periodic validation) ---
    if logger and local_rank == 0: 
        logger.info("Starting training with periodic validation...")
    
    train_main(config, logger, exp_dir, timestamp_str) 
    
    if args.distributed:
        if torch.distributed.is_initialized():
            destroy_process_group()
    
    if local_rank == 0 and logger: 
        logger.info("Main script execution finished.")

if __name__ == "__main__":
    main() 