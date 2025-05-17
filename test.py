import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from utils.metric import *
from dataset.data_util import MinMaxScaler
from utils.utils import mask_data_general, ddp_setup
# Diffusion model will be imported directly in the main script that calls this test function.
# from diffProModel.Diffusion import Diffusion # No, pass model as argument

def test_model(test_dataloader, diffusion_model, short_samples_model, config, epoch, 
               prototypes, device, logger, exp_dir):
    """
    Test the unified Diffusion model (DDPM or DDIM) on the test dataset.
    
    Args:
        test_dataloader: DataLoader for test data.
        diffusion_model: The unified diffusion model (instance of diffProModel.Diffusion.Diffusion).
        short_samples_model: Trajectory transformer model for feature extraction.
        config: Configuration object.
        epoch: Current epoch number (or identifier for the test run).
        prototypes: Prototype vectors (e.g., from TrajectoryTransformer or K-Means).
        device: Device to run the model on (already determined by the caller).
        logger: Logger object.
        exp_dir: Experiment directory path.
    """
    # Determine distributed status and local_rank first
    distributed = config.training.dis_gpu
    local_rank = 0
    if distributed:
        # If DDP is active, LOCAL_RANK should be set by the environment.
        # ddp_setup should have been called by the parent process (e.g., train_main or main for DDP launch)
        # test_model itself typically does not re-initialize DDP.
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        except ValueError:
            if logger: logger.warning("LOCAL_RANK environment variable not a valid integer. Defaulting to 0.")
            local_rank = 0 
    # The 'device' argument passed to this function should be the correct one to use.

    thresholds = [i for i in range(1000, 11000, 1000)] # Thresholds for TC metric
    # Initialize lists to store metrics for each batch
    mtd_list, mppe_list, maepp_list, maeps_list, aptc_list, avg_aptc_list, max_td_list = [], [], [], [], [], [], []

    # Get sampling parameters from config (assuming they are in config.sampling)
    sampling_type = getattr(config.sampling, 'type', 'ddpm') # Default to ddpm if not specified
    ddim_steps = getattr(config.sampling, 'ddim_steps', 50)
    ddim_eta = getattr(config.sampling, 'ddim_eta', 0.0)
    debug_mode = getattr(config, 'debug', False) # General debug flag

    if logger and local_rank == 0: # Ensure logger operations happen on rank 0 if distributed
        logger.info(f"Testing with sampling_type: {sampling_type} for epoch {epoch}")
        if sampling_type == 'ddim':
            logger.info(f"DDIM steps: {ddim_steps}, DDIM eta: {ddim_eta}")

    diffusion_model.eval() # Ensure diffusion model is in eval mode
    short_samples_model.eval() # Ensure feature extractor is in eval mode

    pbar_desc = f"Epoch {epoch} Test Progress ({sampling_type.upper()})"
    for batch_idx, (abs_time, lat, lng) in enumerate(tqdm(test_dataloader, desc=pbar_desc, disable=(local_rank != 0))):
        
        if debug_mode and logger and local_rank == 0:
            logger.info(f"Batch {batch_idx} - Input shapes: abs_time {abs_time.shape}, lat {lat.shape}, lng {lng.shape}")
            logger.info(f"Input data stats - abs_time: min={abs_time.min().item():.4f}, max={abs_time.max().item():.4f}, " +
                        f"lat: min={lat.min().item():.4f}, max={lat.max().item():.4f}, " +
                        f"lng: min={lng.min().item():.4f}, max={lng.max().item():.4f}")
        
        if torch.isnan(abs_time).any() or torch.isnan(lat).any() or torch.isnan(lng).any():
            if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in input data!")
            continue

        # Prepare input tensor (ground truth for start/end points and for scaling)
        # This testx_raw is used for scaler fitting and as test_x0 for diffusion model
        testx_raw = torch.stack([abs_time, lat, lng], dim=-1).to(device) 

        scaler = MinMaxScaler()
        scaler.fit(testx_raw) # Fit scaler on raw data (before permute)
        testx_scaled = scaler.transform(testx_raw) # Scale data
        
        if debug_mode and logger and local_rank == 0:
            logger.info(f"Scaler min: {scaler.min_val.flatten().cpu().numpy()}, max: {scaler.max_val.flatten().cpu().numpy()}")
        
        if torch.isnan(testx_scaled).any():
            if logger and local_rank == 0: 
                logger.error(f"Batch {batch_idx} - NaN detected after scaling!")
                if torch.any(scaler.max_val == scaler.min_val):
                    logger.error("Division by zero in scaler possible: max_val equals min_val for some features.")
            continue
            
        # Permute for diffusion model input: (batch_size, num_features, traj_length)
        testx_scaled_permuted = testx_scaled.permute(0, 2, 1)

        # Create masked input for conditioning (for feature extraction by short_samples_model)
        # mask_data_general expects (batch_size, num_features, traj_length)
        masked_condition_permuted = mask_data_general(testx_scaled_permuted)
        # short_samples_model expects (batch_size, traj_length, num_features)
        masked_condition_for_ssm = masked_condition_permuted.permute(0, 2, 1)

        with torch.no_grad():
            _, query_features = short_samples_model(masked_condition_for_ssm)
            
            if torch.isnan(query_features).any():
                if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in query_features!")
                continue
            if torch.isnan(prototypes).any():
                if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in provided prototypes!")
                continue
                
            # Match query features with prototypes (e.g., via cosine similarity and softmax attention)
            # This logic should align with how matched_prototypes are generated during training
            cos_sim = F.cosine_similarity(query_features.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
            if torch.isnan(cos_sim).any():
                if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in cos_sim!")
                continue
            
            # Using the same attention-weighted sum as in the unified training script
            d_k = query_features.size(-1)
            scaled_cos_sim = F.softmax(cos_sim / np.sqrt(d_k), dim=-1) 
            matched_prototypes_for_diffusion = torch.matmul(scaled_cos_sim, prototypes).to(device)
            
            if torch.isnan(matched_prototypes_for_diffusion).any():
                if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in matched_prototypes!")
                continue
            
            if debug_mode and logger and local_rank == 0:
                logger.info(f"Sampling with type: {sampling_type}, DDIM steps: {ddim_steps}, eta: {ddim_eta}")
                logger.info(f"Input to diffusion model (testx_scaled_permuted) shape: {testx_scaled_permuted.shape}, "
                            f"masked condition (masked_condition_permuted) shape: {masked_condition_permuted.shape}, "
                            f"matched prototypes shape: {matched_prototypes_for_diffusion.shape}")
            
            try:

                pred_x0_scaled = diffusion_model.sample(
                    test_x0=testx_scaled_permuted, # Ground truth (scaled) for start/end points and reference
                    attr=masked_condition_permuted,    # Masked data for conditional U-Net input (GuideNet attr)
                    prototype=matched_prototypes_for_diffusion, # Matched prototypes for GuideNet
                    sampling_type=sampling_type,
                    ddim_num_steps=ddim_steps,
                    ddim_eta=ddim_eta
                )
                
                if torch.isnan(pred_x0_scaled).any():
                    if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected in Diffusion model output!")
                    continue
                    
            except Exception as e:
                if logger and local_rank == 0: logger.error(f"Exception during Diffusion model sampling: {str(e)}")
                import traceback
                if logger and local_rank == 0: logger.error(traceback.format_exc())
                continue

        # pred_x0_scaled is (batch_size, num_features, traj_length)
        pred_x0_scaled_unpermuted = pred_x0_scaled.permute(0, 2, 1) 
        
        if debug_mode and logger and local_rank == 0:
            logger.info(f"pred_x0_scaled_unpermuted stats before inverse_transform: min={pred_x0_scaled_unpermuted.min().item():.4f}, max={pred_x0_scaled_unpermuted.max().item():.4f}")
        
        if (pred_x0_scaled_unpermuted < 0).any() or (pred_x0_scaled_unpermuted > 1).any():
            if logger and local_rank == 0: 
                logger.warning(f"Batch {batch_idx} - Values outside [0,1] in pred_x0_scaled: min={pred_x0_scaled_unpermuted.min().item():.4f}, max={pred_x0_scaled_unpermuted.max().item():.4f}. Clamping.")
            pred_x0_scaled_unpermuted = torch.clamp(pred_x0_scaled_unpermuted, 0, 1)
        
        # Inverse transform to original data scale - ensure this happens on the correct device
        pred_x0_final = scaler.inverse_transform(pred_x0_scaled_unpermuted) 
        
        ground_truth_final = testx_raw.cpu() 
        
        if torch.isnan(pred_x0_final).any() or torch.isnan(ground_truth_final).any():
            if logger and local_rank == 0: logger.error(f"Batch {batch_idx} - NaN detected after inverse transform!")
            continue
            
        # Move to CPU before converting to NumPy for metric calculation
        pred_x0_np = pred_x0_final.cpu().numpy()
        ground_truth_np = ground_truth_final.numpy()
        
        if debug_mode and logger and local_rank == 0:
            logger.info(f"Shapes for metrics: pred_x0_np {pred_x0_np.shape}, ground_truth_np {ground_truth_np.shape}")
            logger.info(f"pred_x0_np stats: min={np.min(pred_x0_np):.4f}, max={np.max(pred_x0_np):.4f}")
            logger.info(f"ground_truth_np stats: min={np.min(ground_truth_np):.4f}, max={np.max(ground_truth_np):.4f}")

        try:
            mtd_list.append(mean_trajectory_deviation(pred_x0_np, ground_truth_np))
            mppe_list.append(mean_point_to_point_error(pred_x0_np, ground_truth_np))
            maepp_list.append(mean_absolute_error_per_point(pred_x0_np[:, :, 0], ground_truth_np[:, :, 0]))
            maeps_list.append(mean_absolute_error_per_sample(pred_x0_np[:, :, 0], ground_truth_np[:, :, 0]))
            aptc_result, avg_aptc_result = trajectory_coverage(pred_x0_np, ground_truth_np, thresholds)
            aptc_list.append(aptc_result)
            avg_aptc_list.append(avg_aptc_result)
            max_td_list.append(max_trajectory_deviation(pred_x0_np, ground_truth_np))
        except Exception as e:
            if logger and local_rank == 0: logger.error(f"Exception during metric calculation in batch {batch_idx}: {str(e)}")
            if debug_mode and logger and local_rank == 0: import traceback; logger.error(traceback.format_exc())
            continue

        if debug_mode and batch_idx == 0 and os.environ.get('PROJECT_DEBUG_MODE', '0') == '1': # Use a distinct env var for this specific break
            if logger and local_rank == 0: logger.info("Project debug mode: Breaking after first test batch")
            break

    # Aggregate and log metrics (only on rank 0 if distributed)
    if local_rank == 0:
        mean_mtd = np.mean(mtd_list) if mtd_list else float('nan')
        mean_mppe = np.mean(mppe_list) if mppe_list else float('nan')
        mean_maepp = np.mean(maepp_list) if maepp_list else float('nan')
        mean_maeps = np.mean(maeps_list) if maeps_list else float('nan')
        mean_avg_aptc = np.mean(avg_aptc_list) if avg_aptc_list else float('nan')
        mean_max_td = np.max(max_td_list) if max_td_list else float('nan') # MaxTD is max over all samples
        mean_aptc_thresholds = {k: np.mean([d[k] for d in aptc_list if k in d]) for k in aptc_list[0]} if aptc_list else {f'TC@{thr}': float('nan') for thr in thresholds}

        if logger:
            logger.info(f"--- Test Results for Epoch {epoch} ({sampling_type.upper()}) ---")
            logger.info(f"Mean MTD: {mean_mtd:.4f}")
            logger.info(f"Mean MPPE: {mean_mppe:.4f}")
            logger.info(f"Mean MAEPP (time): {mean_maepp:.4f}")
            logger.info(f"Mean MAEPS (time): {mean_maeps:.4f}")
            logger.info(f"Mean AVG_TC: {mean_avg_aptc:.4f}")
            logger.info(f"Overall MaxTD: {mean_max_td:.4f}")
            for threshold_val, tc_val in mean_aptc_thresholds.items():
                logger.info(f"Mean {threshold_val}: {tc_val:.4f}")
            if sampling_type == 'ddim':
                 logger.info(f"DDIM sampling with {ddim_steps} steps, eta: {ddim_eta:.2f}")
            else:
                 logger.info(f"DDPM sampling with {config.diffusion.num_diffusion_timesteps} steps")

        # Save results to .npy files
        results_dir = exp_dir / 'results'
        os.makedirs(results_dir, exist_ok=True)
        sampling_prefix = f"{sampling_type.upper()}_"

        def save_metric_npy(metric_name, value, current_epoch):
            file_path = results_dir / f"{sampling_prefix}Test_mean_{metric_name}.npy"
            if np.isnan(value): return # Don't save if NaN
            if os.path.exists(file_path):
                try:
                    existing_data = np.load(file_path, allow_pickle=True).item()
                except: # Handle empty or corrupted file
                    existing_data = {}
                existing_data[current_epoch] = value
            else:
                existing_data = {current_epoch: value}
            np.save(file_path, existing_data)

        save_metric_npy('mtd', mean_mtd, epoch)
        save_metric_npy('mppe', mean_mppe, epoch)
        save_metric_npy('maepp', mean_maepp, epoch)
        save_metric_npy('maeps', mean_maeps, epoch)
        save_metric_npy('avg_aptc', mean_avg_aptc, epoch)
        save_metric_npy('max_td', mean_max_td, epoch)
        for threshold_key, tc_value in mean_aptc_thresholds.items():
            metric_key_name = threshold_key.replace('@', '_at_') # Sanitize for filename
            save_metric_npy(f"tc_{metric_key_name}", tc_value, epoch)
        
        if logger: logger.info(f"Saved test metrics to {results_dir}")
    
    # Ensure all processes finish if in DDP, though testing is usually single-process or rank 0 handles results
    if torch.distributed.is_initialized():
        torch.distributed.barrier() # Wait for all processes if any were involved

    return { # Return main metrics, could be useful for main script
        "mean_mtd": mean_mtd,
        "mean_mppe": mean_mppe
    } if local_rank == 0 else {} 