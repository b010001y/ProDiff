import os
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
import sys
from utils.logger import Logger
from dataset.data_util import MinMaxScaler, TrajectoryDataset
from utils.utils import IterativeKMeans, assign_labels, get_positive_negative_pairs, mask_data_general, get_data_paths
from diffProModel.loss import ContrastiveLoss
from diffProModel.protoTrans import TrajectoryTransformer
from diffProModel.Diffusion import Diffusion
from test import test_model # Import test_model


def ddp_setup(distributed):
    """Initialize the process group for distributed data parallel if distributed is True."""
    if distributed:
        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def setup_model_save_directory(exp_dir, timestamp):
    """Set up the directory for saving model checkpoints."""
    model_save_path = exp_dir / 'models' / (timestamp + '/')
    os.makedirs(model_save_path, exist_ok=True)
    return model_save_path


def lr_lambda_fn(current_epoch, warmup_epochs, total_epochs):
    if current_epoch < warmup_epochs:
        return float(current_epoch) / float(max(1, warmup_epochs))
    return 0.5 * (1. + torch.cos(torch.tensor(torch.pi * (current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs))))


def train_main(config, logger, exp_dir, timestamp_str):
    """Main function to run the training and testing pipeline for DDPM or DDIM."""
    distributed = config.training.dis_gpu
    local_rank = 0 # Default for non-DDP logging/master tasks
    #logger.info(config.training.validation_freq) # 1
    if distributed:
        ddp_setup(distributed) # This also calls torch.cuda.set_device(os.environ['LOCAL_RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
    else:
        device_id_to_use = config.device_id 
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id_to_use) 
            device = torch.device(f'cuda:{device_id_to_use}')
        else:
            device = torch.device('cpu')
    
    train_file_paths = get_data_paths(config.data, for_train=True)

    diffusion_model = Diffusion(loss_type=config.model.loss_type, config=config).to(device)

    lr = config.training.learning_rate
    model_save_dir = setup_model_save_directory(exp_dir, timestamp_str)

    train_dataset = TrajectoryDataset(train_file_paths, config.data.traj_length)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=(train_sampler is None),
                                num_workers=config.data.num_workers, 
                                drop_last=True, 
                                sampler=train_sampler,
                                pin_memory=True)

    # Create Test DataLoader
    test_file_paths = get_data_paths(config.data, for_train=False)
    test_dataset = TrajectoryDataset(test_file_paths, config.data.traj_length)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.sampling.batch_size, # Use sampling batch_size from config
                                 shuffle=False,
                                 num_workers=config.data.num_workers,
                                 drop_last=False, # Typically False for full test set evaluation
                                 pin_memory=True)

    if distributed:
        diffusion_model = DDP(diffusion_model, device_ids=[local_rank], find_unused_parameters=False)
    
    short_samples_model = TrajectoryTransformer(
        input_dim=config.trans.input_dim, 
        embed_dim=config.trans.embed_dim, 
        num_layers=config.trans.num_layers, 
        num_heads=config.trans.num_heads, 
        forward_dim=config.trans.forward_dim,  
        seq_len=config.data.traj_length, 
        n_cluster=config.trans.N_CLUSTER,
        dropout=config.trans.dropout 
    ).to(device)
    
    if distributed:
        short_samples_model = DDP(short_samples_model, device_ids=[local_rank], find_unused_parameters=False)
        
    optim = torch.optim.AdamW(itertools.chain(diffusion_model.parameters(), short_samples_model.parameters()), lr=lr, foreach=False)
    
    warmup_epochs = config.training.warmup_epochs
    total_epochs = config.training.n_epochs
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: lr_lambda_fn(epoch, warmup_epochs, total_epochs))

    losses_dict = {}  
    contrastive_loss_fn = ContrastiveLoss(margin=config.training.contrastive_margin)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(1, config.training.n_epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        epoch_losses = []
        previous_features_for_kmeans = []
        
        if local_rank == 0:
            logger.info(f"<----Epoch-{epoch}---->")
        
        kmeans = IterativeKMeans(num_clusters=config.trans.N_CLUSTER, device=device)

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", disable=(local_rank != 0))
        for batch_idx, (abs_time, lat, lng) in enumerate(pbar):
            trainx_raw = torch.stack([abs_time, lat, lng], dim=-1).to(device)
            
            scaler = MinMaxScaler()
            scaler.fit(trainx_raw)
            trainx_scaled = scaler.transform(trainx_raw)
            
            prototypes_from_transformer, features_for_kmeans_and_contrastive = short_samples_model(trainx_scaled)
            
            if not previous_features_for_kmeans:
                current_batch_prototypes_kmeans, _ = kmeans.fit(features_for_kmeans_and_contrastive.detach()) 
            else:
                features_memory = torch.cat(previous_features_for_kmeans, dim=0).detach()
                current_batch_prototypes_kmeans, _ = kmeans.update(features_for_kmeans_and_contrastive.detach(), features_memory)
            
            if len(previous_features_for_kmeans) < config.training.kmeans_memory_size:
                 previous_features_for_kmeans.append(features_for_kmeans_and_contrastive.detach())
            elif config.training.kmeans_memory_size > 0 :
                previous_features_for_kmeans.pop(0)
                previous_features_for_kmeans.append(features_for_kmeans_and_contrastive.detach())

            
            x0_for_diffusion = trainx_scaled.permute(0, 2, 1)

            masked_x0_condition_diffusion = mask_data_general(x0_for_diffusion) 
            masked_x0_permuted_for_ssm = masked_x0_condition_diffusion.permute(0, 2, 1) 

            with torch.no_grad():
                _, query_features_from_masked = short_samples_model(masked_x0_permuted_for_ssm)
            
            cos_sim = F.cosine_similarity(query_features_from_masked.unsqueeze(1), prototypes_from_transformer.unsqueeze(0), dim=-1)
            d_k = query_features_from_masked.size(-1)
            scaled_cos_sim = F.softmax(cos_sim / np.sqrt(d_k), dim=-1)
            matched_prototypes_for_diffusion = torch.matmul(scaled_cos_sim, prototypes_from_transformer)

            positive_pairs, negative_pairs = get_positive_negative_pairs(prototypes_from_transformer, features_for_kmeans_and_contrastive)
            contrastive_loss_val = contrastive_loss_fn(features_for_kmeans_and_contrastive, positive_pairs, negative_pairs)
            contrastive_loss_val = contrastive_loss_val * config.training.contrastive_loss_weight

            labels_from_transformer_protos = assign_labels(prototypes_from_transformer.detach(), features_for_kmeans_and_contrastive.detach()).long()
            labels_from_kmeans = kmeans.predict(features_for_kmeans_and_contrastive.detach()).long()
            
            ce_loss_val = torch.tensor(0.0, device=device)
            if config.training.ce_loss_weight > 0:
                logits_for_ce = features_for_kmeans_and_contrastive @ F.normalize(prototypes_from_transformer.detach(), dim=-1).T
                ce_loss_val = ce_loss_fn(logits_for_ce, labels_from_kmeans)                    
                ce_loss_val = ce_loss_val * config.training.ce_loss_weight

            diffusion_model_ref = diffusion_model.module if distributed else diffusion_model
            diffusion_loss_val = diffusion_model_ref.trainer(
                x0_for_diffusion.float(), 
                masked_x0_condition_diffusion.float(),
                matched_prototypes_for_diffusion.float(), 
                weights=config.training.diffusion_loss_weight
            )
            
            total_loss = diffusion_loss_val + ce_loss_val + contrastive_loss_val
        
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(diffusion_model.parameters(), short_samples_model.parameters()), max_norm=1.0)
            optim.step()

            epoch_losses.append(total_loss.item())
            if local_rank == 0:
                pbar.set_postfix({
                    'Loss': total_loss.item(), 
                    'Diff': diffusion_loss_val.item(), 
                    'Cont': contrastive_loss_val.item(), 
                    'CE': ce_loss_val.item(), 
                    'LR': optim.param_groups[0]['lr']
                })
            
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        losses_dict[epoch] = avg_epoch_loss
        scheduler.step()

        if local_rank == 0:
            logger.info(f"Epoch {epoch} Avg Loss: {avg_epoch_loss:.4f}")
            logger.info(f"Current LR: {optim.param_groups[0]['lr']:.6f}")
        
        if epoch % config.training.validation_freq == 0 and local_rank == 0:
            # Save model snapshot
            diffusion_state_dict = diffusion_model.module.state_dict() if distributed else diffusion_model.state_dict()
            transformer_state_dict = short_samples_model.module.state_dict() if distributed else short_samples_model.state_dict()
            
            torch.save(diffusion_state_dict, model_save_dir / f"diffusion_model_epoch_{epoch}.pt")
            torch.save(transformer_state_dict, model_save_dir / f"transformer_epoch_{epoch}.pt")
            
            if 'prototypes_from_transformer' in locals(): # Check if prototypes were generated in this epoch
                 np.save(model_save_dir / f"prototypes_transformer_epoch_{epoch}.npy", prototypes_from_transformer.detach().cpu().numpy())
            
            all_losses_path = exp_dir / 'results' / 'all_epoch_losses.npy'
            current_losses_to_save = {e: l for e, l in losses_dict.items()}
            if os.path.exists(all_losses_path):
                try:
                    existing_losses = np.load(all_losses_path, allow_pickle=True).item()
                    existing_losses.update(current_losses_to_save)
                    np.save(all_losses_path, existing_losses)
                except Exception as e:
                    if logger: logger.error(f"Error loading/updating losses file: {e}. Saving current losses only.")
                    np.save(all_losses_path, current_losses_to_save)
            else:
                np.save(all_losses_path, current_losses_to_save)
            if logger: logger.info(f"Saved model and prototypes snapshot at epoch {epoch} to {model_save_dir}")
            
            # Periodic validation call
            if logger: logger.info(f"--- Starting validation for epoch {epoch} ---")
            
            diffusion_model_to_test = diffusion_model.module if distributed else diffusion_model
            short_samples_model_to_test = short_samples_model.module if distributed else short_samples_model
            
            diffusion_model_to_test.eval() 
            short_samples_model_to_test.eval() 
            
            current_prototypes_for_test = short_samples_model_to_test.prototypes.detach()

            with torch.no_grad(): 
                test_model(
                    test_dataloader=test_dataloader,
                    diffusion_model=diffusion_model_to_test,
                    short_samples_model=short_samples_model_to_test,
                    config=config,
                    epoch=epoch, 
                    prototypes=current_prototypes_for_test,
                    device=device, 
                    logger=logger,
                    exp_dir=exp_dir
                )
            
            diffusion_model_to_test.train()
            short_samples_model_to_test.train()
            if logger: logger.info(f"--- Finished validation for epoch {epoch} ---")
            
    if distributed:
        destroy_process_group()
    if logger and local_rank == 0: # Ensure logger calls are rank-specific
        logger.info("Training finished.")
