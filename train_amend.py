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

from utils.logger import Logger
from dataset.data_util import MinMaxScaler, TrajectoryDataset, PatternDataset
from utils.utils import IterativeKMeans, assign_labels, get_positive_negative_pairs, mask_data_general
from utils.metric import *
from diffProModel.loss import ContrastiveLoss
from diffProModel.protoTrans import TrajectoryTransformer
from diffProModel.Diffusion import Diffusion
from test import test_model


def ddp_setup(distributed):
    """Initialize the process group for distributed data parallel if distributed is True."""
    if distributed:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def get_data_directories(traj_path1):
    """Get the directories for training and testing data."""
    dir_list = os.listdir(traj_path1)
    dir_list.sort()
    train_dir_list = [os.path.join(traj_path1, dir_list[i]) for i in range(30)]
    test_dir_list = [os.path.join(traj_path1, dir_list[i]) for i in range(30, 33)]

    # train_dir_list = [os.path.join(traj_path1, dir_list[i]) for i in range(5)]
    # test_dir_list = [os.path.join(traj_path1, dir_list[i]) for i in range(5, 6)]
    return train_dir_list, test_dir_list


def setup_model_save_directory(exp_dir):
    """Set up the directory for saving model checkpoints."""
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    model_save = exp_dir / 'models' / (timestamp + '/')
    os.makedirs(model_save, exist_ok=True)
    return model_save


def preprocess_data(Patterns):
    """Preprocess the data by padding and scaling trajectories."""
    trajectories = Patterns.pad_trajectories()
    trajectories = torch.tensor(trajectories, dtype=torch.float32).clone().detach()
    scaler = MinMaxScaler()
    scaler.fit(trajectories)
    trajectories = scaler.transform(trajectories)
    trajectories = torch.tensor(trajectories, dtype=torch.float32).clone().detach()
    return trajectories


def initialize_model(config, device, trajectories, distributed):
    """Initialize the TrajectoryTransformer model and obtain the outputs for clustering."""
    ## 这里不需要梯度
    with torch.no_grad():
        seq_len = len(trajectories[0])
        dataset = TensorDataset(trajectories)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        model = TrajectoryTransformer(
            config.trans.input_dim, config.trans.embed_dim, config.trans.num_layers, 
            config.trans.num_heads, config.trans.forward_dim, seq_len, config.trans.dropout
        ).to(device)
        
        if distributed:
            model = DDP(model, device_ids=[device])

        all_outputs = []
        for batch in dataloader:
            batch = batch[0].to(device)
            output = model(batch)
            all_outputs.append(output.cpu().detach().numpy())
        all_outputs = np.concatenate(all_outputs, axis=0)

    ## release the memory
    del model
    return all_outputs



def main(config, logger, exp_dir):
    """Main function to run the training and testing pipeline."""
    distributed = config.training.dis_gpu
    ddp_setup(distributed)
    
    device = int(os.environ['LOCAL_RANK']) if distributed else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir_list, test_dir_list = get_data_directories(config.data.traj_path1)
    unet = Diffusion(loss_type="l3", config=config).to(device)

    lr = 2e-4
    losses = []
    model_save = setup_model_save_directory(exp_dir)

    train_dataset = TrajectoryDataset(train_dir_list, config.data.traj_length)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, drop_last=True, sampler=train_sampler)
    test_dataset = TrajectoryDataset(test_dir_list, config.data.traj_length)
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, drop_last=True, shuffle=False, num_workers=16)

    # seq_len = config.data.traj_length
    if distributed:
        unet = DDP(unet, device_ids=[device])
    short_samples_model = TrajectoryTransformer(
        config.training.batch_size, config.trans.input_dim, config.trans.embed_dim, config.trans.num_layers, 
        config.trans.num_heads, config.trans.forward_dim,  config.data.traj_length, config.trans.dropout
    ).to(device)
    
    if distributed:
        short_samples_model = DDP(short_samples_model, device_ids=[device])
        
    optim = torch.optim.AdamW(itertools.chain(unet.parameters(), short_samples_model.parameters()), lr=lr)
    losses_dict = {}  
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    
    for epoch in range(1, config.training.n_epochs + 1):
        previous_features = []
        logger.info("<----Epoch-{}---->".format(epoch))
        NUM_CLUSTER = 20
        kmeans = IterativeKMeans(NUM_CLUSTER, device)
        for batch_idx, (abs_time, lat, lng) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} Progress")):
            trainx = torch.stack([abs_time, lat, lng], dim=-1) #(batch_size, traj_length, 3)
            scaler = MinMaxScaler()
            scaler.fit(trainx)
            trainx = scaler.transform(trainx)
            trainx = trainx.to(device)

            prototypes_transformer, features = short_samples_model(trainx)
            #都先做归一化
            # prototypes_transformer = F.normalize(prototypes_transformer, p=2, dim=1) 
            # features = F.normalize(features, p=2, dim=1)
            
            if batch_idx == 0:
        # Kmeans, 如果是第一个batch，初始化prototypes, labels
                prototypes, _ = kmeans.fit(features)
            else:
                # 如果是之后的batch，更新prototypes, labels
                features_memory = torch.cat(previous_features, dim=0)
                prototypes, _ = kmeans.update(features, features_memory) #update的问题
            # prototypes = F.normalize(prototypes, p=2, dim=1)
            # 保存上一步的features
            previous_features.append(features) #列表
            
            trainx = trainx.permute(0, 2, 1) #(batch_size, 3, traj_length)
            x0 = trainx.to(device)

            masked_trainx = mask_data_general(x0)
            masked_trainx = masked_trainx.permute(0, 2, 1).to(device) #(batch_size, traj_length, 3)

            # 这里的作为条件1
            with torch.no_grad():
                _, query_features = short_samples_model(masked_trainx)
            # query_features = F.normalize(query_features, p=2, dim=1)
            cos_sim = F.cosine_similarity(query_features.unsqueeze(1), prototypes_transformer.unsqueeze(0), dim=-1) #amend at 7.14
            ## 将cos_sim 与 prototypes 通过矩阵乘法相乘或是爱因斯坦求和约定
            matched_prototypes = torch.matmul(cos_sim, prototypes_transformer).to(device) #(1, 512)

            # best_prototype_idx = torch.argmax(cos_sim, dim=-1)
            # matched_prototypes = prototypes[best_prototype_idx].to(device)
            
            #这里的作为条件2
            masked_trainx = masked_trainx.permute(0, 2, 1).to(device) #(batch_size, 3, traj_length)

            #1.对比损失
            positive_pairs, negative_pairs = get_positive_negative_pairs(prototypes_transformer, features)
            contrastive_loss = contrastive_loss_fn(features, positive_pairs, negative_pairs)

            ## 2.计算完contrastive_loss之后，再重新聚类辅助计算CE_loss
            initial_labels = assign_labels(prototypes_transformer, features)
            # new_labels = assign_labels(prototypes, features)
            new_labels = kmeans.predict(features)

            initial_labels = initial_labels.float()  
            new_labels = new_labels.float()  
            ce_loss = ce_loss_fn(initial_labels, new_labels)

            # 3.diffusion loss
            if distributed:
                diffusion_loss = unet.module.trainer(x0.float(), masked_trainx.float(), matched_prototypes.float(), weights=100.0)
            else:
                diffusion_loss = unet.trainer(x0.float(), masked_trainx.float(), matched_prototypes.float(), weights=100.0)
            loss = diffusion_loss + ce_loss + contrastive_loss
            losses.append(loss.item())  

            optim.zero_grad()
            loss.backward()
            optim.step()

            # Free up GPU memory
            del trainx, masked_trainx, query_features, cos_sim, matched_prototypes
            torch.cuda.empty_cache()
        losses_dict[epoch] = loss.item()
        print((f"Avg loss at epoch {epoch}: {loss.item():.4f}"))
        if (epoch) % 10 == 0:
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            transformer_path = model_save / f"transformer_{epoch}.pt"
            torch.save(short_samples_model.state_dict(), transformer_path)
            prototypes_path = model_save / f"prototypes_{epoch}.npy"
            np.save(prototypes_path, prototypes_transformer.detach().cpu().numpy())
            all_losses_path = exp_dir / 'results' / 'all_losses.npy'

            if os.path.exists(all_losses_path):
                existing_losses = np.load(all_losses_path, allow_pickle=True).item()
                existing_losses.update(losses_dict)
            else:
                existing_losses = losses_dict

            np.save(all_losses_path, existing_losses)
            # print((f"Avg loss at epoch {epoch}: {loss.item():.4f}"))
            test_model(test_dataloader, unet, short_samples_model, config, epoch, prototypes_transformer, device, logger, exp_dir)
    destroy_process_group()
