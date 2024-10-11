import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


from utils.metric import *
from dataset.data_util import MinMaxScaler
from utils.utils import mask_data_general



def test_model(test_dataloader, unet, short_samples_model, config, epoch, prototypes, device, logger, exp_dir):
    thresholds = [i for i in range(1000, 11000, 1000)]
    mtd_list = []
    mppe_list = []
    maepp_list = []
    maeps_list = []
    aptc_list = []
    avg_aptc_list = []
    max_td_list = []

    for abs_time, lat, lng in tqdm(test_dataloader, desc=f"Epoch {epoch} Test Progress"):
        testx = torch.stack([abs_time, lat, lng], dim=-1) #(batch_size, traj_length, 3)

        scaler = MinMaxScaler()
        scaler.fit(testx)
        testx = scaler.transform(testx)
        testx = testx.permute(0, 2, 1).to(device) #(batch_size, 3, traj_length)

        masked_testx = mask_data_general(testx) 
        masked_testx = masked_testx.permute(0, 2, 1).to(device) #(batch_size, traj_length, 3)

        with torch.no_grad():
            _, query_features = short_samples_model(masked_testx)
        # query_features = F.normalize(query_features, p=2, dim=1)
            cos_sim = F.cosine_similarity(query_features.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
            matched_prototypes = torch.matmul(cos_sim, prototypes).to(device)
            
            masked_testx = masked_testx.permute(0, 2, 1).to(device) #(batch_size, 3, traj_length)
            
            pred_x0 = unet(testx, masked_testx, matched_prototypes)

        pred_x0 = pred_x0.permute(0, 2, 1).to(device) #(batch_size, traj_length, 3)
        pred_x0 = scaler.inverse_transform(pred_x0.cpu()) #[0, 1]的reverse
        testx = testx.permute(0, 2, 1).to(device) #(batch_size, traj_length, 3)
        testx = scaler.inverse_transform(testx.cpu())
        pred_x0 = pred_x0.numpy()
        testx = testx.numpy()

        mtd_list.append(mean_trajectory_deviation(pred_x0, testx))
        mppe_list.append(mean_point_to_point_error(pred_x0, testx))
        maepp_list.append(mean_absolute_error_per_point(pred_x0[:, :, 0], testx[:, :, 0]))
        maeps_list.append(mean_absolute_error_per_sample(pred_x0[:, :, 0], testx[:, :, 0]))
        aptc, avg_aptc = trajectory_coverage(pred_x0, testx, thresholds)
        max_td_list.append(max_trajectory_deviation(pred_x0, testx))
        aptc_list.append(aptc)
        avg_aptc_list.append(avg_aptc)
        
        # Free up CPU memory
        # del pred_x0, testx
        # torch.cuda.empty_cache()

    mean_mtd = np.mean(mtd_list)
    mean_mppe = np.mean(mppe_list)
    mean_maepp = np.mean(maepp_list)
    mean_maeps = np.mean(maeps_list)
    mean_aptc = {k: np.mean([d[k] for d in aptc_list]) for k in aptc_list[0]}
    mean_avg_aptc = np.mean(avg_aptc_list)
    mean_max_td = np.max(max_td_list)

    logger.info(f"Epoch {epoch} Test mean MTD: {mean_mtd:.4f}")
    logger.info(f"Epoch {epoch} Test mean MPPE: {mean_mppe:.4f}")
    logger.info(f"Epoch {epoch} Test mean MAEPP: {mean_maepp:.4f}")
    logger.info(f"Epoch {epoch} Test mean MAEPS: {mean_maeps:.4f}")
    logger.info(f"Epoch {epoch} Test mean AVGTC: {mean_avg_aptc:.4f}")
    logger.info(f"Epoch {epoch} Test mean MaxTD: {mean_max_td:.4f}")
    for threshold, tc in mean_aptc.items():
        logger.info(f"Epoch {epoch} Test mean TC at threshold {threshold}: {tc:.4f}")

    def update_npy(file_path, data):
        if os.path.exists(file_path):
            existing_data = np.load(file_path, allow_pickle=True).item()
            existing_data[epoch] = data
        else:
            existing_data = {epoch: data}
        np.save(file_path, existing_data)

    update_npy(exp_dir / 'results' / 'Test_mean_mtd.npy', mean_mtd)
    update_npy(exp_dir / 'results' / 'Test_mean_mppe.npy', mean_mppe)
    update_npy(exp_dir / 'results' / 'Test_mean_maepp.npy', mean_maepp)
    update_npy(exp_dir / 'results' / 'Test_mean_maeps.npy', mean_maeps)
    update_npy(exp_dir / 'results' / 'Test_mean_avg_aptc.npy', mean_avg_aptc)
    update_npy(exp_dir / 'results' / 'Test_mean_max_td.npy', mean_max_td)
    for threshold, tc in mean_aptc.items():
        update_npy(exp_dir / 'results' / f"Test_mean_tc_threshold={threshold}.npy", tc)
