import os
import torch
import datetime
import shutil
from pathlib import Path
import argparse
from types import SimpleNamespace
from train_amend import main as train_main
from utils.logger import Logger, log_info
from conf.config import load_config
from utils.utils import set_seed

def setup_environment(config, timestamp, exp_dir):
    os.makedirs(exp_dir / 'results', exist_ok=True)
    os.makedirs(exp_dir / 'models', exist_ok=True)
    os.makedirs(exp_dir / 'logs', exist_ok=True)
    os.makedirs(exp_dir / 'Files', exist_ok=True)
    files_save = exp_dir / 'Files' / (timestamp + '/')
    os.makedirs(files_save, exist_ok=True)
    shutil.copy(__file__, files_save)
    shutil.copy('utils/utils.py', files_save)
    shutil.copy('train.py', files_save)
    shutil.copy('main.py', files_save)
    # shutil.copy('data.py', files_save)
    return files_save


if __name__ == "__main__":
    torch.cuda.set_device(2)
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args_global = parser.parse_args()
    world_size = torch.cuda.device_count()

    # Set the random seed
    set_seed(args_global.seed)


    config = load_config()
    for i in range(6, 7):
        config['data']['traj_length'] = i
        config['training']['n_epochs'] = 200
        config['training']['batch_size'] = 2560
        config["diffusion"]["num_diffusion_timesteps"] = 500
        config['model']['in_channels'] = 3
        config['model']['out_ch'] = 3
        # if i < 4:
        #     config['training']['batch_size'] = 5120
        # else:
        #     config['training']['batch_size'] = 10240

        temp = {k: SimpleNamespace(**v) for k, v in config.items()}
        config = SimpleNamespace(**temp)

        root_dir = Path(__file__).resolve().parent
        result_name = '{}_steps={}_len={}_{}_bs={}'.format(
            config.data.dataset, config.diffusion.num_diffusion_timesteps,
            config.data.traj_length, config.diffusion.beta_end,
            config.training.batch_size)
        exp_dir = root_dir / "Backups" / result_name
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        files_save = setup_environment(config, timestamp, exp_dir)
        
        logger = Logger(
            __name__,
            log_path=exp_dir / "logs" / (timestamp + '.log'),
            colorize=True,
        )
        log_info(config, logger)
        train_main(config, logger, exp_dir)