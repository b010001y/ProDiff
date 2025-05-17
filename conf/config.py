"""
Configuration settings for the trajectory interpolation project.

This file defines a function `load_config()` which returns a dictionary 
containing various parameters grouped by their purpose (e.g., data, model, 
diffusion, training, sampling).
"""
from types import SimpleNamespace

def load_config():
    args = {
        'data': {
            'dataset': 'TKY',
            'traj_path1': './data/',
            'traj_length': 3,
            'channels': 2,
            'uniform_dequantization': False,
            'gaussian_dequantization': False,
            'num_workers': True,
        },
        'model': {
            'type': "simple",
            'attr_dim': 8,
            'guidance_scale': 2,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 2, 2],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.1,
            'var_type': 'fixedlarge',
            'resamp_with_conv': True,
        },
        'trans': {
            'input_dim': 3,
            'embed_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'forward_dim': 256,
            'dropout': 0.1,
            'N_CLUSTER': 20,
        },
        'diffusion': {
            'beta_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.05,
            'num_diffusion_timesteps': 500,
        },
        'training': {
            'batch_size': 256,
            'n_epochs': 100,
            'n_iters': 5000000,
            'snapshot_freq': 5000,
            'validation_freq': 10,
            'dis_gpu': False,
        },
        'sampling': {
            'batch_size': 64,
            'last_only': True,
        }
    }

    return args