from types import SimpleNamespace

def load_config():
    args = {
        'data': {
            'dataset': 'WuXi',
            'traj_path1': '//data4/lvxin/butianci/butianci/data/WUXI_10days_Minutes/',
            'head_path2': '/xxxxxxx',
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
            'in_channels': 2,
            'out_ch': 2,
            'ch': 128,
            'ch_mult': [1, 2, 2, 2],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.1,
            'var_type': 'fixedlarge',
            'ema_rate': 0.9999,
            'ema': True,
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
            'validation_freq': 2000,
            'dis_gpu': False,
        },
        'sampling': {
            'batch_size': 64,
            'last_only': True,
        }
    }

    return args