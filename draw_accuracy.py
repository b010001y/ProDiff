import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import os

def load_npy_files(exp_dir, metric_prefix, start_epoch=10, end_epoch=200):
    values = {}
    filename = f'{metric_prefix}.npy'
    filepath = os.path.join(exp_dir, filename)
    if os.path.isfile(filepath):
        data = np.load(filepath, allow_pickle=True).item()
        for epoch in range(start_epoch, end_epoch + 1, 10):
            if epoch in data:
                values[epoch] = data[epoch]
    return values

def find_max_metric(values):
    max_epoch = max(values, key=values.get)
    max_value = values[max_epoch]
    return max_epoch, max_value

def find_min_metric(values):
    min_epoch = min(values, key=values.get)
    min_value = values[min_epoch]
    return min_epoch, min_value

args = {
    'data': {
        'dataset': 'WuXi',
    },
    'model': {
        'type': "simple",
        'attr_dim': 8,
        'guidance_scale': 2,  # 是指导的缩放比例，用于在无条件噪声和条件噪声之间进行加权
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
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}

root_dir = Path(__name__).resolve().parents[0]
thresholds = list(range(1000, 11000, 1000))
results = {thresh: {} for thresh in thresholds}

for i in range(3, 10):
    args['data']['traj_length'] = i
    args['training']['batch_size'] = 5120
    args['training']['n_epochs'] = 200
    args["diffusion"]["num_diffusion_timesteps"] = 500
    if i > 4:
        args['training']['batch_size'] = 2560
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)

    exp_dir = root_dir / "Backups" / result_name / "results"

    for j in thresholds:
        metric_prefix = f'Test_mean_tc_threshold=TC@{j}'
        values = load_npy_files(exp_dir, metric_prefix)
        if values:
            max_epoch, max_value = find_max_metric(values)
            min_epoch, min_value = find_min_metric(values)

            results[j][i] = max_value

# Plotting the results
plt.figure(figsize=(12, 8))
for thresh, res in results.items():
    traj_lengths = list(res.keys())
    accuracies = list(res.values())
    plt.plot(traj_lengths, accuracies, marker='o', label=f'Threshold: {thresh} meters')

plt.xlabel('Trajectory Length')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Trajectory Lengths at Various Thresholds')
plt.legend()
plt.grid(True)

# Save the plot
output_path = root_dir / "accuracy_comparison.png"
plt.savefig(output_path)
print(f"Saved plot to {output_path}")

# Show the plot (optional)
# plt.show()
