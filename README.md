# ProDiff: Prototype-Guided Diffusion for Minimal Information Trajectory Imputation

This repository contains the official PyTorch implementation for the ICML 2025 paper **"ProDiff: Prototype-Guided Diffusion for Minimal Information Trajectory Imputation"** .
The paper introduces ProDiff, a novel framework where learned trajectory prototypes are used to guide and enhance the performance of diffusion models for complex trajectory-related tasks, specifically focusing on imputing trajectories with minimal available information. This codebase provides the tools to reproduce the experiments presented in the paper and to further explore the ProDiff methodology.

The following sections detail the structure of this codebase, setup instructions, and how to run training and inference.

## Overview

This project implements a deep learning model for trajectory generation and related tasks, leveraging a combination of Diffusion Models and a Trajectory Transformer. The system is designed to learn from trajectory data, generate new trajectories, and potentially perform tasks like trajectory completion or imputation under certain conditions. It incorporates a contrastive learning objective and uses prototype-based representations learned by the Transformer to guide the diffusion process.

The model architecture consists of:
1.  **Trajectory Transformer (`diffProModel/protoTrans.py`)**: Processes input trajectories (sequences of time, latitude, longitude) to extract meaningful features and generate representative prototypes. It employs self-attention mechanisms.
2.  **Diffusion Model (`diffProModel/Diffusion.py`)**: A UNet-based model that learns to denoise trajectories. It takes masked or corrupted trajectory data and learned prototypes as input to generate or refine trajectories through a reverse diffusion process. Supports both DDPM and DDIM sampling.

The training process involves a combined loss including diffusion loss, contrastive loss (to make learned trajectory embeddings discriminative), and a cross-entropy loss related to prototype assignment.

## Directory Structure

```
.
├── conf/
│   └── config.py           # Configuration file for all parameters
├── data/
│   ├── train.h5            # Placeholder for training data (HDF5 format expected)
│   └── test.h5             # Placeholder for test data (HDF5 format expected)
├── dataset/
│   └── data_util.py        # Data loading and preprocessing utilities (TrajectoryDataset)
├── diffProModel/
│   ├── Diffusion.py        # Core Diffusion model implementation (UNet, sampling)
│   ├── loss.py             # Custom loss functions (e.g., ContrastiveLoss)
│   └── protoTrans.py       # Trajectory Transformer model implementation
├── utils/
│   ├── logger.py           # Logging utilities
│   ├── metric.py           # Evaluation metrics for trajectories
│   └── utils.py            # Helper functions (KMeans, masking, etc.)
├── main.py                 # Main script to run training and experiments
├── train.py                # Contains the main training loop and logic
├── test.py                 # Handles model evaluation and testing
├── run_ddpm.sh             # Example script to run experiments with DDPM sampling
├── run_ddim.sh             # Example script to run experiments with DDIM sampling
└── README.md               # This file
```

## Setup

### Dependencies
The project relies on several Python libraries. Key dependencies include:
-   PyTorch (including `torch.distributed` for DDP)
-   NumPy
-   tqdm
-   h5py (implicitly, for loading `.h5` data files)

Ensure you have a Python environment with these packages installed. For GPU support, a compatible CUDA version and PyTorch build are required.



### Data Preparation
1.  Prepare your trajectory data in HDF5 format (`.h5` files).
2.  Each HDF5 file should be structured in a way that `dataset.data_util.TrajectoryDataset` can read it. Typically, this involves datasets for absolute time, latitude, and longitude for each trajectory.
3.  Place your training data (e.g., `train.h5`) and test data (e.g., `test.h5`) into the `data/` directory, or update the paths in `conf/config.py`.
4.  The `TrajectoryDataset` in `dataset/data_util.py` expects file paths and a trajectory length. Modify `get_data_paths` in `utils/utils.py` if your data storage differs.

## Configuration

All experimental parameters are managed through `conf/config.py`. This includes:
-   **Data settings**: Dataset paths, trajectory length, number of workers for DataLoader.
-   **Model settings**: Architecture details for both the Diffusion model (UNet) and the Transformer (embedding dimensions, number of layers/heads).
-   **Diffusion settings**: Beta schedule, number of timesteps.
-   **Training settings**: Batch size, number of epochs, learning rate, optimizer settings, distributed training flags, loss weights.
-   **Sampling settings**: Batch size for generation, DDIM steps, eta.

Modify `load_config()` in `conf/config.py` to suit your experiment. The configuration is loaded as a nested namespace object in `main.py`.

## Usage

The primary entry point for the project is `main.py`. You can also use the provided shell scripts (`run_ddpm.sh`, `run_ddim.sh`) as templates.

### Training

1.  **Configure**: Adjust parameters in `conf/config.py` as needed (e.g., paths, batch size, epochs).
2.  **Run Training**:
    *   **Single GPU / CPU**:
        ```bash
        python main.py --exp_name your_experiment_name
        ```
    *   **Distributed Data Parallel (DDP) Training**:
        To use DDP, set `training.dis_gpu` to `True` in `conf/config.py`. Then launch using `torchrun` or `torch.distributed.launch`. For example, for a 2-GPU setup:
        ```bash
        torchrun --nproc_per_node=2 main.py --exp_name your_ddp_experiment_name
        ```
        The `run_ddpm.sh` script provides an example of how to launch DDP training. It uses environment variables like `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` which are typically set by the launch utility.

    Training artifacts (model checkpoints, logs, results) will be saved in an experiment directory, usually structured as `experiments_json/<exp_name_with_timestamp>/`.

### Experiment Management
-   Experiments are organized into directories under `experiments_json/`.
-   Each experiment directory will contain:
    -   `config.json`: A copy of the configuration used for the experiment.
    -   `models/`: Saved model checkpoints (diffusion model and transformer model).
    -   `results/`: Saved outputs like losses, generated samples, or evaluation metrics.
    -   Log files.

### Sampling / Testing
-   The `test_model` function (in `test.py`, called from `train.py` during validation) handles the generation of samples and evaluation.
-   To run standalone testing or generation with a trained model:
    1.  Ensure you have trained model checkpoints (`.pt` files for diffusion and transformer, and `.npy` for prototypes if used).
    2.  Modify `main.py` or create a new script to load the checkpoints and call `test_model` with the appropriate configuration (e.g., point to the correct test data, set sampling parameters).
    3.  The `sampling` section in `conf/config.py` allows control over batch size for generation and specific sampling parameters like DDIM steps.

### Shell Scripts
-   `run_ddpm.sh`: An example script to launch training. It sets up environment variables for distributed training and calls `python main.py`. You might need to adapt it to your environment (e.g., GPU count, paths).
-   `run_ddim.sh`: Similar to `run_ddpm.sh`, potentially configured for DDIM sampling during validation/testing (though DDIM/DDPM choice for validation is often controlled within the config or `test_model` logic).

### Key Code Components

-   **`main.py`**:
    -   Parses arguments (like experiment name).
    -   Sets up the experiment directory and logging.
    -   Loads the configuration from `conf/config.py`.
    -   Calls `train_main` to start the training and validation process.
-   **`train.py` (`train_main` function)**:
    -   Sets up DDP if enabled.
    -   Initializes datasets, dataloaders, models (Diffusion and Transformer), optimizer, and scheduler.
    -   Contains the main training loop:
        -   Iterates through epochs and batches.
        -   Performs data scaling and masking.
        -   Passes data through the Transformer to get prototypes and features.
        -   Uses KMeans for clustering features (iteratively).
        -   Computes contrastive loss and cross-entropy loss for the Transformer.
        -   Computes diffusion loss using the Diffusion model, conditioned on masked data and matched prototypes.
        -   Performs backpropagation and optimizer step.
        -   Handles model saving and periodic validation by calling `test_model`.
-   **`test.py` (`test_model` function)**:
    -   Takes trained models, test dataloader, config, and other parameters.
    -   Sets models to evaluation mode.
    -   Iterates through the test data.
    -   Performs sampling using the Diffusion model (DDPM or DDIM based on config) conditioned on input data (e.g., masked trajectories) and prototypes from the Transformer.
    -   Calculates and logs evaluation metrics (from `utils/metric.py`).
    -   Saves generated samples/results.

This README provides a starting point. You may need to delve into the specific scripts and configuration options for more advanced usage or customization. 