#!/bin/bash

# Script to run training with periodic validation (DDPM-style configuration)

# --- Configuration ---
DEVICE_ID=7
EXP_NAME_BASE="tky_ddpm_exp" # Base name for the experiment
SAMPLING_TYPE="ddpm" # Influences periodic validation and experiment naming

# --- Training with Periodic Validation ---
echo "Starting DDPM Training with Periodic Validation..."
python main.py \
    --sampling_type ${SAMPLING_TYPE} \
    --exp_name ${EXP_NAME_BASE} \
    --device_id ${DEVICE_ID}
    # Add other training parameters as needed, e.g.:
    # --config_module_path conf.config_custom \
    # --seed 42

echo "Training finished."