#!/bin/bash

# Script to run training with periodic validation (DDIM-style configuration)

# --- Configuration ---
DEVICE_ID=7
EXP_NAME_BASE="tky_ddim_exp" # Base name for the experiment
SAMPLING_TYPE="ddim" # Influences periodic validation (if DDIM is chosen) and experiment naming

# DDIM Specific Sampling Parameters for Periodic Validation
DDIM_STEPS=50
DDIM_ETA=0.2

# --- Training with Periodic Validation ---
echo "Starting DDIM-Mode Training with Periodic Validation..."
python main.py \
    --sampling_type ${SAMPLING_TYPE} \
    --exp_name ${EXP_NAME_BASE} \
    --device_id ${DEVICE_ID} \
    --ddim_steps ${DDIM_STEPS} \
    --ddim_eta ${DDIM_ETA}
    # Add other training parameters as needed

echo "Training finished."
