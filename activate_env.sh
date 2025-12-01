#!/bin/bash

# Load Conda into the shell
source /root/miniconda3/etc/profile.d/conda.sh

# Activate env (change path as needed)
conda activate /root/autodl-tmp/eeg_reconstruct/env

# run in command line
source activate_env.sh

# run in command line if file has problems
# sed -i 's/\r$//' activate_env.sh

