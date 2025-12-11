#!/bin/bash

home_path="path/to/home"
conda_path="$home_path/miniconda3"
conda_bin="$conda_path/bin"
conda_load="$conda_path/etc/profile.d/conda.sh"
env_path="$home_path/eeg_reconstruct/env"

# Add conda to PATH (bypass pyenv)
export PATH="$conda_bin:$PATH"

# Load Conda into the shell
source "$conda_load"

# Activate env
conda activate "$env_path"

# run in command line
#source activate_env.sh

# run in command line if file has problems
# sed -i 's/\r$//' activate_env.sh

