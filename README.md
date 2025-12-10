# eeg_reconstruct Documentation

## Overview

### Directory Structure

- ``/data``: .edf files. EEG data with seizure information.
	- ``NICU/``: from NICU database
    - ``Our/``: custom eeg data from patients
- ``/result``: generated files from scripts
    - ``data_gen/``: .mat files. reconstructed EEG data
    - ``data_mat/``: .mat files. 500 timestep samples from .edf rawdata
    - ``data_train/``: .mat files. masked data used for training/testing model
    - ``model/``: .pth files. trained model files
    - ``log/``: .txt files. record of training/validation results
    - ``visual/``: .pdf files. visual comparison of reconstructed data from ground truth
- ``/scripts``: python scripts
    - ``classification.py``: seizure classification script
    - ``data_mat.py``: create masked data for training/testing
    - ``data_raw.py``: create data .mat files from raw .edf files
    - ``data_util.py``: utility functions for data processing
    - ``reconstruct.py``: data reconstruction script
    - ``model_cnn.py``: CNN model for classification
    - ``model_cnn_lstm.py``: CNN_LSTM model for classification
    - ``model_transformer.py``: Transformer model for classification
    - ``model_unet.py``: U-Net model for reconstruction
    - ``model_unet_ch_att.py``: U-Net model with Channel Attention for reconstruction
    - ``model_unet_tm_att.py``: U-Net model with Temporal Attention for reconstruction
    - ``model_unet_ch_att.py``: U-Net model with Channel+Temporal Attention for reconstruction
    - ``model_vae.py``: VAE model for reconstruction
    - ``model_sdiffusion.py``: Standard Diffusion model for reconstruction
    - ``model_util.py``: utility functions for model training/testing
    - ``visualize.py``: visualize reconstruction results
    - ``test.py``: functions for how to use scripts
- ``run.py``: execute this file
- ``compare_graph.py``: create comparison graphs for model training
- ``movedata.py``: for copy data to diff folder
- ``activate_env.sh``: setup virtual env in chosen folder, edit path as needed

