# eeg_reconstruct Documentation

## Overview

### Directory Structure

```
������ /data
��   ������ CHB-MIT/
��   ������ NICU/
��   ������ Our/
������ /result
��   ������ data_gen/
��   ������ data_mat/
��   ������ data_train/
��   ������ model/
��   ������ visual/
������ /scripts
������ run.py
```

- ``/data``: .edf files. EEG data with seizure information.
    - ``CHB-MIT/``: from CHB MIT database
	- ``NICU``: from NICU database
    - ``Our/``: from patients
- ``/result``: generated files from scripts
    - ``data_gen/``: .mat files. reconstructed EEG data
    - ``data_mat/``: .mat files. 500 timestep samples from .edf rawdata
    - ``data_train/``: .mat files. masked data used for training/testing model
    - ``model/``: .pth files. trained model files
    - ``visual/``: .pdf files. visual comparison of reconstructed data from ground truth
- ``/scripts``: python scripts
    - ``data_chb.py``
    - ``data_mat.py``
    - ``data_our.py``
    - ``data_util.py``
    - ``ml_cnn.py``
    - ``ml_unet.py``
    - ``model_cnn.py``
    - ``model_unet.py``
    - ``model_util.py``
    - ``visualize.py``
- ``run.py``: edit and execute this file for whatever function you need

