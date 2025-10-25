import os
import random
import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal as sp_signal

# load data from matlab file
def mat2numpy(filename, label):
    mat = loadmat(filename)
    #print(mat.keys())
    data = mat[label]
    data = np.array(data,dtype=float)
    return data

# crop timesteps to fit unet: divisible by 2^(num_encoder_layers)
def crop_timestep(data, div=16):
    crop_num = (data.shape[2]// div) * div
    #print(f"timesteps {data.shape[2]} crop to {crop_num}")
    crop_data = data[:, :, :crop_num]
    return crop_data

# filter data
def filter_data(data, std_max, std_min):
    # get input array and filter bad data
    for sample in data:
        for ch in range(len(sample)):
            ch_std = np.std(sample[ch])
            #print(ch_std)
            if ch_std<std_min or ch_std>std_max:
                sample[ch] = 0
    return data

def SP_50(signal, fs):
    """
    Remove 50 Hz power line interference from signals using a notch filter.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal(s) to process (can be 1D, 2D, or 3D array)
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    signal_processing : numpy.ndarray
        Processed signal with 50 Hz noise removed (same shape as input)
    """
    
    original_shape = signal.shape
    original_ndim = signal.ndim
    
    # Reshape to 2D for processing (samples x channels)
    if signal.ndim == 1:
        # 1D array: (n_samples,)
        signal_2d = signal.reshape(-1, 1).T  # shape: (1, n_samples)
    elif signal.ndim == 2:
        # 2D array: assume (n_channels, n_samples) or (n_samples, n_channels)
        if signal.shape[0] > signal.shape[1]:
            # If more rows than columns, assume (n_samples, n_channels)
            signal_2d = signal.T  # transpose to (n_channels, n_samples)
        else:
            # Assume already (n_channels, n_samples)
            signal_2d = signal
    elif signal.ndim == 3:
        # 3D array: (batch, channels, samples) or (channels, samples, batch)
        # For simplicity, flatten to 2D and remember original shape
        signal_2d = signal.reshape(signal.shape[0], -1)
    else:
        raise ValueError(f"Unsupported number of dimensions: {signal.ndim}")
    
    n_channels, n_samples = signal_2d.shape
    signal_processing = np.zeros((n_channels, n_samples))
    
    # Calculate normalized frequency parameters
    fs2 = fs / 2  # Nyquist frequency
    W0 = 50 / fs2  # Normalized cutoff frequency (50 Hz)
    BW = 0.1  # Bandwidth (3 dB)
    
    # Design IIR notch filter
    b, a = sp_signal.iirnotch(W0, W0/BW)
    
    # Apply filter to each channel
    for i in range(n_channels):
        x = signal_2d[i, :]
        y = sp_signal.lfilter(b, a, x)
        signal_processing[i, :] = y
    
    # Restore original shape
    if original_ndim == 1:
        result = signal_processing.flatten()
    elif original_ndim == 2:
        if original_shape[0] > original_shape[1]:
            # Was (n_samples, n_channels), transpose back
            result = signal_processing.T
        else:
            # Was (n_channels, n_samples)
            result = signal_processing
    elif original_ndim == 3:
        # Reshape back to 3D
        result = signal_processing.reshape(original_shape)
    
    return result

