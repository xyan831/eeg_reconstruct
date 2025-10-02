import os
import sys
import json
import pickle
import random

import mne
import re

import torch
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

import warnings
warnings.filterwarnings("ignore", message="Channels contain different highpass filters")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters")
warnings.filterwarnings("ignore", message="Number of records from the header does not match the file size")
warnings.filterwarnings("ignore", message="Channel names are not unique, found duplicates")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def reconstruct_vit_input_image(patch_tensor):
    """
    Reconstruct the 224x224x3 image that the ViT would "see" if the patch tensor
    represented RGB image patches.
    
    Args:
        patch_tensor (torch.Tensor): shape [196, 768]
    
    Returns:
        image (np.ndarray): shape [224, 224, 3], dtype uint8
    """
    assert patch_tensor.shape == (196, 768), "Expected patch tensor shape [196, 768]"

    patches = patch_tensor.clone().detach().cpu().numpy()
    patches = patches.reshape(196, 16, 16, 3)  # [num_patches, H, W, C]

    # Stitch patches into 224x224 image
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for idx, patch in enumerate(patches):
        row = idx // 14
        col = idx % 14
        image[row*16:(row+1)*16, col*16:(col+1)*16, :] = patch

    return image
    


def upscale_window_linear(window, target_len=768):
    """
    Linearly interpolate each EEG channel to target_len (default 768).

    Args:
        window (np.ndarray): shape (channels, time_steps)
        target_len (int): desired number of time steps after upscaling

    Returns:
        np.ndarray: shape (channels, target_len)
    """
    c, t = window.shape
    upscaled = np.zeros((c, target_len), dtype=np.float32)

    orig_x = np.linspace(0, 1, t)
    target_x = np.linspace(0, 1, target_len)

    for ch in range(c):
        upscaled[ch] = np.interp(target_x, orig_x, window[ch])

    return upscaled


def read_split_data(root: str, val_rate: float = 0.2, pre_seizure_offset_sec: int = 0):
    
    warnings.filterwarnings("ignore", message="Scaling factor is not defined")
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    seizure_class = ['no-seizure', 'pre-seizure']
    class_indices = {k: v for v, k in enumerate(seizure_class)}

    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    pre_seizure_segments = []
    non_seizure_files = []

    # For each patient, read metadata describing seizure events
    patients = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p)) and p != "patch_cache"]
    for patient in tqdm(patients):
        # Parse the summary file for seizure start/end times
        summary_path = os.path.join(root, patient, f"{patient}_summary.txt")
        assert os.path.exists(summary_path), f"dataset summary: {summary_path} does not exist."

        with open(summary_path, "r") as f:
            lines = f.readlines()

        file_info = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("File Name:"):
                fname = line.split(":")[1].strip()
                seizure_times = []

                while not lines[i].startswith("Number of Seizures"):
                    i += 1
                num_seizures = int(re.search(r"\d+", lines[i]).group())
                i += 1

                for _ in range(num_seizures):
                    start_sec = int(re.search(r":\s*(\d+)", lines[i]).group(1))
                    end_sec = int(re.search(r":\s*(\d+)", lines[i + 1]).group(1))
                    seizure_times.append((start_sec, end_sec))
                    i += 2

                file_info.append((fname, seizure_times))
            else:
                i += 1

        for fname, seizures in file_info:
            edf_path = os.path.join(root, patient, fname)
            if not os.path.exists(edf_path):
                continue

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                for ch in raw.info['chs']:
                    if ch.get('cal', None) is None or ch['cal'] == 0:
                        ch['cal'] = 0.000001
            except Exception:
                continue

            signal_rate = int(raw.info['sfreq'])
            duration_sec = 120
            offset = pre_seizure_offset_sec
            min_gap = duration_sec + offset
            # Define 1-second windows prior to each seizure (pre-seizure), as per L1-Attn preprocessing
            valid_seizures = []
            for j, (start, end) in enumerate(seizures):
                if start < min_gap:
                    continue
                if any(start - prev_end < min_gap for _, prev_end in seizures[:j]):
                    continue
                valid_seizures.append(start)
            
            for start_sec in valid_seizures:
                base_time = start_sec - offset - duration_sec
                for delta in range(0, 2 * (duration_sec - 2)):
                    # Sliding window (stride=0.5s) to create multiple 1s segments before seizure
                    t_start = base_time + delta * 0.5
                    t_end = t_start + 1.0
                    if t_end > base_time + duration_sec:
                        break
                    pre_seizure_segments.append(((edf_path, round(t_start, 3)), class_indices['pre-seizure']))

            # Files without seizures are used to sample negative (no-seizure) examples
            if not seizures:
                non_seizure_files.append((edf_path, raw.n_times // signal_rate))

    # Sample same number of non-seizure segments from beginning of available files (ignores first 30s to avoid start-up noise)
    non_seizure_segments = []
    duration_sec = 1
    needed = len(pre_seizure_segments)
    per_file_quota = max(1, needed // len(non_seizure_files))
    quota_remainder = needed - per_file_quota * len(non_seizure_files)

    for edf_path, file_len_sec in non_seizure_files:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        except Exception:
            continue

        segment_count = 0
        start_time = 30
        max_time = file_len_sec - duration_sec

        for t_start in np.arange(start_time, max_time, 0.5):
            if len(non_seizure_segments) >= needed:
                break
            non_seizure_segments.append(((edf_path, round(t_start, 3)), class_indices['no-seizure']))
            segment_count += 1
            if segment_count > per_file_quota:
                quota_remainder -= 1
            if segment_count >= (per_file_quota + (1 if quota_remainder > 0 else 0)):
                break

        if len(non_seizure_segments) >= needed:
            break

    all_data = pre_seizure_segments + non_seizure_segments
    train_data, train_labels, val_data, val_labels = [], [], [], []
    for record, label in all_data:
        if random.random() < val_rate:
            val_data.append(record)
            val_labels.append(label)
        else:
            train_data.append(record)
            train_labels.append(label)

    print(f"{len(pre_seizure_segments)} valid pre-seizure segments found.")
    print(f"Total: {len(train_data) + len(val_data)} 1-second windows generated.")
    print(f"{len(train_data)} for training.")
    print(f"{len(val_data)} for validation.")
    assert len(train_data) > 0, "number of training segments must be greater than 0."
    assert len(val_data) > 0, "number of validation segments must be greater than 0."

    return train_data, train_labels, val_data, val_labels



def eeg_to_vit_patches(edf_path, start_time_sec, vit_num_patches=196):
    """
    Load 1s of EEG from EDF and convert to ViT patch input.

    Args:
        edf_path (str): Path to .edf EEG file
        start_time_sec (int): Start time in seconds
        vit_num_patches (int): Total ViT patch count (e.g. 14x14 = 196)

    Returns:
        patches_tensor: torch.Tensor (vit_num_patches, 768)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    signal_rate = int(raw.info['sfreq'])
    picks = mne.pick_types(raw.info, eeg=True)

    # Select only supported EEG montages (21, 23, 28, 30 channels)
    if len(picks) != 23 and len(picks) != 21 and len(picks) != 30 and len(picks) != 28:
        raise ValueError("File has unsupported number of EEG channels")

    data, _ = raw[picks[:len(picks)], :]
    start_sample = int(start_time_sec * signal_rate)
    end_sample = start_sample + signal_rate * 1
    
    # reference for where channels should be mapped to
    channel_pos_string_map = {
        
        "NZ":  20.5,
        
        "FP1": 33,  "FPZ": 34.5,"FP2": 36,
        
        "F9":  44,  "AF7": 46,  "AF3": 47,  "AFZ": 48.5,
        "AF4": 50,  "AF8": 51,  "F10": 53,
        
        "F7":  59,  "F5":  60,  "F3":  61,  "F1":  62,  "FZ":  62.5,
        "F2":  63,  "F4":  64,  "F6":  65,  "F8":  66,
        
        "FT9": 71,  "FT7": 72,  "FC5": 73,  "FC3": 74,  "FC1": 75,  "FCZ": 76.5,
        "FC2": 78,  "FC4": 79,  "FC6": 80,  "FT8": 81,  "FT10":82,
        
        "A1":  98,  "T9":  99,  "T7":  100, "C5":  101, "C3":  102, "C1":  103, "CZ":  104.5,
        "C2":  106, "C4":  107, "C6":  108, "T8":  109, "T10": 110, "A2":  111,
        
        "TP9": 113, "TP7": 114, "CP5": 115, "CP3": 116, "CP1": 117, "CPZ": 118.5,
        "CP2": 120, "CP4": 121, "CP6": 122, "TP8": 123, "TP10":124,
        
        "P7":  129, "P5":  130, "P3":  131, "P1":  132, "PZ":  132.5,
        "P2":  133, "P4":  134, "P6":  135, "P8":  136,
        
        "P9":  142, "PO7": 144, "PO3": 145, "POZ": 146.5,
        "PO4": 148, "PO8": 149, "P10": 151,
        
        "O1":  159, "OZ": 160.5,"O2":  162,
        
        "IZ":  174.5
        
    }

    # Mappings for different datasets due to different channels
    channel_pos_map = {
        0: 46,   1: 72,   2: 114,  3: 144,  4: 47,
        5: 74,   6: 116,  7: 145,  8: 50,   9: 79,
        10: 121, 11: 148, 12: 51,  13: 81,  14: 123,
        15: 149, 16: 76.5,17: 118.5,18: 114,19: 85.5,
        20: 90.5,21: 95.5,  22: 123
        # channel 20 (ft9-ft10) is moved downwards (to location of Cz) to not overlap channel 16 (FCz)
    }
    
    if len(picks) == 21:
        channel_pos_map = {
            0: 33,    1: 36,    2: 60,    3: 65,    4: 102,
            5: 107,   6: 130,   7: 135,   8: 159,   9: 162,
            10: 58,   11: 67,   12: 99,   13: 110,  14: 142,
            15: 151,  16: 62.5, 17: 104.5,18:132.5, 19: 6.5,
            20: 188.5
        }
        
    elif len(picks) == 30:
        channel_pos_map = {
            0: 33,    1: 36,    2: 60,    3: 65,    4: 102,
            5: 107,   6: 130,   7: 135,   8: 159,   9: 162,
            10: 58,   11: 67,   12: 99,   13: 110,  14: 142,
            15: 151,  16: 84,   17: 97,   18: 62.5, 19: 104.5,
            20:132.5, 21: 23,   22: 18,   23: 6.5,  24: 182,
            25: 195
        }
        data = data[:26]
        
    elif len(picks) == 28:
        channel_pos_map = {
            0: 46, 1: 72, 2: 114, 3: 144,
            4: 47, 5: 74, 6: 116, 7: 145,
            8: 76.5, 9: 118.5,
            10: 50, 11: 79, 12: 121, 13: 148,
            14: 51, 15: 81, 16: 123, 17: 149,
            18: 114, 19: 85.5, 20: 90.5, 21: 95.5, 22: 123
        }
        exclude = {4, 9, 12, 17, 22}
        # Remove blank channels
        data = data[[i for i in range(data.shape[0]) if i not in exclude]]
        

    if end_sample > data.shape[1]:
        raise ValueError("Timestamp exceeds available data length")

    window = data[:, start_sample:end_sample]  # shape (channels, signal_rate)
    
    # Get 1-second window and interpolate to uniform length
    window = upscale_window_linear(window)  # shape (channels, 768), X ? R^{C × D}

    # Apply FFT to obtain frequency representation: X_fft ? R^{C × F}
    fft_vals = np.fft.fft(window, axis=1)  # shape (channels, 768)
    fft_magnitude = np.abs(fft_vals)       # shape (channels, 768)

    # Normalize each channel's frequency spectrum to [0, 256]
    window_data = fft_magnitude / np.max(fft_magnitude, axis=1, keepdims=True) * 256  # shape (channels, 768)

    # Initialize ViT patch array: 196 patches × 768 dim (frequency domain)
    patches = np.zeros((vit_num_patches, window_data.shape[1]), dtype=np.float32)

    used_patches = set()

    # Map each EEG channel to a spatial location (patch index) using electrode layout
    # Implements the Spatial Positional Embedding strategy described in Section 3.1
    for ch_idx, patch_idx in channel_pos_map.items():
        signal = window_data[ch_idx]
        # Handle fractional indices by splitting across adjacent patches to approximate central electrodes
        if isinstance(patch_idx, float) and patch_idx % 1 == 0.5:
            low, high = int(patch_idx), int(patch_idx) + 1
            if low < vit_num_patches:
                patches[low] += signal
                used_patches.add(low)
            if high < vit_num_patches:
                patches[high] += signal
                used_patches.add(high)
        else:
            idx = int(patch_idx)
            if idx < vit_num_patches:
                patches[idx] = signal
                used_patches.add(idx)

    # Zero out unused patches to keep spatial encoding consistent
    for i in range(vit_num_patches):
        if i not in used_patches:
            patches[i] = 0

    return torch.from_numpy(patches)

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=30)
    
    for step, data in enumerate(data_loader):
        patches, labels = data  # unpack batch
        
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        pred = model(patches)
        
        sample_num += labels.size(0)
        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()
        
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    all_preds = []
    all_labels = []

    seizureCount = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=30)

    for step, data in enumerate(data_loader):
        patches, labels = data
        patches = patches.to(device)
        labels = labels.to(device)

        pred = model(patches)
        loss = loss_function(pred, labels)

        accu_loss += loss.detach()
        sample_num += labels.size(0)

        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()
        seizureCount += (pred_classes == 1).sum()

        all_preds.append(pred_classes.cpu())
        all_labels.append(labels.cpu())
        
        

        data_loader.desc = "[testing epoch {}] loss: {:.3f}, acc: {:.3f}, pred_seizure: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            seizureCount.item() / sample_num
        )
    # calculate sen, spe, f1
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, sen, spe, f1

