import os
import numpy as np
from scipy.io import loadmat, savemat
import mne

from .data_util import filter_data, SP_50

class data_chb:
    def __init__(self, path_list, prefix, chb_pick):
        self.mat_path, self.chb_path = path_list
        self.prefix = prefix
        self.chb_pick = chb_pick

    def make_data(self):
        print("start make data")
        
        chb_lst = os.listdir(self.chb_path)
        chb_lst = [folder for folder in chb_lst if "chb" in folder]
        #print(chb_lst)
        
        # get seizure annotations
        seiz_info = {}
        for chb in self.chb_pick:
            summary_path = os.path.join(self.chb_path, chb, chb+"_summary.txt")
            seiz_info = self.get_seiz_info(summary_path, seiz_info)
        seiz_info = {fname: seizures for fname, seizures in seiz_info.items() if seizures}
        #print(seiz_info.keys())
        
        # get rawdata
        seiz_seg = []
        nseiz_seg = []
        for fname in seiz_info.keys():
            fpath = os.path.join(self.chb_path, fname.split("_")[0])
            if os.path.isfile(os.path.join(fpath, fname)):
                seiz_seg, nseiz_seg = self.get_data_seg(fpath, fname, seiz_info, seiz_seg, nseiz_seg, seg_len=500)
        
        # Convert to NumPy arrays: (samples, channels, 500)
        seiz_data = np.stack(seiz_seg)
        nseiz_data = np.stack(nseiz_seg)
        
        # check standard deviation
        #std_max = 100000000000000000
        #std_min = 0.00000000000000001
        #seiz_data = filter_data(seiz_data, std_max, std_min)
        #nseiz_data = filter_data(nseiz_data, std_max, std_min)
        
        # 50hz filter
        seiz_data = SP_50(seiz_data, 500)
        nseiz_data = SP_50(nseiz_data, 500)
        
        self.save_data(seiz_data, nseiz_data)

    def save_data(self, seiz_data, nseiz_data):
        print("saving data to mat")
        savemat(os.path.join(self.mat_path, f"{self.prefix}_seizure_data.mat"), {"seizure_data":seiz_data})
        savemat(os.path.join(self.mat_path, f"{self.prefix}_non_seizure_data.mat"), {"non_seizure_data":nseiz_data})
        print("save complete")

    def get_seiz_info(self, summary_path, seiz_info):
    # get seizure edf list from summary.txt
        current_file = None
        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("File Name:"):
                    current_file = line.split(":")[1].strip()
                    seiz_info[current_file] = []
                elif line.startswith("Seizure Start Time:"):
                    start = float(line.split(":")[1].strip().split()[0])
                elif line.startswith("Seizure End Time:"):
                    end = float(line.split(":")[1].strip().split()[0])
                    seiz_info[current_file].append((start, end))
        return seiz_info

    def get_data_seg(self, edf_path, edf_name, seiz_info, seiz_seg, nseiz_seg, seg_len=500):
        # create seizure and nonseizure segments from rawdata
        raw = mne.io.read_raw_edf(os.path.join(edf_path, edf_name), preload=True)
        data, times = raw.get_data(return_times=True)  # data: shape (channels, time)
        # make seiz and non-seiz dataset
        seiz_times = seiz_info.get(edf_name, [])  # from seizure_info
        sfreq = int(raw.info['sfreq'])  # sampling rate
        stride = seg_len // 2  # 50% overlap = 250 steps
        total_timesteps = data.shape[1]
        # Convert seizure times from seconds to sample indices
        seiz_samples = [(int(start * sfreq), int(end * sfreq)) for start, end in seiz_times]
        def is_seiz(start, end, seiz_intervals):
            for sz_start, sz_end in seiz_intervals:
                if end <= sz_start:
                    continue
                if start >= sz_end:
                    continue
                return True
            return False
        for start in range(0, total_timesteps - seg_len + 1, stride):
            end = start + seg_len
            chunk = data[:, start:end]
        
            if is_seiz(start, end, seiz_samples):
                seiz_seg.append(chunk)
            else:
                nseiz_seg.append(chunk)
        return seiz_seg, nseiz_seg

