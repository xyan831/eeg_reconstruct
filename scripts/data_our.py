import os
import re
import numpy as np
from scipy.io import loadmat, savemat
import mne

from .data_util import filter_data, SP_50

class data_our:
    def __init__(self, path_list, prefix):
        self.mat_path, self.our_path = path_list
        self.prefix = prefix

    def make_data(self):
        print("start make data")
        #seiz btw 120 sec + nseiz 240 sec
        
        full_list = os.listdir(self.our_path)
        txt_list = [x for x in full_list if ".txt" in x]
        edf_list = [x for x in full_list if ".edf" in x]
        seiz_seg = []
        nseiz_seg = []
        for summary in txt_list:
            summary_path = os.path.join(self.our_path, summary)
            seiz_info = self.get_seiz_info(summary_path)
            for i in range(len(seiz_info)):
                if seiz_info[i]:
                    seiz_path, nseiz_path = self.get_edf_path(edf_list, summary, i)
                    #seiz_seg, nseiz_seg = self.get_data_seg(seiz_path, nseiz_path, seiz_info, seiz_seg, nseiz_seg, seg_len=500)
        
        # Convert to NumPy arrays: (samples, channels, 500)
        #seiz_data = np.stack(seiz_seg)
        #nseiz_data = np.stack(nseiz_seg)
        
        # check standard deviation
        #std_max = 100000000000000000
        #std_min = 0.00000000000000001
        #seiz_data = filter_data(seiz_data, std_max, std_min)
        #nseiz_data = filter_data(nseiz_data, std_max, std_min)
        
        # 50hz filter
        #seiz_data = SP_50(seiz_data, 500)
        #nseiz_data = SP_50(nseiz_data, 500)
        
        #self.save_data(seiz_data, nseiz_data)

    def save_data(self, seiz_data, nseiz_data):
        print("saving data to mat")
        savemat(os.path.join(self.mat_path, f"{self.prefix}_seizure_data.mat"), {"seizure_data":seiz_data})
        savemat(os.path.join(self.mat_path, f"{self.prefix}_non_seizure_data.mat"), {"non_seizure_data":nseiz_data})
        print("save complete")

    def get_seiz_info(self, summary_path):
        print("getting seiz info")
        with open(summary_path, "r", encoding="gbk") as f:
            text = f.readlines()
            seiz_info = text[1:]
            seiz_info = [[x.strip() for x in string.split(" ") if x.strip()] for string in seiz_info]
        return seiz_info

    def get_edf_path(self, edf_list, summary, i):
        patient = re.split(r"[-.]", summary)
        seiz_name = f"{patient[0]}-{patient[1]}-{i+1}.edf"
        nseiz_name = f"{patient[0]}-{patient[1]}-{i+1}-240"
        s_list = [x for x in edf_list if seiz_name in x]
        n_list = [x for x in edf_list if nseiz_name in x]
        print(s_list)
        print(n_list)
        seiz_path = os.path.join(self.our_path, seiz_name)
        nseiz_path = os.path.join(self.our_path, nseiz_name)
        return seiz_path, nseiz_path

    def get_data_seg(self, seiz_path, nseiz_path, seiz_info, seiz_seg, nseiz_seg, seg_len=500):
        try:
            # Load EDF files
            seiz_raw = mne.io.read_raw_edf(seiz_path, preload=True)
            nseiz_raw = mne.io.read_raw_edf(nseiz_path, preload=True)
            
            # Get data as numpy arrays
            seiz_data = seiz_raw.get_data()  # shape: (channels, timesteps)
            nseiz_data = nseiz_raw.get_data()  # shape: (channels, timesteps)
            
            # Get sampling frequency
            sfreq = seiz_raw.info['sfreq']
            
            # Calculate step size (50% overlap)
            step_size = seg_len // 2
            
            # Process seizure data
            seiz_start_idx = 0
            while seiz_start_idx + seg_len <= seiz_data.shape[1]:
                segment = seiz_data[:, seiz_start_idx:seiz_start_idx + seg_len]
                seiz_seg.append(segment)
                seiz_start_idx += step_size
            
            # Process non-seizure data
            nseiz_start_idx = 0
            while nseiz_start_idx + seg_len <= nseiz_data.shape[1]:
                segment = nseiz_data[:, nseiz_start_idx:nseiz_start_idx + seg_len]
                nseiz_seg.append(segment)
                nseiz_start_idx += step_size
                
            print(f"Added {len(seiz_seg) - len(seiz_seg)} seizure segments and {len(nseiz_seg) - len(nseiz_seg)} non-seizure segments")
        
        except FileNotFoundError as e:
            print(f"Warning: File not found - {e}")
        except Exception as e:
            print(f"Error processing files: {e}")
        
        return seiz_seg, nseiz_seg








