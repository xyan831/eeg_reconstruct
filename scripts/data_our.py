import os
import re
import numpy as np
from scipy.io import loadmat, savemat
import mne

from .data_util import filter_data, SP_50

class data_our:
    def __init__(self, name, mat_path, our_path):
        self.mat_path = mat_path
        self.our_path = our_path
        self.seiz_name = f"{name}_seizure_data.mat"
        self.nseiz_name = f"{name}_non_seizure_data.mat"
        self.seiz_label = "seizure_data"
        self.nseiz_label = "non_seizure_data"
        self.config()

    def config(self, timesteps=500, step_size=250, std_min=1e-10, std_max=1e10, max_ch=None):
        self.timesteps = timesteps
        self.step_size = step_size
        self.std_min = std_min
        self.std_max = std_max
        self.max_ch = max_ch

    def make_data(self):
        print("start make data")
        summary_path = os.path.join(self.our_path, "eeg_summary.txt")
        seiz_info = self.get_summary(summary_path)
        
        all_seiz_data, all_nseiz_data = self.get_data(seiz_info)
        
        # Convert lists to numpy arrays
        seiz_data = np.vstack(all_seiz_data) if all_seiz_data else np.array([])
        nseiz_data = np.vstack(all_nseiz_data) if all_nseiz_data else np.array([])
        
        # check standard deviation
        #seiz_data = filter_data(seiz_data, self.std_max, self.std_min)
        #nseiz_data = filter_data(nseiz_data, self.std_max, self.std_min)
        
        # 50hz filter
        seiz_data = SP_50(seiz_data, 500)
        nseiz_data = SP_50(nseiz_data, 500)
        
        # save data
        print("saving data to mat")
        savemat(os.path.join(self.mat_path, self.seiz_name), {self.seiz_label:seiz_data})
        savemat(os.path.join(self.mat_path, self.nseiz_name), {self.nseiz_label:nseiz_data})
        print("save complete")

    def get_summary(self, summary_path):
        seiz_info = {}
        # read summary
        with open(summary_path, 'r') as f:
            summary = f.read()
        # split by edf files
        filesum = re.split(r'File Name: ', summary)[1:]
        for entry in filesum:
            lines = entry.strip().split('\n')
            filename = lines[0].strip()
            # get seizure info
            num_seizures_match = re.search(r'Number of Seizures in File: (\d+)', lines[1])
            num_seizures = int(num_seizures_match.group(1)) if num_seizures_match else 0
            seizures = []
            for i in range(num_seizures):
                start_match = re.search(rf'Seizure_{i+1} Start Time: (\d+) seconds', entry)
                end_match = re.search(rf'Seizure_{i+1} End Time: (\d+) seconds', entry)
                if start_match and end_match:
                    seizures.append({
                        'start': int(start_match.group(1)),
                        'end': int(end_match.group(1))
                    })
            seiz_info[file_name] = seizures
        return seiz_info

    def get_data(self, seiz_info):
        all_seiz_data = []
        all_nseiz_data = []
        
        # Process each EDF file
        for filename in os.listdir(self.our_path):
            if filename.endswith('.edf') and not filename.startswith('._'):
                file_path = os.path.join(self.our_path, filename)
                print(f"Processing {filename}...")
                
                # Load EDF file
                data, sfreq = self.get_edf(file_path, self.max_ch)
                if data is None:
                    continue
                
                # Get seizure intervals for this file
                file_seiz_info = seiz_info.get(filename, [])
                
                # Create segments
                seiz_segments, nseiz_segments = self.get_seg(data, sfreq, file_seiz_info)
                
                if seiz_segments:
                    all_seiz_data.extend(seiz_segments)
                if nseiz_segments:
                    all_nseiz_data.extend(nseiz_segments)
                
                return all_seiz_data, all_nseiz_data

    def get_edf(self, file_path, max_ch=None):
        # load edf file return data and sampling frequency
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            # Limit channels if specified
            if max_ch is not None and max_ch < data.shape[0]:
                data = data[:max_channels, :]
            return data, sfreq
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def get_seg(self, data, sfreq, seizure_intervals):
        # Create seizure and non-seizure segments from EEG data
        seiz_segments = []
        nseiz_segments = []
        
        total_samples = data.shape[1]
        segment_samples = self.timesteps
        step_samples = self.step_size
        
        # Convert seizure intervals from seconds to sample indices
        seizure_samples = []
        for seizure in seizure_intervals:
            start_sample = int(seizure['start'] * sfreq)
            end_sample = int(seizure['end'] * sfreq)
            seizure_samples.append((start_sample, end_sample))
        
        # Create segments with overlap
        start_idx = 0
        while start_idx + segment_samples <= total_samples:
            end_idx = start_idx + segment_samples
            
            # Check if this segment overlaps with any seizure
            is_seizure = False
            for seiz_start, seiz_end in seizure_samples:
                # Check for overlap
                if not (end_idx <= seiz_start or start_idx >= seiz_end):
                    is_seizure = True
                    break
            
            segment = data[:, start_idx:end_idx]
            
            # Reshape to (1, channels, timesteps)
            segment_reshaped = segment[np.newaxis, :, :]
            
            if is_seizure:
                seiz_segments.append(segment_reshaped)
            else:
                nseiz_segments.append(segment_reshaped)
            
            start_idx += step_samples
        
        return seiz_segments, nseiz_segments

