import os
import re
import numpy as np
from scipy.io import loadmat, savemat
import mne

from .data_util import filter_data, convert_fft, SP_50

class data_raw:
    def __init__(self, name, mat_path, raw_path, dataset="our"):
        self.mat_path = mat_path
        self.raw_path = raw_path
        self.label = "data"
        self.seiz_name = f"{name}_seizure_data.mat"
        self.nseiz_name = f"{name}_non_seizure_data.mat"
        
        self.dataset = dataset
        valid_dataset = ["our", "nicu"]
        if dataset not in valid_dataset:
            raise ValueError(f"Invalid dataset type: '{dataset}'. Must be one of: {valid_dataset}")
        
        self.config()
        self.file_limit(max_files=5)

    def config(self, max_ch=None, timesteps=500, step_size=500, std_min=1e-10, std_max=1e10, is_FFT=False):
        self.max_ch = max_ch
        self.timesteps = timesteps
        self.step_size = step_size
        self.std_min = std_min
        self.std_max = std_max
        self.is_FFT = is_FFT

    def file_limit(self, max_files=None, file_pattern=None, exclude_files=None):
        self.max_files = max_files
        self.file_pattern = file_pattern
        self.exclude_files = exclude_files or []

    def make_data(self):
        print("start make data")
        summary_path = os.path.join(self.raw_path, "eeg_summary.txt")
        seiz_info = self.get_summary(summary_path)
        
        all_seiz_data, all_nseiz_data = self.get_data(seiz_info)
        
        # Convert lists to numpy arrays
        seiz_data = np.vstack(all_seiz_data) if all_seiz_data else np.array([])
        nseiz_data = np.vstack(all_nseiz_data) if all_nseiz_data else np.array([])
        
        print(f"Total seizure segments: {len(seiz_data)}")
        print(f"Total non-seizure segments: {len(nseiz_data)}")
        
        # 50hz filter
        seiz_data = SP_50(seiz_data, sf=self.timesteps)
        nseiz_data = SP_50(nseiz_data, sf=self.timesteps)
        
        # std filter
        #seiz_data = filter_data(seiz_data, self.std_max, self.std_min)
        #nseiz_data = filter_data(nseiz_data, self.std_max, self.std_min)
        
        # convert to FFT
        if self.is_FFT:
            seiz_data = convert_fft(seiz_data, sampling_rate=self.timesteps)
            nseiz_data = convert_fft(nseiz_data, sampling_rate=self.timesteps)
        
        # save data
        print("saving data to mat")
        savemat(os.path.join(self.mat_path, self.seiz_name), {self.label:seiz_data})
        savemat(os.path.join(self.mat_path, self.nseiz_name), {self.label:nseiz_data})
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
                if self.dataset=="our":
                    start_match = re.search(rf'Seizure Start Time: (\d+) seconds', entry)
                    end_match = re.search(rf'Seizure End Time: (\d+) seconds', entry)
                elif self.dataset=="nicu":
                    start_match = re.search(rf'Seizure_{i+1} Start Time: (\d+) seconds', entry)
                    end_match = re.search(rf'Seizure_{i+1} End Time: (\d+) seconds', entry)
                if start_match and end_match:
                    seizures.append({
                        'start': int(start_match.group(1)),
                        'end': int(end_match.group(1))
                    })
            seiz_info[filename] = seizures
        return seiz_info

    def _get_edf_file_list(self):
        """Get filtered list of EDF files to process"""
        all_files = []
        for filename in os.listdir(self.raw_path):
            if filename.endswith('.edf') and not filename.startswith('._'):
                # Check if file should be excluded
                if filename in self.exclude_files:
                    continue
                
                # Check file pattern if specified
                if self.file_pattern and not re.match(self.file_pattern, filename):
                    continue
                
                all_files.append(filename)
        
        # Sort files for consistent ordering
        all_files.sort()
        
        # Limit number of files if specified
        if self.max_files:
            all_files = all_files[:self.max_files]
        
        print(f"Processing {len(all_files)} EDF files: {all_files}")
        return all_files

    def get_data(self, seiz_info):
        all_seiz_data = []
        all_nseiz_data = []
        
        # Get list of EDF files with filtering
        edf_files = self._get_edf_file_list()
        
        # Process each EDF file
        for filename in edf_files:
            file_path = os.path.join(self.raw_path, filename)
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
            
            print(f"  - Seizure segments: {len(seiz_segments)}, Non-seizure segments: {len(nseiz_segments)}")
        
        return all_seiz_data, all_nseiz_data

    def get_edf(self, file_path, max_ch=None):
        # load edf file return data and sampling frequency
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            # Limit channels if specified
            if max_ch is not None and max_ch < data.shape[0]:
                data = data[:max_ch, :]
            
            print(f"    Loaded: {data.shape[0]} channels, {data.shape[1]} samples, sfreq: {sfreq}Hz")
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
            print(f"    Seizure interval: {seizure['start']}s-{seizure['end']}s -> samples {start_sample}-{end_sample}")
        
        # Create segments with overlap
        start_idx = 0
        segment_count = 0
        seiz_count = 0
        
        while start_idx + segment_samples <= total_samples:
            end_idx = start_idx + segment_samples
            
            # Check if this segment overlaps with any seizure
            is_seizure = False
            for seiz_start, seiz_end in seizure_samples:
                # Check for overlap - fixed logic
                if start_idx < seiz_end and end_idx > seiz_start:
                    is_seizure = True
                    seiz_count += 1
                    break
            
            segment = data[:, start_idx:end_idx]
            segment_reshaped = segment[np.newaxis, :, :]
            
            if is_seizure:
                seiz_segments.append(segment_reshaped)
            else:
                nseiz_segments.append(segment_reshaped)
            
            start_idx += step_samples
            segment_count += 1
        
        print(f"    Total segments: {segment_count}, Seizure segments found: {seiz_count}")
        return seiz_segments, nseiz_segments

