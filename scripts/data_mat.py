import os
import random
import numpy as np
from scipy.io import savemat

from .data_util import load_mat

class data_mat:
    def __init__(self, path_list, prefix, ch_max=4, data_type="both"):
        self.mat_path, self.data_path = path_list
        self.prefix = prefix
        self.data_type = data_type
        self.ch_max = ch_max

    def make_data(self):
        print("start make data")
        
        # load rawdata
        if self.data_type=="seiz":
            data_orig = load_mat(os.path.join(self.mat_path, f"{self.prefix}_seizure_data.mat"), "seizure_data")
        elif self.data_type=="nseiz":
            data_orig = load_mat(os.path.join(self.mat_path, f"{self.prefix}_non_seizure_data.mat"), "non_seizure_data")
        else:
            # load multiple rawdata and combine
            data_seiz = load_mat(os.path.join(self.mat_path, f"{self.prefix}_seizure_data.mat"), "seizure_data")
            data_nseiz = load_mat(os.path.join(self.mat_path, f"{self.prefix}_non_seizure_data.mat"), "non_seizure_data")
            data_orig = np.concatenate((data_seiz, data_nseiz), axis=0)
        
        print(data_orig.shape)
        
        # get masked data
        data_mask = self.random_mask(data_orig, data_orig.shape[1])
        
        self.save_data(data_orig, data_mask)

    def save_data(self, data_orig, data_mask):
        print("saving data to mat")
        savemat(os.path.join(self.data_path, f"{self.prefix}{self.data_type}_{self.ch_max}_data_norm.mat"), {"data":data_orig})
        savemat(os.path.join(self.data_path, f"{self.prefix}{self.data_type}_{self.ch_max}_data_mask.mat"), {"data":data_mask})
        print("save complete")

    def random_mask(self, data, channels):
        # get masked data by random blocked channels
        full_ch = list(range(1,channels+1))
        mdata_list = []
        for sample in data:
            block_num = random.randint(1, self.ch_max)
            block_ch = random.sample(full_ch, block_num)
            mdata = self.get_mask(sample, channels, block_ch)
            mdata_list.append(mdata)
        return np.array(mdata_list)

    def get_mask(self, data, full_ch, mask_ch):
        # get masked data by block channel
        mask = np.ones(data.shape)
        for i in mask_ch:
            if i > full_ch:
                print("error: contain invalid channel")
                return mask
            mask[i-1] = 0
        return data*mask

