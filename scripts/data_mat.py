import os
import random
import numpy as np
#import h5py
from scipy.io import savemat

from .data_util import mat2numpy

class data_mat:
    def __init__(self, path_config, param_config, file_config):
        self.mat_path = path_config.get("mat_path")
        self.data_path = path_config.get("data_path")
        
        self.data_type = param_config.get("data_type", "both")
        self.ch_max = param_config.get("ch_max", 4)
        self.block_ch = param_config.get("block_ch", [1,2,3,4])
        self.is_custom = param_config.get("is_custom", False)
        
        self.name_prefix = file_config.get("name_prefix", "our01")
        self.namelist = file_config.get("namelist", [self.name_prefix])
        seizlist = [f"{name}_seiz.mat" for name in self.namelist]
        nseizlist = [f"{name}_nseiz.mat" for name in self.namelist]
        
        self.label = "data"
        norm_name = f"{self.name_prefix}_norm_{self.data_type}.mat"
        mask_name = f"{self.name_prefix}_mask_{self.data_type}.mat"
        self.norm_file = os.path.join(self.data_path, norm_name)
        self.mask_file = os.path.join(self.data_path, mask_name)
        if self.data_type=="seiz":
            self.filelist = seizlist
        elif self.data_type=="nseiz":
            self.filelist = nseizlist
        else:
            self.filelist = seizlist + nseizlist

    def make_data(self):
        print("start make data")
        
        # load rawdata
        datalist = []
        for filename in self.filelist:
            filedata = mat2numpy(os.path.join(self.mat_path, filename), self.label)
            if filedata.shape!=(0, 0):
                datalist.append(filedata)
        data_orig = np.concatenate(datalist)
        
        print(data_orig.shape)
        
        # get masked data
        if self.is_custom:
            data_mask = self.custom_mask(data_orig, data_orig.shape[1])
        else:
            data_mask = self.random_mask(data_orig, data_orig.shape[1])
        
        print("saving data to mat")
        savemat(self.norm_file, {self.label:data_orig})
        savemat(self.mask_file, {self.label:data_mask})
        # if files too large try HDF5 format
        #with h5py.File(self.norm_file, 'w') as f:
        #    f.create_dataset(self.label, data=data_orig)
        #with h5py.File(self.mask_file, 'w') as f:
        #    f.create_dataset(self.label, data=data_mask)
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

    def custom_mask(self, data, channels):
        # get masked data by custom blocked channels
        mdata_list = []
        for sample in data:
            mdata = self.get_mask(sample, channels, self.block_ch)
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

