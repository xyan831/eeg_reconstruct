import os
import random
import numpy as np
from scipy.io import savemat

from .data_util import mat2numpy

class data_mat:
    def __init__(self, name, mat_path, data_path):
        self.mat_path = mat_path
        self.data_path = data_path
        self.label = "data"
        self.config()
        self.file_config(name, namelist=[name])

    def config(self, data_type="both", ch_max=4, block_ch=[1,2,3,4], mask_type="random"):
        self.data_type = data_type
        self.ch_max = ch_max
        self.block_ch = block_ch
        self.mask_type = mask_type

    def file_config(self, name, namelist=[]):
        self.norm_name = f"{name}_norm_{self.data_type}.mat"
        self.mask_name = f"{name}_mask_{self.data_type}.mat"
        seizlist = [f"{name}_seizure_data.mat" for name in namelist]
        nseizlist = [f"{name}_non_seizure_data.mat" for name in namelist]
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
        debug = []
        for filename in self.filelist:
            filedata = mat2numpy(os.path.join(self.mat_path, filename), self.label)
            if filedata.shape!=(0, 0):
                datalist.append(filedata)
        data_orig = np.concatenate(datalist)
        
        print(data_orig.shape)
        
        # get masked data
        if self.mask_type=="random":
            data_mask = self.random_mask(data_orig, data_orig.shape[1])
        elif self.mask_type=="custom":
            data_mask = self.custom_mask(data_orig, data_orig.shape[1])
        else:
            mask_type=="random"
            data_mask = self.random_mask(data_orig, data_orig.shape[1])
        
        print("saving data to mat")
        savemat(os.path.join(self.data_path, self.norm_name), {self.label:data_orig})
        savemat(os.path.join(self.data_path, self.mask_name), {self.label:data_mask})
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

