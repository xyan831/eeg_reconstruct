import os

from scripts.data_mat import data_mat
from scripts.data_raw import data_raw

from scripts.ml_unet import ml_unet
from scripts.ml_cnn import ml_cnn

def run_cnn(path_list, run_type, name, model):
    # get folder paths
    model_path, mat_path, gen_path = path_list
    
    if run_type=="train":
        data_path = mat_path
    elif run_type=="test":
        data_path = gen_path
    else:
        print("invalid run type")
    
    model_path = os.path.join(model_path, f'{model}_best_cnn.pth')
    cnn1 = ml_cnn(data_path, model_path, name)
    
    if run_type=="train":
        cnn1.train()
    elif run_type=="test":
        cnn1.test()
    else:
        print("invalid run type")

def run_unet(path_list, run_type, name, model, epoch_num, data_type, sample):
    data_path, model_path, gen_path, visual_path = path_list
    
    unet1 = ml_unet(data_path, model_path, gen_path, visual_path, name, model)
    unet1.config(data_type=data_type, epoch_num=epoch_num, sample=sample)
    
    if run_type=="train":    # train model
        unet1.train()
    elif run_type=="test":    # test model
        unet1.test()
    else:
        print("invalid run type")

def run_mat(path_list, name, ch_max, block_ch, data_type, mask_type):
    # get folder paths
    mat_path, data_path = path_list
    
    mat = data_mat(name, mat_path, data_path)
    mat.config(data_type=data_type, ch_max=ch_max, block_ch=block_ch, mask_type=mask_type)
    mat.make_data()
    
def run_data(path_list, dataset, name):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    
    if dataset=="our":
        our = data_raw(name, mat_path, our_path, dataset="our")
        #our.file_limit(file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')  # files 01-10
        our.file_limit(file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')  # files 11-20
        our.make_data()
    elif dataset=="nicu":
        nicu = data_raw(name, mat_path, nicu_path, dataset="our")
        nicu.file_limit(file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')  # files 01-10
        #nicu.file_limit(file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')  # files 11-20
        nicu.make_data()
    else:
        print("invalid data type")

if __name__ == "__main__":
    # folder paths
    model_path = "result/model"
    visual_path = "result/visual"
    gen_path = "result/data_gen"
    data_path = "result/data_train"
    mat_path = "result/data_mat"
    our_path = "../L1-Transformer/data/Our/eeg"
    nicu_path = "../L1-Transformer/data/NICU/eeg"
    
    ch_max = 4                 # for random mask
    block_ch = [1, 2, 3, 4]    # for custom mask
    epoch_num = 10
    
    # prepare rawdata
    path_list = [mat_path, our_path, nicu_path]
    #run_data(path_list, "our", "ourNM2")
    #run_data(path_list, "nicu", "nicuNM1")

    # prepare dataset
    path_list = [mat_path, data_path]
    #run_mat(path_list, "ourNM1", ch_max, block_ch, "both", "random")
    #run_mat(path_list, "ourNM2", ch_max, block_ch, "seiz", "random")
    #run_mat(path_list, "ourNM2", ch_max, block_ch, "nseiz", "random")
    
    # reconstruction model training/testing
    path_list = [data_path, model_path, gen_path, visual_path]
    #run_unet(path_list, "train", "ourNM1", "ourNM1", epoch_num, "both", sample=0)
    run_unet(path_list, "test", "ourNM2", "ourNM1", epoch_num, "seiz", sample=0)
    run_unet(path_list, "test", "ourNM2", "ourNM1", epoch_num, "nseiz", sample=0)
    
    # classification
    path_list = [model_path, mat_path, gen_path]
    #run_cnn(path_list, "train", "ourNM2", "ourNM2")
    #run_cnn(path_list, "test", "ourNM2", "ourNM2")

