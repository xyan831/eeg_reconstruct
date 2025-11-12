import os

from scripts.data_mat import data_mat
from scripts.data_raw import data_raw

from scripts.reconstruct import reconstruct
from scripts.classification import classification

def run_our(path_list, name, isFFT=False):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    our = data_raw(name, mat_path, our_path, dataset="our")
    our.config(max_ch=26, timesteps=500, step_size=500, std_min=1e-10, std_max=1e10, is_FFT=isFFT)
    # files 01-10
    our.file_config(f"{name}01", file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')
    our.make_data()
    # files 11-20
    our.file_config(f"{name}02", file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')
    our.make_data()
    # files 21-30
    our.file_config(f"{name}03", file_pattern=r'eeg_2[1-9]\.edf|eeg_30\.edf')
    our.make_data()
    # files 31-40
    our.file_config(f"{name}04", file_pattern=r'eeg_3[1-9]\.edf|eeg_40\.edf')
    our.make_data()
    # files 41-50
    our.file_config(f"{name}05", file_pattern=r'eeg_4[1-9]\.edf|eeg_50\.edf')
    our.make_data()
    # files 51-60
    our.file_config(f"{name}06", file_pattern=r'eeg_5[1-9]\.edf|eeg_60\.edf')
    our.make_data()
    # files 61-70
    our.file_config(f"{name}07", file_pattern=r'eeg_6[1-9]\.edf|eeg_70\.edf')
    our.make_data()
    # files 71-80
    our.file_config(f"{name}08", file_pattern=r'eeg_7[1-9]\.edf|eeg_80\.edf')
    our.make_data()
    # files 81-90
    our.file_config(f"{name}09", file_pattern=r'eeg_8[1-9]\.edf|eeg_90\.edf')
    our.make_data()
    # files 91-99
    our.file_config(f"{name}10", file_pattern=r'eeg_9[1-9]\.edf')
    our.make_data()

def run_nicu(path_list, dataset, name, isFFT=False):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    nicu = data_raw(name, mat_path, nicu_path, dataset="nicu")
    nicu.config(max_ch=21, timesteps=500, step_size=500, std_min=1e-10, std_max=1e10, is_FFT=isFFT)
    # files 01-10
    nicu.file_config(f"{name}01", file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')
    nicu.make_data()
    # files 11-20
    nicu.file_config(f"{name}02", file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')
    nicu.make_data()

def run_mat(path_list, name, listname, ch_max, block_ch, data_type, mask_type):
    # get folder paths
    mat_path, data_path = path_list
    namelist = [f"{listname}01", f"{listname}02", f"{listname}03", f"{listname}04", f"{listname}05",
                f"{listname}06", f"{listname}07", f"{listname}08", f"{listname}09", f"{listname}10"]
    mat = data_mat(name, mat_path, data_path)
    mat.config(data_type=data_type, ch_max=ch_max, block_ch=block_ch, mask_type=mask_type)
    mat.file_config(name, namelist=namelist)
    mat.make_data()

def run_unet(path_list, run_type, name, model, epoch_num, data_type, sample):
    data_path, model_path, log_path, gen_path, visual_path = path_list
    
    unet1 = reconstruct(data_path, model_path, log_path, gen_path, visual_path, name, model)
    unet1.config(data_type=data_type, epoch_num=epoch_num, sample=sample)
    
    if run_type=="train":    # train model
        unet1.train()
    elif run_type=="test":    # test model
        unet1.test()
    else:
        print("invalid run type")

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
    cnn1 = classification(data_path, model_path, name)
    
    if run_type=="train":
        cnn1.train()
    elif run_type=="test":
        cnn1.test()
    else:
        print("invalid run type")

if __name__ == "__main__":
    # folder paths
    model_path = "result/model"
    visual_path = "result/visual"
    log_path = "result/log"
    gen_path = "result/data_gen"
    data_path = "result/data_train"
    mat_path = "result/data_mat"
    our_path = "../L1-Transformer/data/Our/eeg"
    nicu_path = "../L1-Transformer/data/NICU/eeg"
    
    ch_max = 4                 # for random mask
    block_ch = [1, 2, 3, 4]    # for custom mask
    epoch_num = 10
    name = "NM"
    isFFT = False
    
    # prepare rawdata
    path_list = [mat_path, our_path, nicu_path]
    #run_our(path_list, f"ourNM", isFFT=False)
    #run_our(path_list, f"ourFFT", isFFT=True)
    #run_nicu(path_list, f"nicuNM", isFFT=False)
    #run_nicu(path_list, f"nicuFFT", isFFT=True)

    # prepare dataset
    path_list = [mat_path, data_path]
    #run_mat(path_list, f"our{name}01", f"our{name}", ch_max, block_ch, "both", "random")
    #run_mat(path_list, f"our{name}02", f"our{name}", ch_max, block_ch, "seiz", "random")
    #run_mat(path_list, f"our{name}02", f"our{name}", ch_max, block_ch, "nseiz", "random")
    
    # reconstruction model training/testing (80/20)
    path_list = [data_path, model_path, log_path, gen_path, visual_path]
    run_unet(path_list, "train", f"our{name}01", f"our{name}1", epoch_num, "both", sample=0)
    #run_unet(path_list, "test", f"our{name}02", f"our{name}1", epoch_num, "seiz", sample=0)
    #run_unet(path_list, "test", f"our{name}02", f"our{name}1", epoch_num, "nseiz", sample=0)
    
    # classification
    path_list = [model_path, mat_path, gen_path]
    #run_cnn(path_list, "train", f"our{name}01", f"our{name}1")
    #run_cnn(path_list, "test", f"our{name}02", f"our{name}1")

