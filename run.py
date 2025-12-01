import os

from scripts.data_mat import data_mat
from scripts.data_raw import data_raw

from scripts.reconstruct import reconstruct
from scripts.classification import classification
from scripts.visualize import log_graph

def run_our(path_list, name):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    our = data_raw(name, mat_path, our_path, dataset="our")
    our.config(max_ch=26, timesteps=500, step_size=500, std_min=1e-10, std_max=1e10)
    # files 01-10
    our.file_config("our01", file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')
    our.make_data()
    # files 11-20
    our.file_config("our02", file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')
    our.make_data()
    # files 21-30
    our.file_config("our03", file_pattern=r'eeg_2[1-9]\.edf|eeg_30\.edf')
    our.make_data()
    # files 31-40
    our.file_config("our04", file_pattern=r'eeg_3[1-9]\.edf|eeg_40\.edf')
    our.make_data()
    # files 41-50
    our.file_config("our05", file_pattern=r'eeg_4[1-9]\.edf|eeg_50\.edf')
    our.make_data()
    # files 51-60
    our.file_config("our06", file_pattern=r'eeg_5[1-9]\.edf|eeg_60\.edf')
    our.make_data()
    # files 61-70
    our.file_config("our07", file_pattern=r'eeg_6[1-9]\.edf|eeg_70\.edf')
    our.make_data()
    # files 71-80
    our.file_config("our08", file_pattern=r'eeg_7[1-9]\.edf|eeg_80\.edf')
    our.make_data()
    # files 81-90
    our.file_config("our09", file_pattern=r'eeg_8[1-9]\.edf|eeg_90\.edf')
    our.make_data()
    # files 91-99
    our.file_config("our10", file_pattern=r'eeg_9[1-9]\.edf')
    our.make_data()

def run_nicu(path_list, name):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    nicu = data_raw(name, mat_path, nicu_path, dataset="nicu")
    nicu.config(max_ch=21, timesteps=500, step_size=500, std_min=1e-10, std_max=1e10)
    # files 01-10
    nicu.file_config("nicu01", file_pattern=r'eeg_0[1-9]\.edf|eeg_10\.edf')
    nicu.make_data()
    # files 11-20
    nicu.file_config("nicu02", file_pattern=r'eeg_1[1-9]\.edf|eeg_20\.edf')
    nicu.make_data()

def run_mat(path_list, name, listname, ch_max, block_ch, data_type, is_custom):
    # get folder paths
    mat_path, data_path = path_list
    # for our
    #namelist = [f"{listname}01", f"{listname}02", f"{listname}03", f"{listname}04", f"{listname}05"]
    namelist = [f"{listname}06", f"{listname}07", f"{listname}08", f"{listname}09", f"{listname}10"]
    # for nicu
    #namelist = [f"{listname}01"]
    #namelist = [f"{listname}02"]
    mat = data_mat(name, mat_path, data_path)
    mat.config(data_type=data_type, ch_max=ch_max, block_ch=block_ch, is_custom=is_custom)
    mat.file_config(name, namelist=namelist)
    mat.make_data()

def run_recon(path_list, run_type, name, model, modeltype, epoch_num, data_type, isFFT, savebest, sample):
    data_path, model_path, log_path, gen_path, visual_path = path_list
    
    unet1 = reconstruct(data_path, model_path, log_path, gen_path, visual_path)
    unet1.config(name, model, modeltype, 
                data_type=data_type, epoch_num=epoch_num, isFFT=isFFT, savebest=savebest, sample=sample)
    
    if run_type=="train":    # train model
        unet1.train()
    elif run_type=="test":    # test model
        unet1.test()
    else:
        print("invalid run type")

def run_class(path_list, run_type, name, dataset, model, model_type, num_epochs):
    # get folder paths
    model_path, log_path, mat_path, gen_path, data_path = path_list
    if run_type=="train":
        data_path = mat_path
        if dataset=="our":
            namelist = [f"{dataset}01", f"{dataset}02", f"{dataset}03", f"{dataset}04", f"{dataset}05"]
        elif dataset=="nicu":
            namelist = [f"{dataset}01"]
        else:
            print("invalid datset")
    elif run_type=="test":
        data_path = gen_path
        namelist = [name]
    else:
        print("invalid run type")
    cnn1 = classification(data_path, model_path, log_path, name, model,
                            model_type=model_type, num_epochs=num_epochs)
    cnn1.file_config(name, namelist=namelist)
    if run_type=="train":
        cnn1.train()
    elif run_type=="test":
        cnn1.test()
    else:
        print("invalid run type")

def run_log(visual_path, log_path, single_log=True, log_num=0):
    log_files = [f for f in os.listdir(log_path) if f.endswith(".txt")]
    print(log_files)
    if single_log:
        log_name = log_files[log_num]
        log_graph(visual_path, log_path, log_name)
    else:
        for log_name in log_files:
            log_graph(visual_path, log_path, log_name)

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

    epoch_num = 10
    isFFT = True
    # mask_type
    is_custom = False
    ch_max = 4
    block_ch = [1, 2, 3, 4]
    name = f"{ch_max}r"
    # dataset type
    dataset = "nicu"
    # reconstruction models: unet, unet-ch, unet-tm, unet-fl, vae, diffusion
    recon_type = "unet"
    savebest = False
    # classification models: cnn, lstm, transformer
    class_type = "transformer"
    # file names
    train_data = f"{dataset}{name}02"
    recon_model = f"{dataset}{name}01"
    class_model = f"{dataset}{name}01"
    sample = 0
    
    # prepare rawdata
    path_list = [mat_path, our_path, nicu_path]
    #run_our(path_list, "our")
    #run_nicu(path_list, "nicu")

    # prepare dataset
    path_list = [mat_path, data_path]
    #run_mat(path_list, train_data, dataset, ch_max, block_ch, "both", is_custom)
    #run_mat(path_list, train_data, dataset, ch_max, block_ch, "seiz", is_custom)
    #run_mat(path_list, train_data, dataset, ch_max, block_ch, "nseiz", is_custom)
    
    # reconstruction model training/testing (80/20)
    path_list = [data_path, model_path, log_path, gen_path, visual_path]
    #run_recon(path_list, "train", train_data, recon_model, recon_type, epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data, recon_model, recon_type, epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data, recon_model, recon_type, epoch_num, "nseiz", isFFT, savebest, sample)
    
    # classification
    path_list = [model_path, log_path, mat_path, gen_path, data_path]
    run_class(path_list, "train", train_data, dataset, class_model, class_type, epoch_num)
    #run_class(path_list, "test", train_data, dataset, class_model, class_type, epoch_num)

    # graph training log
    #run_log(visual_path, log_path, single_log=True, log_num=0)
    #run_log(visual_path, log_path, single_log=False, log_num=0)

