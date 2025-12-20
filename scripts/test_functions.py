import os

from .data_mat import data_mat
from .data_raw import data_raw

from .reconstruct import reconstruct
from .classification import classification

def run_data(path_config, dataset, max_ch, name_pre, file_pattern):
    param_config = {
        "dataset": dataset,
        "max_ch": max_ch,
        "timesteps": 500,
        "step_size": 500,
        "std_min": 1e-10,
        "std_max": 1e10
    }
    file_config = {
        "name_prefix": name_pre,
        "max_files": None,
        "file_pattern": file_pattern,
        "exclude_files": []
    }
    data = data_raw(path_config, param_config, file_config)
    data.make_data()

def run_mat(path_config, datatype, ch_max, block_ch, is_custom, name_pre, namelist):
    param_config = {
        "data_type": datatype,
        "ch_max": ch_max,
        "block_ch": block_ch,
        "is_custom": is_custom
    }
    file_config = {
        "name_prefix": name_pre,
        "namelist": namelist
    }
    mat = data_mat(path_config, param_config, file_config)
    mat.make_data()

def run_recon(path_config,run_type,name_pre,model_pre,modeltype,epoch,learn_rate,datatype, isFFT, savebest, makevis, sample):
    model_config = {
        "model_type": modeltype,
        "epoch_num": epoch,
        "learning_rate": learn_rate
    }
    param_config = {
        "name_prefix": name_pre,
        "model_prefix": model_pre,
        "data_type": datatype,
        "isFFT": isFFT,
        "savebest": savebest,
        "makevisual": makevis,
        "sample": sample
    }
    unet1 = reconstruct(path_config, model_config, param_config)
    if run_type=="train":    # train model
        unet1.train()
    elif run_type=="test":    # test model
        unet1.test()
    else:
        print("invalid run type")

def run_class(path_config, run_type, name, dataset, model, model_type, num_epochs):
    # get folder paths
    model_path = path_config.get("model_path")
    log_path = path_config.get("log_path")
    data_path = path_config.get("data_path")
    if run_type=="train":
        data_path = path_config.get("mat_path")
        if dataset=="our":
            namelist = ["our01","our02","our03","our04","our05"]
        elif dataset=="nicu":
            namelist = ["nicu01"]
        else:
            print("invalid datset")
    elif run_type=="test":
        data_path = path_config.get("gen_path")
        namelist = [name]
    else:
        print("invalid run type")
    cnn1 = classification(data_path, model_path, log_path, name, model, model_type=model_type, num_epochs=num_epochs)
    cnn1.file_config(name, namelist=namelist)
    if run_type=="train":
        cnn1.train()
    elif run_type=="test":
        cnn1.test()
    else:
        print("invalid run type")

if __name__ == "__main__":
    # model config
    epoch = 10
    learn_rate = 0.001
    isFFT = False
    makevis = False
    sample = 0
    
    # mask_type
    is_custom = False
    ch_max = 4
    block_ch = [1, 2, 3, 4]
    name_ch = f"{ch_max}r"
    
    # dataset type: our, nicu
    dataset = "our"
    
    # data_type: both, seiz, nseiz
    data_type = "both"
    
    # reconstruction models: unet, unet-ch, unet-tm, unet-fl, vae, diffusion
    recon_type = "unet"
    savebest = False
    
    # classification models: cnn, lstm, transformer
    class_type = "transformer"
    
    # file names
    train_data = f"{dataset}{name_ch}01"
    recon_model = f"{dataset}{name_ch}01"
    class_model = f"{dataset}{name_ch}01"
    
    # folder paths
    if dataset=="our":
        raw_path = "../L1-Transformer/data/Our/eeg"
    elif dataset=="nicu":
        raw_path = "../L1-Transformer/data/NICU/eeg"
    
    path_config = {
        "model_path": "result/model",
        "visual_path": "result/visual",
        "log_path": "result/log",
        "gen_path": "result/data_gen",
        "data_path": "result/data_train",
        "mat_path": "result/data_mat",
        "raw_path": raw_path
    }
    
    # prepare rawdata
    #run_data(path_config, "our", 26, "our01", r'eeg_0[1-9]\.edf|eeg_10\.edf')
    #run_data(path_config, "nicu", 21, "nicu01", r'eeg_0[1-9]\.edf|eeg_10\.edf')

    # prepare dataset
    #run_mat(path_config, "both", ch_max, block_ch, is_custom, f"our{name_ch}01", ["our01","our02","our03"])
    #run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06"])
    #run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06])
    #run_mat(path_config, "both", ch_max, block_ch, is_custom, f"nicu{name_ch}01", ["nicu01"])
    #run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])
    #run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])
    
    # reconstruction model training/testing (80/20)
    #run_recon(path_config, "train", train_data, recon_model, recon_type, epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data, recon_model, recon_type, epoch, learn_rate, data_type, isFFT, savebest, makevis, sample)
        
    # classification
    #run_class(path_config, "train", train_data, dataset, class_model, class_type, epoch)
    #run_class(path_config, "test", train_data, dataset, class_model, class_type, epoch)

