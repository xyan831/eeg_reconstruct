import os

from scripts.test_functions import run_data, run_mat, run_recon, run_class

if __name__ == "__main__":
    # model settings
    epoch = 10
    learn_rate = 0.001
    isFFT = False
    savebest = False
    makevis = False
    sample = 0
    
    # mask_type
    is_custom = False
    ch_max = 4
    block_ch = [1, 2, 3, 4]
    name_ch = f"{ch_max}r"
    
    # dataset type: our, nicu
    dataset = "nicu"
    
    # file names
    train_data1 = f"{dataset}{name_ch}01"
    train_data2 = f"{dataset}{name_ch}02"
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
    #run_data(path_config, "our", 26, "our02", r'eeg_1[1-9]\.edf|eeg_20\.edf')
    #run_data(path_config, "our", 26, "our03", r'eeg_2[1-9]\.edf|eeg_30\.edf')
    #run_data(path_config, "our", 26, "our04", r'eeg_3[1-9]\.edf|eeg_40\.edf')
    #run_data(path_config, "our", 26, "our05", r'eeg_4[1-9]\.edf|eeg_50\.edf')
    #run_data(path_config, "our", 26, "our06", r'eeg_5[1-9]\.edf|eeg_60\.edf')
    #run_data(path_config, "our", 26, "our07", r'eeg_6[1-9]\.edf|eeg_70\.edf')
    #run_data(path_config, "our", 26, "our08", r'eeg_7[1-9]\.edf|eeg_80\.edf')
    #run_data(path_config, "our", 26, "our09", r'eeg_8[1-9]\.edf|eeg_90\.edf')
    #run_data(path_config, "our", 26, "our10", r'eeg_9[1-9]\.edf|eeg_100\.edf')
    
    #run_data(path_config, "nicu", 21, "nicu01", r'eeg_0[1-9]\.edf|eeg_10\.edf')
    #run_data(path_config, "nicu", 21, "nicu02", r'eeg_1[1-9]\.edf|eeg_20\.edf')

    # prepare dataset
    #run_mat(path_config, "both", ch_max, block_ch, is_custom, f"our{name_ch}01", ["our01","our02","our03"])
    #run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06"])
    #run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06"])
    #run_mat(path_config, "both", ch_max, block_ch, is_custom, f"nicu{name_ch}01", ["nicu01"])
    #run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])
    #run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])
    
    # reconstruction model training (80/20)
    #run_recon(path_config, "train", train_data1, recon_model, "unet", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "train", train_data1, recon_model, "unet-ch", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "train", train_data1, recon_model, "unet-tm", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "train", train_data1, recon_model, "unet-fl", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    
    # for our dataset
    #run_recon(path_config, "train", train_data1, recon_model, "vae", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    # for nicu dataset
    #run_recon(path_config, "train", train_data1, recon_model, "vae", epoch, 0.0001, "both", isFFT, savebest, makevis, sample)
    
    #run_recon(path_config, "train", train_data1, recon_model, "diffusion", epoch, learn_rate, "both", isFFT, savebest, makevis, sample)
    
    # reconstruction model testing
    #run_recon(path_config, "test", train_data2, recon_model, "unet", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-ch", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-ch", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-tm", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-tm", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-fl", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "unet-fl", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "vae", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "vae", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "diffusion", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    #run_recon(path_config, "test", train_data2, recon_model, "diffusion", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    
    # classification
    epoch = 50
    # train
    #run_class(path_config, "train", train_data1, dataset, class_model, "cnn", epoch)
    #run_class(path_config, "train", train_data1, dataset, class_model, "lstm", epoch)
    #run_class(path_config, "train", train_data1, dataset, class_model, "transformer", epoch)
    
    # test1
    #run_class(path_config, "test", train_data2+"_norm", dataset, class_model, "cnn", epoch)
    #run_class(path_config, "test", train_data2+"_mask", dataset, class_model, "cnn", epoch)
    #run_class(path_config, "test", train_data2+"_norm", dataset, class_model, "lstm", epoch)
    #run_class(path_config, "test", train_data2+"_mask", dataset, class_model, "lstm", epoch)
    #run_class(path_config, "test", train_data2+"_norm", dataset, class_model, "transformer", epoch)
    #run_class(path_config, "test", train_data2+"_mask", dataset, class_model, "transformer", epoch)
    
    # test2
    run_class(path_config, "test", train_data2+"_unet", dataset, class_model, "cnn", epoch)
    run_class(path_config, "test", train_data2+"_unet-ch", dataset, class_model, "cnn", epoch)
    run_class(path_config, "test", train_data2+"_unet-tm", dataset, class_model, "cnn", epoch)
    run_class(path_config, "test", train_data2+"_unet-fl", dataset, class_model, "cnn", epoch)
    run_class(path_config, "test", train_data2+"_vae", dataset, class_model, "cnn", epoch)
    run_class(path_config, "test", train_data2+"_diffusion", dataset, class_model, "cnn", epoch)
    
    run_class(path_config, "test", train_data2+"_unet", dataset, class_model, "lstm", epoch)
    run_class(path_config, "test", train_data2+"_unet-ch", dataset, class_model, "lstm", epoch)
    run_class(path_config, "test", train_data2+"_unet-tm", dataset, class_model, "lstm", epoch)
    run_class(path_config, "test", train_data2+"_unet-fl", dataset, class_model, "lstm", epoch)
    run_class(path_config, "test", train_data2+"_vae", dataset, class_model, "lstm", epoch)
    run_class(path_config, "test", train_data2+"_diffusion", dataset, class_model, "lstm", epoch)
    
    run_class(path_config, "test", train_data2+"_unet", dataset, class_model, "transformer", epoch)
    run_class(path_config, "test", train_data2+"_unet-ch", dataset, class_model, "transformer", epoch)
    run_class(path_config, "test", train_data2+"_unet-tm", dataset, class_model, "transformer", epoch)
    run_class(path_config, "test", train_data2+"_unet-fl", dataset, class_model, "transformer", epoch)
    run_class(path_config, "test", train_data2+"_vae", dataset, class_model, "transformer", epoch)
    run_class(path_config, "test", train_data2+"_diffusion", dataset, class_model, "transformer", epoch)




