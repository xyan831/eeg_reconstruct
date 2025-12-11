import os

from scripts.test import run_our, run_nicu, run_mat1, run_mat2, run_recon, run_class

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

    # model settings
    epoch_num = 10
    isFFT = False
    savebest = False
    # mask_type
    is_custom = False
    ch_max = 4
    block_ch = [1, 2, 3, 4]
    name = f"{ch_max}r"
    # dataset type
    dataset = "our"
    # file names
    train_data1 = f"{dataset}{name}01"
    train_data2 = f"{dataset}{name}02"
    recon_model = f"{dataset}{name}01"
    class_model = f"{dataset}{name}01"
    sample = 0
    
    # prepare rawdata
    path_list = [mat_path, our_path, nicu_path]
    #run_our(path_list, "our")
    #run_nicu(path_list, "nicu")

    # prepare dataset
    path_list = [mat_path, data_path]
    #run_mat1(path_list, train_data1, dataset, ch_max, block_ch, "both", is_custom)
    #run_mat2(path_list, train_data2, dataset, ch_max, block_ch, "seiz", is_custom)
    #run_mat2(path_list, train_data2, dataset, ch_max, block_ch, "nseiz", is_custom)
    
    # reconstruction model training/testing (80/20)
    path_list = [data_path, model_path, log_path, gen_path, visual_path]
    # train
    #run_recon(path_list, "train", train_data1, recon_model, "unet", epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "train", train_data1, recon_model, "unet-ch", epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "train", train_data1, recon_model, "unet-tm", epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "train", train_data1, recon_model, "unet-fl", epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "train", train_data1, recon_model, "vae", epoch_num, "both", isFFT, savebest, sample)
    #run_recon(path_list, "train", train_data1, recon_model, "diffusion", epoch_num, "both", isFFT, savebest, sample)
    # test
    #run_recon(path_list, "test", train_data2, recon_model, "unet", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet", epoch_num, "nseiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-ch", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-ch", epoch_num, "nseiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-tm", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-tm", epoch_num, "nseiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-fl", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "unet-fl", epoch_num, "nseiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "vae", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "vae", epoch_num, "nseiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "diffusion", epoch_num, "seiz", isFFT, savebest, sample)
    #run_recon(path_list, "test", train_data2, recon_model, "diffusion", epoch_num, "nseiz", isFFT, savebest, sample)
    
    # classification
    path_list = [model_path, log_path, mat_path, gen_path, data_path]
    # train
    #run_class(path_list, "train", train_data1, dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "train", train_data1, dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "train", train_data1, dataset, class_model, "transformer", epoch_num)
    # test1
    #run_class(path_list, "test", train_data2+"_norm", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_mask", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_norm", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_mask", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_norm", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_mask", dataset, class_model, "transformer", epoch_num)
    
    # test2
    #run_class(path_list, "test", train_data2+"_unet", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-ch", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-tm", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-fl", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_vae", dataset, class_model, "cnn", epoch_num)
    #run_class(path_list, "test", train_data2+"_diffusion", dataset, class_model, "cnn", epoch_num)
    
    #run_class(path_list, "test", train_data2+"_unet", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-ch", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-tm", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-fl", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_vae", dataset, class_model, "lstm", epoch_num)
    #run_class(path_list, "test", train_data2+"_diffusion", dataset, class_model, "lstm", epoch_num)
    
    #run_class(path_list, "test", train_data2+"_unet", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-ch", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-tm", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_unet-fl", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_vae", dataset, class_model, "transformer", epoch_num)
    #run_class(path_list, "test", train_data2+"_diffusion", dataset, class_model, "transformer", epoch_num)




