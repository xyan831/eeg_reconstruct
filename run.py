import os

from scripts.test_functions import run_data, run_mat, run_recon, run_class

def prepare_rawdata(path_config):
    # our dataset
    run_data(path_config, "our", 26, "our01", r'eeg_0[1-9]\.edf|eeg_10\.edf')
    run_data(path_config, "our", 26, "our02", r'eeg_1[1-9]\.edf|eeg_20\.edf')
    run_data(path_config, "our", 26, "our03", r'eeg_2[1-9]\.edf|eeg_30\.edf')
    run_data(path_config, "our", 26, "our04", r'eeg_3[1-9]\.edf|eeg_40\.edf')
    run_data(path_config, "our", 26, "our05", r'eeg_4[1-9]\.edf|eeg_50\.edf')
    run_data(path_config, "our", 26, "our06", r'eeg_5[1-9]\.edf|eeg_60\.edf')
    run_data(path_config, "our", 26, "our07", r'eeg_6[1-9]\.edf|eeg_70\.edf')
    run_data(path_config, "our", 26, "our08", r'eeg_7[1-9]\.edf|eeg_80\.edf')
    run_data(path_config, "our", 26, "our09", r'eeg_8[1-9]\.edf|eeg_90\.edf')
    run_data(path_config, "our", 26, "our10", r'eeg_9[1-9]\.edf|eeg_100\.edf')
    # nicu dataset
    run_data(path_config, "nicu", 21, "nicu01", r'eeg_0[1-9]\.edf|eeg_10\.edf')
    run_data(path_config, "nicu", 21, "nicu02", r'eeg_1[1-9]\.edf|eeg_20\.edf')

def prepare_dataset(path_config, ch_max, block_ch, is_custom, name_ch):
    run_mat(path_config, "both", ch_max, block_ch, is_custom, f"our{name_ch}01", ["our01","our02","our03"])
    run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06"])
    run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"our{name_ch}02", ["our04","our05","our06"])
    run_mat(path_config, "both", ch_max, block_ch, is_custom, f"nicu{name_ch}01", ["nicu01"])
    run_mat(path_config, "seiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])
    run_mat(path_config, "nseiz", ch_max, block_ch, is_custom, f"nicu{name_ch}02", ["nicu02"])

def recon_train(path_config, dataset, train_data, recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample):
    run_recon(path_config,"train",train_data,recon_model,"unet",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)
    run_recon(path_config,"train",train_data,recon_model,"unet-ch",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)
    run_recon(path_config,"train",train_data,recon_model,"unet-tm",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)
    run_recon(path_config,"train",train_data,recon_model,"unet-fl",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)
    if dataset=="our":
        run_recon(path_config,"train",train_data,recon_model,"vae",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)
    elif dataset=="nicu":
        run_recon(path_config,"train",train_data,recon_model,"vae",epoch,0.0001,"both",isFFT,savebest,makevis,sample)
    run_recon(path_config,"train",train_data,recon_model,"diffusion",epoch,learn_rate,"both",isFFT,savebest,makevis,sample)

def recon_test(path_config, train_data, recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample):
    run_recon(path_config, "test", train_data, recon_model, "unet", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-ch", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-ch", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-tm", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-tm", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-fl", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "unet-fl", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "vae", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "vae", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "diffusion", epoch, learn_rate, "seiz", isFFT, savebest, makevis, sample)
    run_recon(path_config, "test", train_data, recon_model, "diffusion", epoch, learn_rate, "nseiz", isFFT, savebest, makevis, sample)

def class_train(path_config, name_ch, train_data, class_model, epoch, learn_rate, trainlist):
    run_class(path_config,"train",train_data,class_model,"cnn",epoch,learn_rate,True,False,trainlist)
    run_class(path_config,"train",train_data,class_model,"lstm",epoch,learn_rate,True,False,trainlist)
    run_class(path_config,"train",train_data,class_model,"transformer",epoch,learn_rate,True,False,trainlist)

def class_test1(path_config, train_data, class_model, epoch, learn_rate):
    norm_data = train_data+"_norm"
    mask_data = train_data+"_mask"
    run_class(path_config,"test",norm_data,class_model,"cnn",epoch,learn_rate,False,True,[norm_data])
    run_class(path_config,"test",mask_data,class_model,"cnn",epoch,learn_rate,False,True,[mask_data])
    run_class(path_config,"test",norm_data,class_model,"lstm",epoch,learn_rate,False,True,[norm_data])
    run_class(path_config,"test",mask_data,class_model,"lstm",epoch,learn_rate,False,True,[mask_data])
    run_class(path_config,"test",norm_data,class_model,"transformer",epoch,learn_rate,False,True,[norm_data])
    run_class(path_config,"test",mask_data,class_model,"transformer",epoch,learn_rate,False,True,[mask_data])

def class_test2(path_config, train_data, class_model, epoch, learn_rate):
    # test2
    unet_data = train_data+"_unet"
    unet_ch_data = train_data+"_unet-ch"
    unet_tm_data = train_data+"_unet-tm"
    unet_fl_data = train_data+"_unet-fl"
    vae_data = train_data+"_vae"
    diffusion_data = train_data+"_diffusion"
    run_class(path_config,"test",unet_data,class_model,"cnn",epoch,learn_rate,False,False,[unet_data])
    run_class(path_config,"test",unet_ch_data,class_model,"cnn",epoch,learn_rate,False,False,[unet_ch_data])
    run_class(path_config,"test",unet_tm_data,class_model,"cnn",epoch,learn_rate,False,False,[unet_tm_data])
    run_class(path_config,"test",unet_fl_data,class_model,"cnn",epoch,learn_rate,False,False,[unet_fl_data])
    run_class(path_config,"test",vae_data,class_model,"cnn",epoch,learn_rate,False,False,[vae_data])
    run_class(path_config,"test",diffusion_data,class_model,"cnn",epoch,learn_rate,False,False,[diffusion_data])
    
    run_class(path_config,"test",unet_data,class_model,"lstm",epoch,learn_rate,False,False,[unet_data])
    run_class(path_config,"test",unet_ch_data,class_model,"lstm",epoch,learn_rate,False,False,[unet_ch_data])
    run_class(path_config,"test",unet_tm_data,class_model,"lstm",epoch,learn_rate,False,False,[unet_tm_data])
    run_class(path_config,"test",unet_fl_data,class_model,"lstm",epoch,learn_rate,False,False,[unet_fl_data])
    run_class(path_config,"test",vae_data,class_model,"lstm",epoch,learn_rate,False,False,[vae_data])
    run_class(path_config,"test",diffusion_data,class_model,"lstm",epoch,learn_rate,False,False,[diffusion_data])
    
    run_class(path_config,"test",unet_data,class_model,"transformer",epoch,learn_rate,False,False,[unet_data])
    run_class(path_config,"test",unet_ch_data,class_model,"transformer",epoch,learn_rate,False,False,[unet_ch_data])
    run_class(path_config,"test",unet_tm_data,class_model,"transformer",epoch,learn_rate,False,False,[unet_tm_data])
    run_class(path_config,"test",unet_fl_data,class_model,"transformer",epoch,learn_rate,False,False,[unet_fl_data])
    run_class(path_config,"test",vae_data,class_model,"transformer",epoch,learn_rate,False,False,[vae_data])
    run_class(path_config,"test",diffusion_data,class_model,"transformer",epoch,learn_rate,False,False,[diffusion_data])

if __name__ == "__main__":
    # model settings
    epoch = 10
    learn_rate = 0.001
    isFFT = False
    savebest = False
    makevis = False
    sample = 0
    
    # mask_type
    is_custom = True
    ch_max = 4
    block_ch = [1, 2, 3, 4]
    #name_ch = f"{ch_max}r"
    name_ch = "1-4c"
    
    # dataset type: our, nicu
    dataset = "our"
    
    # file names
    our_data1 = f"our{name_ch}01"
    our_data2 = f"our{name_ch}02"
    nicu_data1 = f"nicu{name_ch}01"
    nicu_data2 = f"nicu{name_ch}02"
    our_recon_model = f"our{name_ch}01"
    nicu_recon_model = f"our{name_ch}01"
    our_class_model = f"nicu{name_ch}01"
    nicu_class_model = f"nicu{name_ch}01"
    
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
    
    #prepare_rawdata(path_config)
    
    #prepare_dataset(path_config, ch_max, block_ch, is_custom, name_ch)
    
    recon_train(path_config, "our", our_data1, our_recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample)
    recon_train(path_config, "nicu", nicu_data1, nicu_recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample)
    
    recon_test(path_config, our_data2, our_recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample):
    recon_test(path_config, nicu_data2, nicu_recon_model, epoch, learn_rate, isFFT, savebest, makevis, sample):
    
    epoch = 50
    class_train(path_config, name_ch, our_data1, our_recon_model, epoch, learn_rate, ["our01","our02","our03"])
    class_train(path_config, name_ch, nicu_data1, nicu_recon_model, epoch, learn_rate, ["nicu01"])
    
    class_test1(path_config, our_data2, our_recon_model, epoch, learn_rate)
    class_test1(path_config, nicu_data2, nicu_recon_model, epoch, learn_rate)
    
    class_test2(path_config, our_data2, our_recon_model, epoch, learn_rate)
    class_test2(path_config, nicu_data2, nicu_recon_model, epoch, learn_rate)





