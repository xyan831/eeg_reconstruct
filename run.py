import os

from scripts.data_mat import data_mat
from scripts.data_our import data_our
from scripts.data_chb import data_chb

from scripts.ml_unet import ml_unet
from scripts.ml_cnn import ml_cnn

def run_unet(path_list, run_type, prefix, model, data_type):
    # define epoch number and filenames
    epoch_num = 10
    
    model_name = f"{model}_e{epoch_num}_unet.pth"
    result_name = f"{prefix}{type}_e{epoch_num}_unet"
    
    if data_type=="seiz":
        gen_name = [f"{prefix}_seizure_data.mat","seizure_data"]
    elif data_type=="nseiz":
        gen_name = [f"{prefix}_non_seizure_data.mat","non_seizure_data"]
    else:
        data_type = "both"
        gen_name = [f"{prefix}_data.mat", "data"]
    
    name_list = [model_name, result_name, gen_name]
    
    #crop_orig, crop_mask = unet_dataload(data_path, prefix, data_type)
    unet1 = ml_unet(path_list, name_list, prefix, data_type)
    
    if run_type=="train":
        # train model
        unet1.train(epoch_num=epoch_num, sample=0)
    elif run_type=="test":
        # test model
        unet1.test(epoch_num=epoch_num, sample=0)
    else:
        print("invalid run type")

def run_cnn(path_list, run_type, prefix, model):
    # get folder paths
    model_path, mat_path, gen_path = path_list
    
    if run_type=="train":
        data_path = mat_path
    elif run_type=="test":
        data_path = gen_path
    else:
        print("invalid run type")
    
    model_path = os.path.join(model_path, f'{model}_best_cnn.pth')
        
    cnn1 = ml_cnn(data_path, model_path, prefix)
    
    if run_type=="train":
        cnn1.train()
    elif run_type=="test":
        cnn1.test()
    else:
        print("invalid run type")

def run_mat(path_list, prefix, data_type):
    # get folder paths
    mat_path, data_path = path_list
    ch_max = 4
    
    if data_type=="seiz":
        mat = data_mat(path_list, prefix, ch_max=ch_max, data_type="seiz")
    elif data_type=="nseiz":
        mat = data_mat(path_list, prefix, ch_max=ch_max, data_type="nseiz")
    else:
        mat = data_mat(path_list, prefix, ch_max=ch_max, data_type="both")
    mat.make_data()
    
def run_data(path_list, data_type, prefix):
    # get folder paths
    mat_path, our_path, chb_path = path_list
    
    if data_type=="our2mat":
        path_list = [mat_path, our_path]
        our = data_our(path_list, prefix)
        our.make_data()
        #ourdata_2_matraw(path_list, prefix)
    elif data_type=="chb2mat":
        path_list = [mat_path, chb_path]
        chb_pick = ["chb05"]
        chb = data_chb(path_list, prefix, chb_pick)
        chb.make_data()
        #chbmit_2_matraw(path_list, prefix, chb_pick)
    else:
        print("invalid data type")

if __name__ == "__main__":
    # folder paths
    model_path = "result/model"
    visual_path = "result/visual"
    gen_path = "result/data_gen"
    data_path = "result/data_train"
    mat_path = "result/data_mat"
    our_path = "data/ourdata"
    chb_path = "data/chb-mit"
    
    #prefix = "NM"
    prefix = "NM_our1"
    model = "NM2"
    
    # prepare matraw
    path_list = [mat_path, data_path]
    #run_mat(path_list, prefix, "seiz")
    #matraw_2_dataset(path_list, prefix, "seiz")
    #matraw_2_dataset(path_list, prefix, "nseiz")
    #matraw_2_dataset(path_list, prefix, "both")
    
    # prepare dataset
    path_list = [mat_path, our_path, chb_path]
    run_data(path_list, "our2mat", prefix)
    #run_data(path_list, "chb2mat", prefix)
    
    # unet datagen
    path_list = [data_path, model_path, gen_path, visual_path]
    #run_unet(path_list, "train", prefix, model, "both")
    #run_unet(path_list, "test", prefix, model, "nseiz")
    
    # classification
    path_list = [model_path, mat_path, gen_path]
    #run_cnn(path_list, "train", prefix, model)
    #run_cnn(path_list, "test", prefix, model)

