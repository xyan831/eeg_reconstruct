import os

from scripts.data_mat import data_mat
from scripts.data_our import data_our

from scripts.ml_unet import ml_unet
from scripts.ml_cnn import ml_cnn

def run_unet(path_list, run_type, name, model, data_type):
    data_path, model_path, gen_path, visual_path = path_list
    epoch_num = 10
    ch_max=4
    
    unet1 = ml_unet(name, model, data_path, model_path, gen_path, visual_path)
    unet1.config(data_type=data_type, mask_type="random", epoch_num=epoch_num, ch_max=ch_max)
    
    if run_type=="train":    # train model
        unet1.train(sample=0)
    elif run_type=="test":    # test model
        unet1.test(sample=0)
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

def run_mat(path_list, name, data_type):
    # get folder paths
    mat_path, data_path = path_list
    ch_max = 4
    
    mat = data_mat(name, mat_path, data_path)
    mat.config(ch_max=ch_max, data_type=data_type, mask_type="random")
    mat.make_data()
    
def run_data(path_list, data_type, name):
    # get folder paths
    mat_path, our_path, nicu_path = path_list
    
    if data_type=="our2mat":
        our = data_our(name, mat_path, our_path)
        our.make_data()
    elif data_type=="nicu2mat":
        nicu = data_our(name, mat_path, nicu_path)
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
    path_list = [mat_path, our_path, nicu_path]
    run_data(path_list, "our2mat", prefix)
    #run_data(path_list, "nicu2mat", prefix)
    
    # unet datagen
    path_list = [data_path, model_path, gen_path, visual_path]
    #run_unet(path_list, "train", prefix, model, "both")
    #run_unet(path_list, "test", prefix, model, "nseiz")
    
    # classification
    path_list = [model_path, mat_path, gen_path]
    #run_cnn(path_list, "train", prefix, model)
    #run_cnn(path_list, "test", prefix, model)

