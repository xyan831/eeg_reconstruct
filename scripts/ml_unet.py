import os
import numpy as np
from scipy.io import savemat
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data_util import load_mat, crop_timestep

from .model_util import normalize, torch_dataloader, train_model, predict
from .model_unet import UNet1D
from .visualize import save_pred_side

class ml_unet:
    def __init__(self, name, model, data_path, model_path, gen_path, visual_path):
        self.data_path
        self.model_path
        self.gen_path
        self.visual_path = path_list
        self.name = name
        self.model = model
        self.label = "data"
        self.config()

    def config(data_type="both", mask_type="random", epoch_num=10, ch_max=4):
        self.model_name = f"{self.model}_{mask_type}{ch_max}_e{epoch_num}_unet.pth"
        self.result_name = f"{self.name}{data_type}_e{epoch_num}_unet"
        self.norm_name = f"{self.name}{data_type}_{mask_type}{ch_max}_data_norm.mat"
        self.mask_name = f"{self.name}{data_type}_{mask_type}{ch_max}_data_mask.mat"
        self.epoch_num = epoch_num
        
        if data_type=="seiz":
            self.gen_name = [f"{self.name}_seizure_data.mat", "seizure_data"]
        elif data_type=="nseiz":
            self.gen_name = [f"{self.name}_non_seizure_data.mat", "non_seizure_data"]
        else:
            self.gen_name = [f"{self.name}_data.mat", self.label]

    def get_data(self):
        # load normal and masked data
        print("Loading dataset")
        data_orig = load_mat(os.path.join(self.data_path, self.norm_name), self.label)
        data_mask = load_mat(os.path.join(self.data_path, self.mask_name), self.label)
        
        # crop timesteps to fit unet (divisible by 2^(num_encoder_layers) = 2^4 = 16)
        print("Cropping data")
        crop_orig = crop_timestep(data_orig, 16)
        crop_mask = crop_timestep(data_mask, 16)
        #print(crop_norm.shape, crop_mask.shape)
        
        print("Data cropped")
        
        return crop_mask, crop_orig

    def visualize(self, y_pred, y_test, X_test, sample=0):
        # visualize and compare prediction (saved to pdf)
        print("Visualize results")
        ch_list = list(range(1,out_ch+1))
        #print(ch_list)
        result_name_norm = f"{self.result_name}_norm_s{sample}.pdf"
        result_name_unnorm = f"{self.result_name}_unnorm_s{sample}.pdf"
        # normalized
        save_pred_side(os.path.join(self.visual_path, result_name_norm), y_pred[0], y_test[0], X_test[0], ch_list, sample)
        print("normalized results recorded as", result_name_norm)
        # un-normalized
        save_pred_side(os.path.join(self.visual_path, result_name_unnorm), y_pred[1], yo_test[1], Xo_test[1], ch_list, sample)
        print("un_normalized results recorded as", result_name_unnorm)

    def train(self, sample=0):
        # prepare data
        print("Prepare training data")
        X_orig, y_orig = self.get_data()
        
        # Split the data into training and testing sets
        Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
        
        # normalize data
        scaler1 = StandardScaler()
        X_train = normalize(Xo_train, scaler1, "normal")
        X_test = normalize(Xo_test, scaler1, "normal")
        y_train = normalize(yo_train, scaler1, "normal")
        y_test = normalize(yo_test, scaler1, "normal")
        
        # convert to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        print(f"train samples = {X_train.shape[0]}\ntest samples = {X_test.shape[0]}")
        print(f"input_shape = {X_orig[0].shape}\noutput_shape = {y_orig[0].shape}")
        
        # prepare dataloader
        train_loader = torch_dataloader(X_train, y_train, batch_size=32, datatype="train")
        test_loader = torch_dataloader(X_test, y_test, batch_size=32, datatype="test")
        
        # in channels out channels
        in_ch = X_orig[0].shape[0]
        out_ch = y_orig[0].shape[0]
        print(f"in_channel = {in_ch}\nout_channel = {out_ch}")
        
        print("Prepare complete")
        
        # Initialize Model, Loss, Optimizer
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet1D(in_channels=in_ch, out_channels=out_ch).to(device)
        criterion = nn.MSELoss()  # For regression tasks
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train and save model
        print("Training model")
        train_model(device, model, train_loader, criterion, optimizer, epochs=self.epoch_num)
        torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name))
        print("Model train complete, saved as", self.model_name)
        
        # Get predictions for test set
        y_pred, bn_pred = predict(device, model, test_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        test_mse = np.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        
        # Inverse transform the predicted ECG to original scale
        y_pred2 = normalize(y_pred,scaler1,"reverse")
        
        # replace non blocked channels with original values
        y_pred3 = Xo_test.copy()
        for s_pred,s_input in zip(y_pred2, y_pred3):
            for ch in range(len(s_input)):
                if all(value==0 for value in s_input[ch]):
                    s_input[ch] = s_pred[ch]
        
        self.visualize([y_pred, y_pred3], [y_test, yo_test], [X_test, Xo_test], sample=sample)

    def test(self, sample=0):
        # prepare data
        print("Prepare training data")
        X_orig, y_orig = self.get_data()
        
        # normalize data
        scaler1 = StandardScaler()
        X_norm = normalize(X_orig, scaler1, "normal")
        y_norm = normalize(y_orig, scaler1, "normal")
        
        # convert to tensor
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        y_norm = torch.tensor(y_norm, dtype=torch.float32)
        
        # prepare dataloader full dataset no split
        full_loader = torch_dataloader(X_norm, y_norm, batch_size=32, datatype="test")
        
        # in channels out channels
        in_ch = X_orig[0].shape[0]
        out_ch = y_orig[0].shape[0]
        print(f"in_channel = {in_ch}\nout_channel = {out_ch}")
        
        print("Data Prepare complete")
        
        # Initialize Model, Loss, Optimizer
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet1D(in_channels=in_ch, out_channels=out_ch).to(device)
        criterion = nn.MSELoss()  # For regression tasks
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # load pretrained model
        state_dict = torch.load(os.path.join(self.model_path, self.model_name), weights_only=True)
        model.load_state_dict(state_dict)
        
        # Get generated data
        y_pred, bn_pred = predict(device, model, full_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        test_mse = np.mean((y_norm - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        
        # Inverse transform the predicted data to original scale
        y_pred2 = normalize(y_pred,scaler1,"reverse")
        
        # replace non blocked channels with original values
        y_pred3 = X_orig.copy()
        for s_pred,s_input in zip(y_pred2, y_pred3):
            for ch in range(len(s_input)):
                if all(value==0 for value in s_input[ch]):
                    s_input[ch] = s_pred[ch]
        
        # save un-normalized prediction as mat file
        savemat(os.path.join(self.gen_path, self.gen_name[0]), {self.gen_name[1]:y_pred3})
        print("generated data saved as", self.gen_name[0])
        
        self.visualize([y_pred, y_pred3], [y_norm, y_orig], [X_norm, X_orig], sample=sample)

