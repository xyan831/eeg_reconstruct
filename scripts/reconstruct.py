import os
import numpy as np
from scipy.io import savemat
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data_util import mat2numpy, crop_timestep, convert_fft
from .model_util import normalize, torch_dataloader
from .model_unet import UNet1D as unet
from .model_unet_ch_att import UNet1D as unet_ch
from .model_unet_tm_att import UNet1D as unet_tm
from .model_unet_fl_att import UNet1D as unet_fl
from .model_vae import VAE1D, VAELoss
from .model_sdiffusion import StableDiffusionEEG, DiffusionLoss, EEGReconstructionMetrics
from .visualize import save_pred_side

class reconstruct:
    def __init__(self, path_config, model_config, param_config):
        self.data_path = path_config.get("data_path")
        self.model_path = path_config.get("model_path")
        self.log_path = path_config.get("log_path")
        self.gen_path = path_config.get("gen_path")
        self.visual_path = path_config.get("visual_path")
        
        self.model_type = model_config.get("model_type", "unet")
        self.epoch_num = model_config.get("epoch_num", 10)
        self.learning_rate = model_config.get("learning_rate", 0.001)
        
        self.name_prefix = param_config.get("name_prefix")
        self.model_prefix = param_config.get("model_prefix")
        self.data_type = param_config.get("data_type", "both")
        self.isFFT = param_config.get("isFFT", False)
        self.savebest = param_config.get("savebest", False)
        self.makevisual = param_config.get("makevisual", False)
        self.sample = param_config.get("sample", 0)
        
        #self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label = "data"
        self.file_config()

    def file_config(self):
        model_name = f"{self.model_prefix}_recon{self.epoch_num}e_{self.model_type}.pth"
        log_name = f"{self.model_prefix}_recon_{self.model_type}.txt"
        test_log_name = "test_log.txt"
        time_log_name = "time_log.txt"
        norm_name = f"{self.name_prefix}_norm_{self.data_type}.mat"
        mask_name = f"{self.name_prefix}_mask_{self.data_type}.mat"
        if self.data_type=="seiz":
            gen_name = f"{self.name_prefix}_{self.model_type}_seiz.mat"
        elif self.data_type=="nseiz":
            gen_name = f"{self.name_prefix}_{self.model_type}_nseiz.mat"
        else:
            gen_name = f"{self.name_prefix}_data.mat"
        vis_norm_name = f"{self.name_prefix}{self.data_type}_e{self.epoch_num}_{self.model_type}_norm_s{self.sample}.pdf"
        vis_unnorm_name = f"{self.name_prefix}{self.data_type}_e{self.epoch_num}_{self.model_type}_unnorm_s{self.sample}.pdf"
        self.model_file = os.path.join(self.model_path, model_name)
        self.log_file = os.path.join(self.log_path, log_name)
        self.test_log_file = os.path.join(self.log_path, test_log_name)
        self.time_log_file = os.path.join(self.log_path, time_log_name)
        self.norm_file = os.path.join(self.data_path, norm_name)
        self.mask_file = os.path.join(self.data_path, mask_name)
        self.gen_file = os.path.join(self.gen_path, gen_name)
        self.vis_norm_file = os.path.join(self.visual_path, vis_norm_name)
        self.vis_unnorm_file = os.path.join(self.visual_path, vis_unnorm_name)

    def get_data(self, file_path, cropsize=16):
        # load normal and masked data
        print("Loading dataset")
        data = mat2numpy(file_path, self.label)
        # crop timesteps to fit unet (divisible by 2^(num_encoder_layers) = 2^4 = 16)
        print("Cropping data")
        data = crop_timestep(data, div=cropsize)
        #print(crop_norm.shape, crop_mask.shape)
        print("Dataset ok")
        return data

    def get_model(self, in_ch, out_ch, load_pretrain=False):
        # Initialize Model, Loss, Optimizer
        if "unet" in self.model_type:
            if self.model_type=="unet-ch":
                model = unet_ch(in_channels=in_ch, out_channels=out_ch).to(self.device)
            elif self.model_type=="unet-tm":
                model = unet_tm(in_channels=in_ch, out_channels=out_ch).to(self.device)
            elif self.model_type=="unet-fl":
                model = unet_fl(in_channels=in_ch, out_channels=out_ch).to(self.device)
            else:
                print("using model unet")
                model = unet(in_channels=in_ch, out_channels=out_ch).to(self.device)
            criterion = nn.MSELoss()  # For regression tasks
        elif self.model_type=="vae":
            model = VAE1D(in_channels=in_ch, out_channels=out_ch, latent_dim=128, seq_len=496).to(self.device)
            criterion = VAELoss(reconstruction_loss='mse', beta=1.0)  # For regression tasks
        elif self.model_type=="diffusion":
            model = StableDiffusionEEG(in_channels=in_ch, out_channels=out_ch, timesteps=50).to(self.device)
            criterion = DiffusionLoss('l1')
        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        if load_pretrain==True:
            # load pretrained model
            print("load model:", self.model_file)
            state_dict = torch.load(self.model_file, weights_only=True)
            model.load_state_dict(state_dict)
        return model, criterion, optimizer

    def fill_block(self, block_data, fill_data):
        # replace non blocked channels with original values
        data = block_data.copy()
        for s_pred,s_input in zip(fill_data, data):
            for ch in range(len(s_input)):
                if all(value==0 for value in s_input[ch]):
                    s_input[ch] = s_pred[ch]
        return data

    def visualize(self, out_ch, y_pred, y_test, X_test):
        # visualize and compare prediction (saved to pdf)
        print("Visualize results")
        ch_list = list(range(1, out_ch+1))
        #print(ch_list)
        # normalized
        save_pred_side(self.vis_norm_file, y_pred[0], y_test[0], X_test[0], ch_list, self.sample)
        print("normalized results recorded as", self.result_name_norm)
        # un-normalized
        save_pred_side(self.vis_unnorm_file, y_pred[1], y_test[1], X_test[1], ch_list, self.sample)
        print("un_normalized results recorded as", self.result_name_unnorm)

    def train(self):
        # prepare data
        print("Prepare training data")
        X_orig = self.get_data(self.mask_file, cropsize=16)
        y_orig = self.get_data(self.norm_file, cropsize=16)
        
        # Apply FFT along the timesteps dimension (axis=2)
        if self.isFFT==True:
            X_orig = convert_fft(X_orig, sampling_rate=500, return_magnitude=True, inverse=False)
            y_orig = convert_fft(y_orig, sampling_rate=500, return_magnitude=True, inverse=False)
        
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
        model, criterion, optimizer = self.get_model(in_ch, out_ch, load_pretrain=False)
        
        # Train and save model
        print("Training model")
        self.train_model(model, criterion, optimizer, train_loader)
        
        if self.savebest==False:
            torch.save(model.state_dict(), self.model_file)
            print("Model train complete, saved at", self.model_file)
        print("Results logged at", self.log_file)
        
        # Get predictions for test set
        y_pred = self.predict(model, test_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        #test_mse = np.mean((y_test - y_pred) ** 2)
        test_mse = torch.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        
        # Inverse transform the predicted ECG to original scale
        y_pred2 = normalize(y_pred, scaler1, "reverse")
        
        # replace non blocked channels with original values
        y_pred3 = self.fill_block(Xo_test, y_pred2)
        
        if self.makevisual:
            self.visualize(out_ch, [y_pred, y_pred3], [y_test, yo_test], [X_test, Xo_test])

    def test(self):
        # prepare data
        print("Prepare testing data")
        X_orig = self.get_data(self.mask_file, cropsize=16)
        y_orig = self.get_data(self.norm_file, cropsize=16)
        
        # Apply FFT along the timesteps dimension (axis=2)
        if self.isFFT==True:
            X_orig = convert_fft(X_orig, sampling_rate=500, return_magnitude=True, inverse=False)
            y_orig = convert_fft(y_orig, sampling_rate=500, return_magnitude=True, inverse=False)
        
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
        
        # Initialize Model, Loss, Optimizer, Load pretrain
        model, criterion, optimizer = self.get_model(in_ch, out_ch, load_pretrain=True)
        
        # Get generated data
        y_pred = self.predict(model, full_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        #test_mse = np.mean((y_norm - y_pred) ** 2)
        test_mse = torch.mean((y_norm - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        with open(self.test_log_file, "a") as f:
            f.write(f"Recon {self.name_prefix}_{self.model_type} MSE:{test_mse:.4f}\n")
        
        # Inverse transform the predicted data to original scale
        y_pred2 = normalize(y_pred, scaler1, "reverse")
        
        # replace non blocked channels with original values
        y_pred3 = self.fill_block(X_orig, y_pred2)
        
        # save un-normalized prediction as mat file
        savemat(self.gen_file, {self.label:y_pred3})
        print("generated data saved at", self.gen_file)
        
        if self.makevisual:
            self.visualize(out_ch, [y_pred, y_pred3], [y_norm, y_orig], [X_norm, X_orig])

    def train_model(self, model, criterion, optimizer, dataloader):
        # create results log file header
        with open(self.log_file, "w") as f:
            if "unet" in self.model_type:
                f.write("epoch,total_loss\n")
            elif self.model_type=="vae":
                f.write("epoch,total_loss,recon_loss,kl_loss\n")
            elif self.model_type == "diffusion":
                f.write("epoch,total_loss,diffusion_loss\n")    
            else:
                raise ValueError(f"Invalid model type {self.model_type}")
        
        # Initialize metrics calculator for diffusion model
        if self.model_type == "diffusion":
            metrics_calculator = EEGReconstructionMetrics()
        
        # track time for process
        start_time = datetime.now()
        model.train()
        for epoch in range(self.epoch_num):
            best_loss = 100
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            running_diffusion_loss = 0.0
            running_perceptual_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                if "unet" in self.model_type:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    batch_recon = loss.item()
                    batch_kl = 0.0
                    batch_diffusion = 0.0
                    batch_perceptual = 0.0
                elif self.model_type=="vae":
                    recon_outputs, mu, logvar = model(X_batch)
                    loss, batch_recon, batch_kl = criterion(recon_outputs, y_batch, mu, logvar)
                    batch_recon = batch_recon.item()
                    batch_kl = batch_kl.item()
                    batch_diffusion = 0.0
                    batch_perceptual = 0.0
                elif self.model_type == "diffusion":
                    # For diffusion models, we predict noise
                    predicted_noise, target_noise, t = model(X_batch)
                    # Calculate diffusion loss (noise prediction)
                    diffusion_loss = criterion(predicted_noise, target_noise)
                    
                    # FIXED: Remove perceptual loss calculation to avoid device issues
                    # Just use diffusion loss for now
                    loss = diffusion_loss
                    
                    batch_diffusion = diffusion_loss.item()
                    batch_perceptual = 0.0  # Set to 0 for now
                    batch_recon = 0.0
                    batch_kl = 0.0
                else:
                    raise ValueError(f"Invalid model type {self.model_type}")

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_recon_loss += batch_recon
                running_kl_loss += batch_kl
                running_diffusion_loss += batch_diffusion
                running_perceptual_loss += batch_perceptual
            
            # calculate epoch average
            avg_loss = running_loss / len(dataloader)
            avg_recon_loss = running_recon_loss / len(dataloader)
            avg_kl_loss = running_kl_loss / len(dataloader)
            avg_diffusion_loss = running_diffusion_loss / len(dataloader)
            avg_perceptual_loss = running_perceptual_loss / len(dataloader)
            
            # save best model
            if (self.savebest==True) and (avg_loss<best_loss):
                best_loss = avg_loss
                torch.save(model.state_dict(), self.model_file)
                print("best model saved as", self.model_file)
            
            # Log results to file
            with open(self.log_file, "a") as f:
                if "unet" in self.model_type:
                    f.write(f"{epoch+1},{avg_loss:.4f}\n")
                elif self.model_type=="vae":
                    f.write(f"{epoch+1},{avg_loss:.4f},{avg_recon_loss:.4f},{avg_kl_loss:.4f}\n")
                elif self.model_type == "diffusion":
                    f.write(f"{epoch+1},{avg_loss:.6f},{avg_diffusion_loss:.6f}\n")
                else:
                    raise ValueError(f"Invalid model type {self.model_type}")
            # print results
            print(f"Epoch {epoch+1}/{self.epoch_num}, Loss: {avg_loss:.4f}")
        
        # track time for process
        end_time = datetime.now()
        time_taken = end_time - start_time
        # Log results to file
        with open(self.time_log_file, "a") as f:
            f.write(f"Recon {self.name_prefix}_{self.model_type} Train Time:{time_taken}\n")
        print("time taken:", time_taken)

    def predict(self, model, dataloader):
        model.eval()
        y_pred = []
        # track time for process
        start_time = datetime.now()
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                # Handle different model types
                if self.model_type=="vae":
                    # For VAE, extract only the reconstruction from the tuple
                    recon_outputs, _, _ = model(X_batch)
                    y_out = recon_outputs
                elif self.model_type == "diffusion":
                    # For diffusion models, we need to reconstruct the signal from predicted noise
                    # Since we're in eval mode, we'll use a simple reconstruction approach
                    # Sample timestep 0 (no noise) to get reconstruction
                    t = torch.zeros(X_batch.shape[0], device=self.device, dtype=torch.long)
                    # Add minimal noise and then denoise
                    noise = torch.randn_like(X_batch) * 0.01  # Small noise
                    x_noisy = X_batch + noise
                    predicted_noise = model.unet(x_noisy, t)
                    # Reconstruct by removing the predicted noise
                    y_out = x_noisy - predicted_noise
                else:
                    # For UNet and other models
                    y_out = model(X_batch)
                y_pred.append(y_out.cpu().numpy())
        # track time for process
        end_time = datetime.now()
        time_taken = end_time - start_time
        # Log results to file
        with open(self.time_log_file, "a") as f:
            f.write(f"Recon {self.name_prefix}_{self.model_type} Test Time:{time_taken}\n")
        print("time taken:", time_taken)
        return np.concatenate(y_pred, axis=0)

