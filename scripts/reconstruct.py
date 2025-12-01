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
    def __init__(self, data_path, model_path, log_path, gen_path, visual_path):
        self.data_path = data_path
        self.model_path = model_path
        self.log_path = log_path
        self.gen_path = gen_path
        self.visual_path = visual_path
        self.label = "data"

    def config(self, name, model, model_type="unet", data_type="both", epoch_num=10, isFFT=False, savebest=False, sample=0):
        self.name = name
        self.model = model
        self.model_type = model_type
        self.epoch_num = epoch_num
        self.isFFT = isFFT
        self.savebest = savebest
        self.sample = sample
        self.model_name = f"{model}_recon{epoch_num}e_{model_type}.pth"
        self.log_name = f"{model}_recon_{self.model_type}.txt"
        self.norm_name = f"{name}_norm_{data_type}.mat"
        self.mask_name = f"{name}_mask_{data_type}.mat"
        if data_type=="seiz":
            self.gen_name = f"{name}_{model_type}_seizure_data.mat"
        elif data_type=="nseiz":
            self.gen_name = f"{name}_{model_type}_non_seizure_data.mat"
        else:
            self.gen_name = f"{name}_data.mat"
        self.result_name_norm = f"{name}{data_type}_e{epoch_num}_{model_type}_norm_s{sample}.pdf"
        self.result_name_unnorm = f"{name}{data_type}_e{epoch_num}_{model_type}_unnorm_s{sample}.pdf"

    def get_logname(self):
        now = datetime.now()
        return f"{self.log_name}_{now.strftime('%Y%m%d_%H%M%S')}.txt"

    def get_data(self, filename, cropsize=16):
        # load normal and masked data
        print("Loading dataset")
        data = mat2numpy(os.path.join(self.data_path, filename), self.label)
        # crop timesteps to fit unet (divisible by 2^(num_encoder_layers) = 2^4 = 16)
        print("Cropping data")
        data = crop_timestep(data, div=cropsize)
        #print(crop_norm.shape, crop_mask.shape)
        print("Dataset ok")
        return data

    def get_model(self, in_ch, out_ch, load_pretrain=False):
        # Initialize Model, Loss, Optimizer
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "unet" in self.model_type:
            if self.model_type=="unet-ch":
                model = unet_ch(in_channels=in_ch, out_channels=out_ch).to(device)
            elif self.model_type=="unet-tm":
                model = unet_tm(in_channels=in_ch, out_channels=out_ch).to(device)
            elif self.model_type=="unet-fl":
                model = unet_fl(in_channels=in_ch, out_channels=out_ch).to(device)
            else:
                print("using model unet")
                model = unet(in_channels=in_ch, out_channels=out_ch).to(device)
            criterion = nn.MSELoss()  # For regression tasks
        elif self.model_type=="vae":
            model = VAE1D(in_channels=in_ch, out_channels=out_ch, latent_dim=128, seq_len=496).to(device)
            criterion = VAELoss(reconstruction_loss='mse', beta=1.0)  # For regression tasks
        elif self.model_type=="diffusion":
            model = StableDiffusionEEG(in_channels=in_ch, out_channels=out_ch, timesteps=50).to(device)
            criterion = DiffusionLoss('l1')
        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if load_pretrain==True:
            # load pretrained model
            print("load model:", self.model_name)
            state_dict = torch.load(os.path.join(self.model_path, self.model_name), weights_only=True)
            model.load_state_dict(state_dict)
        return device, model, criterion, optimizer

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
        save_pred_side(os.path.join(self.visual_path, self.result_name_norm), y_pred[0], y_test[0], X_test[0], ch_list, self.sample)
        print("normalized results recorded as", self.result_name_norm)
        # un-normalized
        save_pred_side(os.path.join(self.visual_path, self.result_name_unnorm), y_pred[1], y_test[1], X_test[1], ch_list, self.sample)
        print("un_normalized results recorded as", self.result_name_unnorm)

    def train(self):
        # prepare data
        print("Prepare training data")
        X_orig = self.get_data(self.mask_name, cropsize=16)
        y_orig = self.get_data(self.norm_name, cropsize=16)
        
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
        device, model, criterion, optimizer = self.get_model(in_ch, out_ch, load_pretrain=False)
        
        # Train and save model
        print("Training model")
        self.train_model(device, model, criterion, optimizer, train_loader)
        
        if self.savebest==False:
            torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name))
            print("Model train complete, saved as", self.model_name)
        print("Results logged in", self.log_name)
        
        # Get predictions for test set
        y_pred = self.predict(device, model, test_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        #test_mse = np.mean((y_test - y_pred) ** 2)
        test_mse = torch.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        
        # Inverse transform the predicted ECG to original scale
        y_pred2 = normalize(y_pred, scaler1, "reverse")
        
        # Inverse fft predicted ECG
        #y_pred2 = np.fft.ifft(y_pred2, axis=2)
        
        # replace non blocked channels with original values
        y_pred3 = self.fill_block(Xo_test, y_pred2)
        
        self.visualize(out_ch, [y_pred, y_pred3], [y_test, yo_test], [X_test, Xo_test])

    def test(self):
        # prepare data
        print("Prepare testing data")
        X_orig = self.get_data(self.mask_name, cropsize=16)
        y_orig = self.get_data(self.norm_name, cropsize=16)
        
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
        device, model, criterion, optimizer = self.get_model(in_ch, out_ch, load_pretrain=True)
        
        # Get generated data
        y_pred = self.predict(device, model, full_loader)
        print(y_pred.shape)
        
        # Evaluation (MSE)
        #test_mse = np.mean((y_norm - y_pred) ** 2)
        test_mse = torch.mean((y_norm - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        
        # Inverse transform the predicted data to original scale
        y_pred2 = normalize(y_pred, scaler1, "reverse")
        
        # Inverse fft predicted ECG
        #y_pred2 = np.fft.ifft(y_pred2, axis=2)
        
        # replace non blocked channels with original values
        y_pred3 = self.fill_block(X_orig, y_pred2)
        
        # save un-normalized prediction as mat file
        savemat(os.path.join(self.gen_path, self.gen_name), {self.label:y_pred3})
        print("generated data saved as", self.gen_name)
        
        self.visualize(out_ch, [y_pred, y_pred3], [y_norm, y_orig], [X_norm, X_orig])

    def run(self, filename):
        # prepare data
        print("Loading dataset")
        data_orig = self.get_data(filename, cropsize=16)
        
        # Apply FFT along the timesteps dimension (axis=2)
        if self.isFFT==True:
            data_orig = convert_fft(data_orig, sampling_rate=500, return_magnitude=True, inverse=False)
        
        # normalize data
        scaler1 = StandardScaler()
        data_norm = normalize(data_orig, scaler1, "normal")
        
        # convert to tensor
        data_norm = torch.tensor(data_norm, dtype=torch.float32)
        
        # prepare dataloader full dataset no split
        data_loader = DataLoader(TensorDataset(data_norm), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        
        # in channels out channels
        in_ch = data_norm[0].shape[0]
        out_ch = data_norm[0].shape[0]
        print(f"in_channel = {in_ch}\nout_channel = {out_ch}")
        
        print("Data Prepare complete")
        
        # Initialize Model, Loss, Optimizer, Load pretrain
        device, model, criterion, optimizer = self.get_model(in_ch, out_ch, load_pretrain=True)
        
        # Get generated data
        pred_data = predict(device, model, data_loader)
        print(pred_data.shape)
        
        # Inverse transform the predicted data to original scale
        pred_data2 = normalize(pred_data, scaler1, "reverse")
        
        # Inverse fft predicted ECG
        #pred_data2 = np.fft.ifft(pred_data2, axis=2)
        
        # replace non blocked channels with original values
        pred_data3 = self.fill_block(data_orig, pred_data2)
        
        # save un-normalized prediction as mat file
        savemat(os.path.join(self.gen_path, self.gen_name), {self.label:pred_data3})
        print("generated data saved as", self.gen_name)

    def train_model(self, device, model, criterion, optimizer, dataloader):
        # create results file for log
        #log_name = self.get_logname()
        log_file = os.path.join(self.log_path, self.log_name)
        # Create results file with header
        with open(log_file, "w") as f:
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
        
        model.train()
        for epoch in range(self.epoch_num):
            best_loss = 100
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            running_diffusion_loss = 0.0
            running_perceptual_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
                torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name))
                print("best model saved as", self.model_name)
            
            # Log results to file
            with open(log_file, "a") as f:
                if "unet" in self.model_type:
                    f.write("{},{:.4f}\n".format(epoch+1, avg_loss))
                elif self.model_type=="vae":
                    f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, avg_loss, avg_recon_loss, avg_kl_loss))
                elif self.model_type == "diffusion":
                    f.write("{},{:.6f},{:.6f}\n".format(epoch+1, avg_loss, avg_diffusion_loss))
                else:
                    raise ValueError(f"Invalid model type {self.model_type}")
            # print results
            print(f"Epoch {epoch+1}/{self.epoch_num}, Loss: {avg_loss:.4f}")

    def predict(self, device, model, dataloader):
        model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                # Handle different model types
                if self.model_type=="vae":
                    # For VAE, extract only the reconstruction from the tuple
                    recon_outputs, _, _ = model(X_batch)
                    y_out = recon_outputs
                elif self.model_type == "diffusion":
                    # For diffusion models, we need to reconstruct the signal from predicted noise
                    # Since we're in eval mode, we'll use a simple reconstruction approach
                    # Sample timestep 0 (no noise) to get reconstruction
                    t = torch.zeros(X_batch.shape[0], device=device, dtype=torch.long)
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
        return np.concatenate(y_pred, axis=0)

