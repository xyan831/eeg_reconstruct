import math
import numpy as np
from scipy import stats
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Stable Diffusion Components for 1D EEG
# ----------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)[:, :, None]
        
        # First block
        h = self.block1(x)
        
        # Add time embedding
        h = h + t_emb
        
        # Second block
        h = self.block2(h)
        
        # Residual connection
        return h + self.residual_conv(x)

# Simple UNet for EEG data with 26 channels
class SimpleUNet1DDiffusion(nn.Module):
    def __init__(self, in_channels=26, out_channels=26, base_channels=64, time_emb_dim=128):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.enc1 = ResidualBlock1D(in_channels, base_channels, time_emb_dim)
        self.down1 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)
        
        self.enc2 = ResidualBlock1D(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.enc3 = ResidualBlock1D(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = nn.Conv1d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = ResidualBlock1D(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder
        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = ResidualBlock1D(base_channels * 8, base_channels * 2, time_emb_dim)
        
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = ResidualBlock1D(base_channels * 4, base_channels, time_emb_dim)
        
        self.up1 = nn.ConvTranspose1d(base_channels, base_channels, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = ResidualBlock1D(base_channels * 2, base_channels, time_emb_dim)
        
        # Final output
        self.final_conv = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        x1 = self.enc1(x, t_emb)
        x2 = self.down1(x1)
        
        x2 = self.enc2(x2, t_emb)
        x3 = self.down2(x2)
        
        x3 = self.enc3(x3, t_emb)
        x4 = self.down3(x3)
        
        # Bottleneck
        x4 = self.bottleneck(x4, t_emb)
        
        # Decoder with skip connections
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, t_emb)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, t_emb)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, t_emb)
        
        # Final output
        x = self.final_conv(x)
        return x

# Diffusion Process with device handling
class EEGDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule - initialized on specified device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def to(self, device):
        """Move all tensors to specified device"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self
        
    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device)
    
    def noise_input(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # All tensors are already on the same device
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t, noise

# Stable Diffusion Model Wrapper with proper device handling
class StableDiffusionEEG(nn.Module):
    def __init__(self, in_channels=26, out_channels=26, timesteps=100):
        super().__init__()
        self.unet = SimpleUNet1DDiffusion(in_channels, out_channels)
        self.diffusion = EEGDiffusion(timesteps)
        self.timesteps = timesteps
        
    def to(self, device):
        """Override to ensure diffusion tensors are moved to device"""
        super().to(device)
        self.diffusion.to(device)
        return self
        
    def forward(self, x, noise=None):
        # Sample random timesteps
        batch_size = x.shape[0]
        t = self.diffusion.sample_timesteps(batch_size, x.device)
        
        # Add noise to input
        x_noisy, target_noise = self.diffusion.noise_input(x, t, noise)
        
        # Predict noise
        predicted_noise = self.unet(x_noisy, t)
        
        return predicted_noise, target_noise, t

    def sample(self, noise_shape, device, guidance_scale=3.0):
        """Generate samples from noise"""
        self.unet.eval()
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(noise_shape, device=device)
            
            # Reverse diffusion process
            for i in reversed(range(self.timesteps)):
                t = torch.full((noise_shape[0],), i, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.unet(x, t)
                
                # Update x using the reverse process
                alpha_t = self.diffusion.alphas[t].view(-1, 1, 1)
                alpha_bar_t = self.diffusion.alpha_bars[t].view(-1, 1, 1)
                beta_t = self.diffusion.betas[t].view(-1, 1, 1)
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise
            
            return x

# Loss Functions
class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predicted_noise, target_noise):
        return self.loss_fn(predicted_noise, target_noise)

# Loss Functions
class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predicted_noise, target_noise):
        return self.loss_fn(predicted_noise, target_noise)

# Loss Functions
class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predicted_noise, target_noise):
        return self.loss_fn(predicted_noise, target_noise)

class HybridLoss(nn.Module):
    def __init__(self, diffusion_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.perceptual_weight = perceptual_weight
        self.diffusion_loss = DiffusionLoss('l1')
        
    def forward(self, predicted_noise, target_noise, reconstructed=None, original=None):
        loss = self.diffusion_weight * self.diffusion_loss(predicted_noise, target_noise)
        
        if reconstructed is not None and original is not None:
            perceptual_loss = F.l1_loss(reconstructed, original)
            loss += self.perceptual_weight * perceptual_loss
            
        return loss

# Evaluation Metrics
class EEGReconstructionMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, original, reconstructed):
        """Calculate all metrics between original and reconstructed EEG signals"""
        original = original.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        metrics = {}
        
        # Basic reconstruction metrics
        metrics['mse'] = mean_squared_error(original.flatten(), reconstructed.flatten())
        metrics['mae'] = mean_absolute_error(original.flatten(), reconstructed.flatten())
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(original**2)
        noise_power = np.mean((original - reconstructed)**2)
        metrics['snr_db'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Pearson correlation
        corrs = []
        for i in range(original.shape[0]):  # Batch dimension
            for j in range(original.shape[1]):  # Channel dimension
                corr, _ = stats.pearsonr(original[i, j], reconstructed[i, j])
                corrs.append(corr)
        metrics['pearson_correlation'] = np.mean(corrs)
        
        # Frechet Distance (approximate for 1D signals)
        metrics['frechet_distance'] = self.calculate_frechet_distance(original, reconstructed)
        
        # Spectral metrics
        spectral_metrics = self.calculate_spectral_metrics(original, reconstructed)
        metrics.update(spectral_metrics)
        
        return metrics
    
    def calculate_frechet_distance(self, original, reconstructed):
        """Approximate Frechet distance for 1D signals"""
        # For simplicity, using Wasserstein distance as approximation
        from scipy.stats import wasserstein_distance
        
        wasserstein_dists = []
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                w_dist = wasserstein_distance(original[i, j], reconstructed[i, j])
                wasserstein_dists.append(w_dist)
        
        return np.mean(wasserstein_dists)
    
    def calculate_spectral_metrics(self, original, reconstructed):
        """Calculate spectral domain metrics"""
        from scipy.signal import welch
        
        metrics = {}
        
        # Power spectral density correlation
        original_psd = welch(original.flatten())[1]
        reconstructed_psd = welch(reconstructed.flatten())[1]
        
        min_len = min(len(original_psd), len(reconstructed_psd))
        psd_corr = np.corrcoef(original_psd[:min_len], reconstructed_psd[:min_len])[0, 1]
        metrics['psd_correlation'] = psd_corr
        
        # Dominant frequency preservation
        original_dom_freq = np.argmax(original_psd)
        reconstructed_dom_freq = np.argmax(reconstructed_psd)
        metrics['dominant_freq_error'] = abs(original_dom_freq - reconstructed_dom_freq)
        
        return metrics

# Updated Training Function
def train_diffusion_model(device, model, dataloader, criterion, optimizer, epochs=10, 
                         results_file="diffusion_results.txt"):
    
    with open(results_file, "w") as f:
        f.write("epoch,diffusion_loss,sampling_quality\n")
    
    metrics_calculator = EEGReconstructionMetrics()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass through diffusion model
            predicted_noise, target_noise, t = model(X_batch)
            
            # Calculate loss
            loss = criterion(predicted_noise, target_noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Generate samples for quality evaluation
        model.eval()
        with torch.no_grad():
            # Sample from the model
            sample_shape = (4, X_batch.shape[1], X_batch.shape[2])  # Small batch for evaluation
            generated_samples = model.sample(sample_shape, device)
            
            # Calculate metrics on generated samples
            # For demonstration, using random targets - in practice, use validation set
            dummy_targets = torch.randn_like(generated_samples)
            sample_metrics = metrics_calculator.calculate_all_metrics(dummy_targets, generated_samples)
            sampling_quality = sample_metrics['pearson_correlation']  # Use correlation as quality measure
        
        # Log results
        avg_loss = running_loss / len(dataloader)
        with open(results_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{sampling_quality:.6f}\n")
        
        print(f"Epoch {epoch+1}/{epochs}, Diffusion Loss: {avg_loss:.6f}, "
              f"Sampling Quality: {sampling_quality:.6f}")


