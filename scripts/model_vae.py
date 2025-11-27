import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------
# Fixed VAE Model Definition for EEG Reconstruction
# ----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE1D(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=64, seq_len=496):  # Reduced latent dim
        super(VAE1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.model_type = "vae"  # Add model type identifier
        
        # Encoder with smaller channels and better initialization
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # L/2
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # L/4
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # L/8
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate flattened size
        enc_out_size = seq_len // 8  # 496/8 = 62
        self.enc_fc = nn.Linear(128 * enc_out_size, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 128 * enc_out_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            # No activation - let the loss function handle it
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.enc_fc(h), 0.2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.leaky_relu(self.dec_fc(z), 0.2)
        h = F.leaky_relu(self.dec_fc2(h), 0.2)
        h = h.view(h.size(0), 128, self.seq_len // 8)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# Alternative simpler VAE for faster experimentation
class SimpleVAE1D(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=64, seq_len=496):
        super(SimpleVAE1D, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # L/2
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # L/4
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # L/8
            nn.ReLU(),
        )
        
        enc_out_size = seq_len // 8  # 496/8 = 62
        self.enc_fc = nn.Linear(128 * enc_out_size, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 128 * enc_out_size)
        
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(h.size(0), 128, self.seq_len // 8)
        return self.dec_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss function
class VAELoss(nn.Module):
    def __init__(self, reconstruction_loss='mse', beta=1.0, kl_weight=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.kl_weight = kl_weight  # Additional weight for KL divergence
        
        if reconstruction_loss == 'mse':
            self.recon_loss_fn = nn.MSELoss(reduction='mean')  # Changed to mean for better scaling
        elif reconstruction_loss == 'l1':
            self.recon_loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError("reconstruction_loss must be 'mse' or 'l1'")
    
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss (mean instead of sum for better scaling)
        recon_loss = self.recon_loss_fn(recon_x, x)
        
        # KL divergence with numerical stability
        # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with weights
        total_loss = recon_loss + self.beta * self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss

